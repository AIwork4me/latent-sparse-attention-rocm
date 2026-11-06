"""The tree beam on the GPU - a vectorized walk of upstream's ``read_hierarchy``.

Upstream's reference beam is a per-query loop (``hierarchy.beam_pick``); the served
path runs it through the Triton beam kernels keyed to a page-managed cache. This
module is the port's middle road: the same walk, vectorized over the queries of one
growth (a page's queries all see the levels alike), in torch GPU ops, with upstream's
Triton ``expand_in_order`` for the expansion. ``gates.py`` holds it set-equal to the
reference walk, so what is timed is provably the same selection.
"""

import os

import torch

from lsa.core.hierarchy import Growth, growth_at
from lsa.core.hierarchy import Hierarchy as ReferenceHierarchy

from lsa_rocm_lab.port import triton_expansion
from lsa_rocm_lab.port.pipeline import Buckets, Hierarchy, PortConfig


def to_reference(hierarchy: Hierarchy, config: PortConfig) -> ReferenceHierarchy:
    """Adapt the GPU hierarchy to upstream's counts-only ``growth_at`` reader."""
    return ReferenceHierarchy(
        [level.centroids for level in hierarchy.levels],
        [level.children for level in hierarchy.levels],
        [level.key_sums for level in hierarchy.levels],
        hierarchy.bucket_key_sums,
        config.node_bucket_size,
        tuple(config.node_cluster_block_size for _ in hierarchy.levels),
    )


def tree_select_batched(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    config: PortConfig,
    top_k: int | None = None,
):
    """Select by the beam, expand by the Triton kernel - see ``pipeline.tree_select``.

    v0.3.0/T4 outcome: the page-batched walk (``_tree_select_pages``, set-equal to
    the reference by gate C and kept behind ``LSA_TREE_IMPL=batched``) measured
    0.81-0.87x the per-page walk at 8k/32k prefill - the launch tax it removes is
    cheap because each launch is tiny, while batch-max candidate widths and the
    extra copies cost more than the loop saves. The production default stays the
    proven per-page walk; a true Triton per-level beam (upstream's layout) is the
    remaining lever, not torch-level batching.
    """
    impl = os.environ.get("LSA_TREE_IMPL", "per_page")
    top_k = config.top_k if top_k is None else top_k
    if impl == "batched":
        picked = _tree_select_pages(index_queries, positions, buckets, hierarchy, config)
        return triton_expansion.expand_in_order(
            picked.int(), buckets.members, positions.int(), config.window_size, config.sink_size, top_k
        )
    return tree_select_per_page(index_queries, positions, buckets, hierarchy, config, top_k)


def tree_select_per_page(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    config: PortConfig,
    top_k: int | None = None,
):
    """The v0.2.1 per-page python-loop walk - kept as the A/B reference (T4)."""
    top_k = config.top_k if top_k is None else top_k
    reference = to_reference(hierarchy, config)
    queries = index_queries.float()
    picked = torch.empty(queries.shape[0], config.top_p, dtype=torch.int64, device=queries.device)
    pages = positions // config.cluster_block_size
    page_buckets = config.block_bucket_count  # page_size == cluster_block_size
    for page in torch.unique(pages).tolist():
        rows = (pages == page).nonzero(as_tuple=True)[0]
        growth = growth_at(reference, page * page_buckets, page_buckets, page_buckets)
        picked[rows] = _beam_pick_rows(queries[rows], positions[rows], buckets, hierarchy, growth, config)
    # The picked rows through the same Triton expansion flat's read takes
    return triton_expansion.expand_in_order(
        picked.int(), buckets.members, positions.int(), config.window_size, config.sink_size, top_k
    )


def _row_ranges(start: int, stops: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ranges [start, stop) as a padded [rows, most] tensor and its validity mask."""
    most = max(int(stops.max()) - start, 0) if stops.numel() else 0
    if most <= 0:
        empty = torch.zeros(len(stops), 0, dtype=torch.int64, device=device)
        return empty, torch.zeros(len(stops), 0, dtype=torch.bool, device=device)
    span = torch.arange(start, start + most, device=device)
    values = span.expand(len(stops), -1).clone()
    valid = values < stops[:, None]
    return values.masked_fill(~valid, 0), valid


def _growth_replay(page_ids: list[int], config: PortConfig) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """made/closed/covered per page - ``growth_at``'s replay, one incremental pass.

    The reference replays growing at every page end up to each page's closed
    bucket count; successive pages only add, so one pass snapshots every page.
    Pure integer work on the CPU (no GPU launches) - it replaces P calls into
    ``growth_at`` with the same arithmetic, and gate C holds the results
    set-equal to the reference walk that used them.
    """
    size = config.node_bucket_size
    first_block = config.node_cluster_block_size
    page_buckets = config.block_bucket_count
    top_level = config.top_level
    covered, made, closed = 0, [0] * top_level, [0] * top_level
    covered_t, made_t, closed_t = [], [], []
    max_end = max(page_ids) * page_buckets if page_ids else 0
    page_set = sorted(set(page_ids))
    # page 0 closes no buckets - growth_at's replay range is empty for it, so
    # its snapshot is the zero state; snapshots then append in ascending page
    # end order (every multiple of page_buckets in range gets an iteration)
    if page_set and page_set[0] == 0:
        covered_t.append(covered)
        made_t.append(list(made))
        closed_t.append(list(closed))
    snap_ends = {p * page_buckets for p in page_set if p > 0}
    for end in range(page_buckets, max_end + 1, page_buckets):
        blocks = (end - covered) // first_block
        if blocks < max(1, page_buckets // first_block):
            if end in snap_ends:
                covered_t.append(covered); made_t.append(list(made)); closed_t.append(list(closed))
            continue
        covered += blocks * first_block
        made[0] += blocks * first_block // size
        for level in range(1, top_level):
            block = config.node_cluster_block_size
            taken = (made[level - 1] - closed[level - 1]) // block * block
            if not taken:
                break
            closed[level - 1] += taken
            made[level] += taken // size
        if end in snap_ends:
            covered_t.append(covered); made_t.append(list(made)); closed_t.append(list(closed))
    return covered_t, made_t, closed_t


def _tree_select_pages(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    config: PortConfig,
) -> torch.Tensor:
    """The whole beam as one batched op set per level, chunked over pages.

    The per-page walk's cost was its python page loop (T2: 19,940 launches/iter
    at 32k); here each level runs once over all pages, in chunks of
    ``PAGE_CHUNK`` so the gathered-centroid einsum intermediate stays bounded
    (the full [P, R, C, d] gather is multiple GiB at 32k and thrashes UMA).
    The per-row math - ascending candidates, stable descending argsort,
    first-occurrence dedup, lower-row ties - is the per-page walk's exactly;
    gate C holds the batched result set-equal to the reference walk.
    """
    device = index_queries.device
    queries = index_queries.float()
    pages_cpu = (positions // config.cluster_block_size).cpu()
    sort_idx = torch.argsort(pages_cpu, stable=True)
    sorted_pages = pages_cpu[sort_idx]
    unique_pages, counts = torch.unique_consecutive(sorted_pages, return_counts=True)
    P, R = len(unique_pages), int(counts.max())
    counts_d = counts.to(device)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)[:-1]]).to(device)
    row_index = starts[:, None] + torch.arange(R, device=device)[None, :]
    row_valid = torch.arange(R, device=device)[None, :] < counts_d[:, None]
    row_index = row_index.clamp_max(len(pages_cpu) - 1)
    page_rows = sort_idx.to(device)[row_index]  # [P, R] original row ids
    q = queries[page_rows]
    pos = positions[page_rows]
    visible = (pos // config.cluster_block_size * config.block_bucket_count).clamp_max(buckets.rows)

    covered_list, made_list, closed_list = _growth_replay(unique_pages.tolist(), config)
    n_levels = len(made_list[0]) if made_list else 0
    made_t = torch.tensor(made_list, device=device)
    closed_t = torch.tensor(closed_list, device=device)
    covered = torch.tensor(covered_list, device=device)
    levels_count = (made_t > 0).sum(1)
    beam_width = config.beam_width

    carried = torch.zeros(P, R, 0, dtype=torch.int64, device=device)
    carried_valid = torch.zeros(P, R, 0, dtype=torch.bool, device=device)

    def _chunked_level_step(level, made_l, first_open, active):
        """One beam level over all pages in chunks - the walk's per-level math."""
        nonlocal carried, carried_valid
        fresh_w = max(int((made_l - first_open).max().item()) if P else 0, 0)
        carried_w = carried.shape[2]
        if carried_w + fresh_w <= 0:
            return
        keep_cols = min(beam_width, carried_w + fresh_w)
        children = hierarchy.levels[level - 1].children
        new_carried = torch.empty(P, R, keep_cols * config.node_bucket_size,
                                  dtype=torch.int64, device=device)
        new_valid = torch.empty(P, R, keep_cols * config.node_bucket_size,
                                dtype=torch.bool, device=device)
        for c0 in range(0, P, PAGE_CHUNK):
            c1 = min(c0 + PAGE_CHUNK, P)
            acts = active[c0:c1, None, None]
            rows_ok = row_valid[c0:c1, :, None]
            if fresh_w > 0:
                fresh = first_open[c0:c1, None, None] \
                    + torch.arange(fresh_w, device=device)[None, None, :].expand(c1 - c0, R, fresh_w)
                fresh_valid = (torch.arange(fresh_w, device=device)[None, None, :]
                               < (made_l - first_open)[c0:c1, None, None]).expand(c1 - c0, R, fresh_w)
                cand = torch.cat([carried[c0:c1], fresh], dim=2)
                valid = torch.cat([carried_valid[c0:c1], fresh_valid & acts & rows_ok], dim=2)
            else:
                cand = carried[c0:c1].clone()
                valid = carried_valid[c0:c1].clone()
            cand, perm = torch.sort(cand, dim=2)
            valid = torch.gather(valid, 2, perm)
            scores = torch.einsum("prd,prnd->prn", q[c0:c1],
                                  hierarchy.levels[level - 1].centroids[cand])
            scores = _dedup_scores(cand.reshape(-1, cand.shape[2]),
                                   scores.reshape(-1, cand.shape[2]),
                                   valid.reshape(-1, cand.shape[2])).reshape(c1 - c0, R, -1)
            top_idx = scores.reshape(-1, scores.shape[2]).argsort(
                dim=1, descending=True, stable=True)[:, :keep_cols]
            picked = cand.reshape(-1, cand.shape[2]).gather(1, top_idx).reshape(c1 - c0, R, -1)
            new_carried[c0:c1] = children[picked.reshape(-1)].reshape(c1 - c0, R, -1)
            kept_ok = (picked >= 0).reshape(c1 - c0, R, keep_cols) \
                & active[c0:c1, None, None] & row_valid[c0:c1, :, None]
            # each kept node contributes node_bucket_size children columns
            new_valid[c0:c1] = kept_ok.repeat_interleave(config.node_bucket_size, dim=2)
        # pages not walking this level keep their carried (they enter at a lower
        # level or never enter; their carried is empty/invalid either way, so
        # the where matters only when widths happen to coincide)
        if carried.shape[2] == new_carried.shape[2] and bool((~active).any()):
            carried = torch.where(active[:, None, None], new_carried, carried)
            carried_valid = torch.where(active[:, None, None], new_valid, carried_valid)
        else:
            carried, carried_valid = new_carried, new_valid

    for level in range(n_levels, 0, -1):
        active = levels_count >= level
        if not bool(active.any()):
            continue
        entering = levels_count == level
        first_open = torch.where(entering, torch.zeros_like(closed_t[:, level - 1]),
                                 closed_t[:, level - 1])
        _chunked_level_step(level, made_t[:, level - 1], first_open, active)

    # Level 0 for every page: carried children plus the uncovered visible buckets.
    # No pad to top_p - padding tripled the sort/dedup width (top_p 1024 vs the
    # natural ~320 at 32k); the gather takes min(width, top_p) columns instead.
    fresh, fresh_valid = _row_ranges_batched(covered, visible)
    cand = torch.cat([carried, fresh], dim=2)
    valid = torch.cat([carried_valid, fresh_valid & row_valid[:, :, None]], dim=2)
    cand, perm = torch.sort(cand, dim=2)
    valid = torch.gather(valid, 2, perm)
    width0 = min(cand.shape[2], config.top_p)
    picked_final = torch.full((P, R, config.top_p), -1, dtype=torch.int64, device=device)
    centroids = buckets.centroids_f32
    for c0 in range(0, P, PAGE_CHUNK):
        c1 = min(c0 + PAGE_CHUNK, P)
        chunk_cand, chunk_valid = cand[c0:c1], valid[c0:c1]
        scores = torch.einsum("prd,prnd->prn", q[c0:c1], centroids[chunk_cand])
        scores = _dedup_scores(chunk_cand.reshape(-1, chunk_cand.shape[2]),
                               scores.reshape(-1, scores.shape[2]),
                               chunk_valid.reshape(-1, chunk_valid.shape[2])).reshape(c1 - c0, R, -1)
        order0 = scores.reshape(-1, scores.shape[2]).argsort(
            dim=1, descending=True, stable=True)[:, :width0]
        chosen = chunk_cand.reshape(-1, chunk_cand.shape[2]).gather(1, order0).reshape(c1 - c0, R, -1)
        # an invalid column (pad/covered/out-of-range) is the walk's "not
        # picked": -1, never its placeholder id (bucket 0 would expand!)
        chosen_ok = chunk_valid.reshape(-1, chunk_valid.shape[2]).gather(1, order0).reshape(c1 - c0, R, -1)
        picked_final[c0:c1, :, :width0] = torch.where(chosen_ok, chosen, torch.full_like(chosen, -1))
    picks = torch.full((queries.shape[0], config.top_p), -1, dtype=torch.int64, device=device)
    picked_rows = page_rows.reshape(-1)
    picked_ok = row_valid.reshape(-1)
    picks[picked_rows[picked_ok]] = picked_final.reshape(-1, config.top_p)[picked_ok]
    return picks


PAGE_CHUNK = 4


def _row_ranges_batched(starts: torch.Tensor, stops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[P, R] per-row ranges [start_p, stop_pr) padded to the widest - ``_row_ranges`` batched."""
    most = max(int((stops - starts[:, None]).max()), 0)
    span = torch.arange(most, device=starts.device)
    values = (starts[:, None, None] + span[None, None, :]).expand(stops.shape[0], stops.shape[1], most)
    valid = span[None, None, :] < (stops[:, :, None] - starts[:, None, None])
    return values.masked_fill(~valid, 0), valid


def _dedup_scores(cand: torch.Tensor, scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mask out-of-range and duplicate candidate columns so one top-k keeps each row once."""
    order, _ = torch.sort(cand, dim=1, stable=True)
    dup_sorted = torch.cat(
        [torch.zeros_like(order[:, :1], dtype=torch.bool), order[:, 1:] == order[:, :-1]], dim=1
    )
    # Back to the original column order: a column is a duplicate if its value appeared before it
    ranks = order.argsort(1)
    dup = dup_sorted.gather(1, ranks)
    return scores.masked_fill(~valid | dup, float("-inf"))


def _beam_pick_rows(
    queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    growth: Growth,
    config: PortConfig,
) -> torch.Tensor:
    """The beam of every row of one growth - ``hierarchy.beam_pick`` vectorized.

    Each level: candidates are the carried children plus the level's open node
    buckets (its whole start level when the beam starts there), scored by centroid
    dot, the best ``beam_width`` kept. At level 0 the carried children plus the
    uncovered buckets pick ``top_p`` - a tie going to the lower row, as the packed
    sort order upstream uses does. Reference semantics: candidates sorted, dedup by
    first occurrence.
    """
    rows = queries.shape[0]
    levels = sum(1 for count in growth.made if count)
    carried = torch.zeros(rows, 0, dtype=torch.int64, device=queries.device)
    carried_valid = torch.zeros(rows, 0, dtype=torch.bool, device=queries.device)
    beam_width = config.beam_width
    assert config.start_nodes is None, "the exact-parity walk covers start_nodes=None (the top level down)"

    for level in range(levels, 0, -1):
        made, closed = growth.made[level - 1], growth.closed[level - 1]
        # The start level whole when a level above covers some of it, else its open node buckets
        first_open = 0 if level == levels else closed
        fresh = torch.arange(first_open, made, device=queries.device).expand(rows, -1)
        # Candidates ascending, as the reference sorts them - a tie goes to the lower row
        cand, _ = torch.sort(torch.cat([carried, fresh], dim=1), dim=1)
        valid = torch.cat([carried_valid, torch.ones_like(fresh, dtype=torch.bool)], dim=1)
        scores = torch.einsum("rd,rnd->rn", queries, hierarchy.levels[level - 1].centroids[cand])
        scores = _dedup_scores(cand, scores, valid)
        kept = scores.argsort(dim=1, descending=True, stable=True)[:, : min(beam_width, cand.shape[1])]
        picked_nodes = torch.gather(cand, 1, kept)
        children = hierarchy.levels[level - 1].children
        carried = children[picked_nodes].reshape(rows, -1)
        carried_valid = torch.ones_like(carried, dtype=torch.bool)

    # Level 0: the carried children plus every uncovered visible bucket, picked to top_p.
    # A query before any growth sees no candidates at all - it reads only its window
    # and sinks, as the reference's beam_pick returns empty and expand takes the tail.
    visible = (positions // config.cluster_block_size * config.block_bucket_count).clamp_max(buckets.rows)
    fresh, fresh_valid = _row_ranges(growth.covered_buckets, visible, queries.device)
    picks = torch.full((rows, config.top_p), -1, dtype=torch.int64, device=queries.device)
    if carried.shape[1] == 0 and fresh.shape[1] == 0:
        return picks
    cand, _ = torch.sort(torch.cat([carried, fresh], dim=1), dim=1)
    valid = torch.cat([carried_valid, fresh_valid], dim=1)
    scores = torch.einsum("rd,rnd->rn", queries, buckets.centroids_f32[cand])
    scores = _dedup_scores(cand, scores, valid)
    order = scores.argsort(dim=1, descending=True, stable=True)[:, : config.top_p]
    chosen = cand.gather(1, order)
    picks[:, : chosen.shape[1]] = chosen
    return picks
