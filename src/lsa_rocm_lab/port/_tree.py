"""The tree beam on the GPU - a vectorized walk of upstream's ``read_hierarchy``.

Upstream's reference beam is a per-query loop (``hierarchy.beam_pick``); the served
path runs it through the Triton beam kernels keyed to a page-managed cache. This
module is the port's middle road: the same walk, vectorized over the queries of one
growth (a page's queries all see the levels alike), in torch GPU ops, with upstream's
Triton ``expand_in_order`` for the expansion. ``gates.py`` holds it set-equal to the
reference walk, so what is timed is provably the same selection.
"""

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
    """Select by the beam, expand by the Triton kernel - see ``pipeline.tree_select``."""
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
