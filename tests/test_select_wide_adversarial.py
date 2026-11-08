"""Wide-row adversarial selection tests - the sampled-listing contract, pinned.

v0.4.0's independent audit left one HIGH finding open (MEASURED_SELECT_FUSION.md
§4.2): the port's ``best_first_keys`` drops upstream ``TopK.best()``'s
``~listed`` mask, so on a sampled listing (``columns > 4 * top_p``) it runs
top-k over key-buffer slots the listing kernel never wrote - uninitialized
memory. These tests pin the FIXED contract (mask unlisted slots to EMPTY
exactly as upstream does) and prove the two properties the fix must hold:

- BYTE-FREEZE on the whole path (``columns <= 4 * top_p``): masking is a no-op
  there (every slot is written, EMPTY past the row's length), so the v0.3.1
  oracle's outputs are bit-identical at every benchmarked shape.
- CORRECTNESS on the sampled path: outputs equal the exact best-k of the live
  columns (upstream guarantees the listing is a superset of the true best k,
  with repair), deterministic regardless of what garbage the unlisted slots
  hold, and byte-equal to the fused implementation's length-respecting
  fallback.

Plus the audit's open coverage gaps: kernel-B expansion at non-default bucket
capacities, and end-to-end wide-context (65536, columns > 4*top_p) select
byte-exactness through the implementation switch.

CPU-runnable cases (pure-torch masking semantics) execute everywhere incl.
GitHub-hosted CI; listing/kernel cases skip loudly when no ROCm device.
"""

import dataclasses
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa_rocm_lab.port import pipeline, select_fused  # noqa: E402
from lsa_rocm_lab.port.pipeline import PortConfig  # noqa: E402
from lsa.core.kernels.triton import keys as keys_mod  # noqa: E402
from lsa.core.kernels.triton.keys import EMPTY  # noqa: E402
from lsa.core.kernels.triton import top_k as triton_top_k  # noqa: E402


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _gpu_available() else [])
TIME = 20261108


def _live_packed(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Upstream pack of every column with EMPTY past each row's length (whole-path form)."""
    columns = torch.arange(scores.shape[1], device=scores.device)
    ids = torch.where(columns[None, :] < lengths[:, None], columns.expand_as(scores),
                      torch.tensor(-1, device=scores.device)).int()
    return keys_mod.pack_keys(scores, ids)


def _masked_want(keys: torch.Tensor, counts: torch.Tensor, top_p: int) -> torch.Tensor:
    """Upstream TopK.best() semantics: unlisted slots masked EMPTY, then top-k."""
    listed = torch.arange(keys.shape[1], device=keys.device)[None, :] < counts[:, None]
    return torch.topk(keys.masked_fill(~listed, EMPTY), k=min(top_p, keys.shape[1]),
                      dim=1, sorted=True).values


# --------------------------------------------- CPU + GPU: the masking contract


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,width,counts_kind", [
    (4, 4096, "ragged"),     # production whole-path width, partial rows
    (6, 2048, "zeros"),      # empty-slot rows (nothing listed)
    (8, 1024, "full"),
])
def test_unlisted_slots_are_masked_not_read(rows, width, counts_kind, device):
    """The fix: slots at or past a row's count must never be read - however
    large the garbage they hold. Old behavior (plain top-k over the row)
    returns the poison whenever it exceeds the k-th real key."""
    torch.manual_seed(TIME + width)
    scores = torch.randn(rows, width, device=device)
    if counts_kind == "full":
        counts = torch.full((rows,), width, device=device, dtype=torch.int32)
    elif counts_kind == "zeros":
        counts = torch.zeros(rows, device=device, dtype=torch.int32)
    else:
        g = torch.Generator(device="cpu").manual_seed(TIME + 1)
        counts = torch.sort(torch.randint(0, width + 1, (rows,), generator=g)).values
        counts = counts.to(device=device, dtype=torch.int32)
    keys = _live_packed(scores, counts.clamp_max(width).long())
    # poison the unlisted slots with values that beat every real key
    poison = torch.full_like(keys, torch.iinfo(torch.int64).max - 1)
    unlisted = torch.arange(width, device=device)[None, :] >= counts.long()[:, None]
    keys = torch.where(unlisted, poison, keys)
    got = pipeline.best_first_keys(keys, 1024, counts=counts)
    want = _masked_want(keys, counts, 1024)
    assert torch.equal(got, want), "unlisted slots leaked into the best-first keys"
    # rows that list nothing come back all EMPTY
    if counts_kind == "zeros":
        assert bool((got == EMPTY).all())
    # the ordering contract holds: every row non-increasing (EMPTY = int64 min sorts last)
    assert bool((got[:, :-1] >= got[:, 1:]).all())
    # and the hazard these tests exist for is real: the UNMASKED top-k returns poison
    # (only provable where unlisted slots exist)
    if bool((counts < width).any()):
        unmasked = torch.topk(keys, k=min(1024, width), dim=1, sorted=True).values
        assert not torch.equal(unmasked, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("columns,top_p", [
    (64, 16),      # gates geometry: columns == 4*top_p exactly
    (512, 1024),   # columns < top_p (short rows, k == width)
    (1024, 1024),  # k == width at production top_p
    (2048, 1024),  # mid ladder
    (4096, 1024),  # 32k-context production width, whole-path boundary
    (300, 128),    # non-power-of-two, k < next_pow2
])
def test_whole_path_byte_freeze(columns, top_p, device):
    """The oracle freeze: on the whole path (every benchmarked shape) masking
    changes nothing - old and fixed best_first_keys agree byte-for-byte."""
    torch.manual_seed(TIME + columns)
    rows = 6
    g = torch.Generator(device="cpu").manual_seed(TIME + 2)
    lengths = torch.tensor([columns, max(1, columns // 2), max(1, columns // 4), 1, 0, columns],
                           device=device, dtype=torch.int32)
    scores = torch.randn(rows, columns, device=device)
    keys = _live_packed(scores, lengths.long())
    old = torch.topk(keys, k=min(top_p, columns), dim=1, sorted=True).values
    new = pipeline.best_first_keys(keys, top_p, counts=lengths)
    assert torch.equal(old, new), "the fix changed whole-path outputs (oracle not frozen)"


# --------------------------------------------------- GPU: the sampled listing


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_sampled_listing_matches_exact_best_k(device):
    """columns > 4*top_p: upstream's listing (with its repair pass) is a
    superset of the true best k, so the masked best-first keys equal the exact
    best-k of the live columns."""
    torch.manual_seed(TIME + 3)
    rows, width, top_p = 8, 8192, 1024
    scores = torch.randn(rows, width, device=device)
    lengths = torch.tensor([width, width, 4096, 1000, 64, 8, 1, 0], device=device, dtype=torch.int32)
    listing = triton_top_k.TopK(top_p, rows=rows, exact=False)
    keys, counts = listing.listed(scores, lengths)
    assert width > 4 * top_p  # sampled contract really is exercised
    got = pipeline.best_first_keys(keys, top_p, counts=counts)
    exact = torch.topk(_live_packed(scores, lengths.long()), k=min(top_p, width),
                       dim=1, sorted=True).values
    assert torch.equal(got, exact)


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_reference_deterministic_under_arbitrary_garbage(device):
    """Whatever the allocator left in the unlisted slots, the fixed selection
    must be deterministic: pre-poison the held buffers with three different
    patterns and demand identical results."""
    torch.manual_seed(TIME + 4)
    rows, width, top_p = 8, 8192, 1024
    scores = torch.randn(rows, width, device=device)
    lengths = torch.tensor([width, width, 4096, 1000, 64, 8, 1, 0], device=device, dtype=torch.int32)
    results = []
    for pattern in (0, torch.iinfo(torch.int64).max - 1, -1234567890123456):
        listing = triton_top_k.TopK(top_p, rows=rows, exact=False)
        listing.held[0].fill_(pattern)  # the slots the kernels never write
        keys, counts = listing.listed(scores, lengths)
        results.append(pipeline.best_first_keys(keys, top_p, counts=counts))
    assert torch.equal(results[0], results[1]) and torch.equal(results[1], results[2]), (
        "selection depends on uninitialized memory contents"
    )


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("top_p", [256, 1024])
def test_fused_wide_fallback_equals_fixed_reference(top_p, device):
    """The fused length-respecting fallback and the FIXED reference now agree
    byte-exactly on sampled shapes - the divergence note is obsolete."""
    torch.manual_seed(TIME + 5 + top_p)
    rows, width = 8, 8192
    scores = torch.randn(rows, width, device=device)
    lengths = torch.tensor([width, width, 4096, 1000, 64, 8, 1, 0], device=device, dtype=torch.int32)
    listing = triton_top_k.TopK(top_p, rows=rows, exact=False)
    keys, counts = listing.listed(scores, lengths)
    ref = pipeline.best_first_keys(keys, top_p, counts=counts)
    with pytest.warns(UserWarning, match="sampled listing"):
        got = select_fused.select_topk_scores(scores, lengths, top_p)
    k = ref.shape[1]
    assert torch.equal(got[:, :k], ref)
    if got.shape[1] > k:
        assert bool((got[:, k:] == EMPTY).all())


# ------------------------------- GPU: expansion at non-default bucket capacities


def _synthetic_members(rows_buckets: int, cap: int, device, seed: int = TIME + 6):
    g = torch.Generator(device="cpu").manual_seed(seed)
    members = (torch.rand(rows_buckets * cap, generator=g) * 997).long().reshape(rows_buckets, cap)
    empty = torch.rand(rows_buckets, cap, generator=g) < 0.15
    members = torch.where(empty, torch.tensor(-1), members)
    return members.to(device, dtype=torch.int32)


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("cap", [8, 32, 64])  # 16 is the production default, covered elsewhere
def test_expand_bucket_cap_variants(cap, device):
    """Kernel B at non-default (power-of-two) bucket capacities: byte-exact vs
    the reference unpack+expand pair (closes the audit's open coverage gap).
    Non-power-of-two caps cannot form the walk tile (Triton arange constraint,
    shared with upstream's _expand_in_order) and must fail loudly instead."""
    torch.manual_seed(TIME + 7 + cap)
    rows, width, window, sink, top_k = 8, 512, 512, 4, 4096
    scores = torch.randn(rows, width, device=device)
    lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    listing = triton_top_k.TopK(min(1024, width), rows=rows, exact=False)
    keys, counts = listing.listed(scores, lengths)
    ordered = pipeline.best_first_keys(keys, min(1024, width), counts=counts)
    members = _synthetic_members(width, cap, device)
    positions = torch.tensor([4095, 511, 512, 513, 1024, 1, 0, 2047][:rows], device=device,
                             dtype=torch.int32)
    got_lists, got_lengths = select_fused.expand_in_order_from_keys(
        ordered, members, positions, window, sink, top_k)
    want_lists, want_lengths = _reference_expand(ordered, members, positions, window, sink, top_k)
    assert torch.equal(got_lists, want_lists), f"bucket_cap={cap}: read lists diverged"
    assert torch.equal(got_lengths, want_lengths), f"bucket_cap={cap}: lengths diverged"


def _reference_expand(ordered, members, positions, window, sink, top_k):
    picked = keys_mod.unpack_ids(ordered)
    from lsa.core.kernels.triton.expansion import expand_in_order

    return expand_in_order(picked, members, positions.int(), window, sink, top_k)


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_non_pow2_bucket_cap_fails_loudly(device):
    """A cap like 24 cannot form a power-of-two walk tile (Triton arange
    constraint, same as upstream): a clear ValueError beats a deep
    CompilationError mid-launch."""
    torch.manual_seed(TIME + 8)
    ordered = torch.full((4, 64), EMPTY, device=device, dtype=torch.int64)
    members = _synthetic_members(64, 24, device)
    positions = torch.full((4,), 64, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="power-of-two"):
        select_fused.expand_in_order_from_keys(ordered, members, positions, 64, 4, 128)


# ------------------------------------ GPU: end-to-end wide context through the switch


def _build_workspace(context: int, device: str = "cuda", seed: int = TIME):
    from lsa_rocm_lab.port.gates import make_index_space
    from lsa_rocm_lab.port.bench import build_geometry

    q, k, v = build_geometry(context, torch.bfloat16, seed=seed + context, heads_q=8, heads_kv=2, dim=64)
    config = PortConfig()
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    return q, k, v, idx_q, buckets, config


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("impl", ["fused", "fused_unpack"])
def test_wide_context_flat_select_byte_exact(impl, device):
    """65536 context: columns run past 4*top_p into the sampled contract for
    the first time END TO END - fused (fallback route) and fused_unpack must
    match the FIXED reference chain byte-for-byte, lists and lengths."""
    q, k, v, idx_q, buckets, config = _build_workspace(65536, device)
    positions = torch.arange(65536, device=device)[::4]  # ragged: columns 64..8128 in one tensor
    rows = idx_q[::4]
    os.environ["LSA_SELECT_IMPL"] = impl
    try:
        got_lists, got_lengths = pipeline.flat_select(rows, positions, buckets, config)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    want_lists, want_lengths = pipeline.flat_select_reference(rows, positions, buckets, config)
    assert torch.equal(got_lists, want_lists), f"{impl}: wide-context lists diverged"
    assert torch.equal(got_lengths, want_lengths), f"{impl}: wide-context lengths diverged"
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_whole_path_reference_is_the_v031_op_sequence(device):
    """The whole-path mask skip pinned: flat_select_reference (columns <= 4*top_p)
    must equal the exact v0.3.1 op sequence - listing + best_first_keys WITHOUT
    counts + unpack + expand - byte-for-byte, so the oracle's measured timings
    and allocations stay comparable to the published v0.3.1 receipts."""
    from lsa.core.kernels.triton import expansion as triton_expansion

    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    positions = torch.arange(2048, device=device)[::16]
    rows = idx_q[::16]
    assert rows.shape[1] <= 4 * config.top_p  # whole path really is taken
    got_lists, got_lengths = pipeline.flat_select_reference(rows, positions, buckets, config)
    scores, lengths = pipeline._production_scores(rows, positions, buckets, config, "bf16")
    listing = triton_top_k.TopK(config.top_p, exact=False)
    keys, _ = listing.listed(scores, lengths)
    picked = keys_mod.unpack_ids(pipeline.best_first_keys(keys, config.top_p))
    want_lists, want_lengths = triton_expansion.expand_in_order(
        picked, buckets.members, positions.int(), config.window_size, config.sink_size, config.top_k
    )
    assert torch.equal(got_lists, want_lists)
    assert torch.equal(got_lengths, want_lengths)
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("bucket_cap", [8, 32])
def test_flat_select_nondefault_bucket_cap_byte_exact(bucket_cap, device):
    """Non-default bucket capacity through the whole production composition."""
    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    config = dataclasses.replace(config, bucket_cap=bucket_cap)
    buckets = pipeline.build_buckets_gpu(idx_q, config)
    positions = torch.arange(2048, device=device)[::16]
    rows = idx_q[::16]
    os.environ["LSA_SELECT_IMPL"] = "fused"
    try:
        got_lists, got_lengths = pipeline.flat_select(rows, positions, buckets, config)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    want_lists, want_lengths = pipeline.flat_select_reference(rows, positions, buckets, config)
    assert torch.equal(got_lists, want_lists), f"bucket_cap={bucket_cap}: lists diverged"
    assert torch.equal(got_lengths, want_lengths), f"bucket_cap={bucket_cap}: lengths diverged"
    del q, k, v
    torch.cuda.empty_cache()
