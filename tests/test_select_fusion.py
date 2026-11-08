"""v0.4.0 measured select fusion - byte-exact equivalence vs the v0.3.1 oracle.

The fused candidates in ``lsa_rocm_lab.port.select_fused`` (kernel A: pack +
order-giving top-k fused; the keys-reading expansion: unpack fused into the
walk) must reproduce the reference chain's read lists and lengths EXACTLY -
integer equality, not set equality - because the read list's order is part of
the algorithm's contract (mission B0/B3). These tests hold on every case the
mission lists: tied scores, EMPTY keys, k == width, short rows (fewer live
keys than k), ragged rows and B > 1, all-invalid rows, window/sink boundaries,
non-power-of-two widths, production widths, fp32/bf16 scoring, determinism,
and the GQA end-to-end read.

Kernel-level cases run on synthetic tensors at production WIDTHS (up to 4096
columns = the 32k-context shape) so width coverage does not need 32k
geometries; composition-level cases run real bucket builds at small/medium
contexts. Pure-contract cases (dispatch, wide-column fallback) run on CPU
everywhere - the GitHub-hosted CI executes them, never skips; the kernel cases
skip loudly when no ROCm device is visible.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa_rocm_lab.port import pipeline, select_fused  # noqa: E402
from lsa_rocm_lab.port.pipeline import PortConfig  # noqa: E402
from lsa.core.kernels.triton import keys as keys_mod  # noqa: E402


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _gpu_available() else [])
TIME = 20261108


# ---------------------------------------------------------------- CPU: contract


def test_switch_unknown_impl_fails_loudly():
    tiny = torch.randn(2, 8)
    config = PortConfig(top_p=4, top_k=8, window_size=4, sink_size=2, cluster_block_size=8,
                        bucket_size=2, bucket_cap=4, pool_size=2, node_bucket_size=2)
    from lsa_rocm_lab.port.pipeline import Buckets

    buckets = Buckets(tiny.bfloat16(), torch.zeros(8, 4, dtype=torch.int32) - 1,
                      torch.zeros(8, dtype=torch.int32), centroids_f32=tiny)
    os.environ["LSA_SELECT_IMPL"] = "bogus"
    try:
        with pytest.raises(ValueError, match="LSA_SELECT_IMPL"):
            pipeline.flat_select(tiny, torch.tensor([7, 7]), buckets, config)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)


def test_switch_default_is_reference(monkeypatch):
    monkeypatch.delenv("LSA_SELECT_IMPL", raising=False)
    assert pipeline.flat_select.__doc__ and "LSA_SELECT_IMPL" in pipeline.flat_select.__doc__


def test_wide_column_fallback_is_length_respecting():
    """columns > 4*top_p: deliberate visible fallback = upstream TopK.best()
    semantics (columns past a row's length are EMPTY, never selected)."""
    torch.manual_seed(TIME)
    scores = torch.randn(4, 64)
    lengths = torch.tensor([64, 40, 8, 0], dtype=torch.int32)
    top_p = 8  # 64 > 4*8 -> fallback branch, pure torch, runs on CPU
    with pytest.warns(UserWarning, match="sampled listing"):
        got = select_fused.select_topk_scores(scores, lengths, top_p)
    columns = torch.arange(64)
    ids = torch.where(columns[None, :] < lengths[:, None], columns.expand(4, 64),
                      torch.tensor(-1).expand(4, 64))
    keys = keys_mod.pack_keys(scores, ids.int())
    want = torch.topk(keys, k=min(top_p, 64), dim=1, sorted=True).values
    assert torch.equal(got, want)
    # the zero-visibility row must come back all EMPTY (no invisible bucket picked)
    assert bool((got[3] == keys_mod.EMPTY).all())


# ---------------------------------------------------------------- GPU: kernel A


def _reference_ordered(scores: torch.Tensor, lengths: torch.Tensor, top_p: int) -> torch.Tensor:
    """The exact reference chain up to ordered keys: upstream listing + best_first_keys."""
    from lsa.core.kernels.triton import top_k as triton_top_k

    listing = triton_top_k.TopK(top_p, exact=False)
    keys, listed = listing.listed(scores, lengths)
    return pipeline.best_first_keys(keys, top_p, counts=listed)


def _eq_k2(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Compare fused [R, K2] output vs reference [R, k]: first k columns equal, tail EMPTY."""
    k = want.shape[1]
    if not torch.equal(got[:, :k], want):
        return False
    return bool((got[:, k:] == keys_mod.EMPTY).all()) if got.shape[1] > k else True


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("rows,width,top_p,lengths_kind", [
    (64, 4096, 1024, "full"),    # production width (32k context), the wide row
    (8, 4096, 1024, "ragged"),   # same width, mixed visibility
    (64, 1024, 1024, "full"),    # k == width
    (7, 300, 128, "ragged"),     # non-power-of-two width, k < next_pow2
    (5, 192, 1024, "full"),      # k > width entirely (1536-context shape), heavy EMPTY tail
    (3, 64, 64, "zeros"),        # all-invalid rows (zero visible columns)
    (16, 2048, 512, "mix0"),     # some rows zero visibility mid-tensor
])
def test_kernel_a_equals_reference_chain(rows, width, top_p, lengths_kind, device):
    torch.manual_seed(TIME + width + rows)
    scores = torch.randn(rows, width, device=device)
    if lengths_kind == "full":
        lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    elif lengths_kind == "ragged":
        lengths = torch.randint(1, width + 1, (rows,), device=device, dtype=torch.int32)
    elif lengths_kind == "zeros":
        lengths = torch.zeros(rows, device=device, dtype=torch.int32)
    else:
        lengths = torch.randint(0, width + 1, (rows,), device=device, dtype=torch.int32)
    got = select_fused.select_topk_scores(scores, lengths, top_p)
    want = _reference_ordered(scores, lengths, top_p)
    assert _eq_k2(got, want), f"kernel A diverged at {rows}x{width} top_p={top_p} {lengths_kind}"


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_kernel_a_tied_scores_break_by_column(device):
    torch.manual_seed(TIME + 1)
    scores = torch.zeros(4, 512, device=device)           # every score tied
    lengths = torch.full((4,), 512, device=device, dtype=torch.int32)
    got = select_fused.select_topk_scores(scores, lengths, 128)
    want = _reference_ordered(scores, lengths, 128)
    assert _eq_k2(got, want)
    # and the tie order is lowest column first, both ways
    unpacked = keys_mod.unpack_ids(got[:, :128])
    assert torch.equal(unpacked, torch.arange(128, device=device).expand(4, 128).contiguous())


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_kernel_a_negative_and_denormal_scores(device):
    torch.manual_seed(TIME + 2)
    scores = torch.randn(8, 1024, device=device) * torch.tensor(
        [-1e30, -1e-30, 0.0, 1e-30, 1e30, -0.0, 1.0, -1.0], device=device)[:, None]
    lengths = torch.full((8,), 1024, device=device, dtype=torch.int32)
    got = select_fused.select_topk_scores(scores, lengths, 256)
    want = _reference_ordered(scores, lengths, 256)
    assert _eq_k2(got, want)


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_kernel_a_nan_scores_match_reference_bitcast_order(device):
    """NaN score bits ride the same ordered_bits map in both implementations:
    whatever order the bitcast gives, fused and reference must agree on it."""
    torch.manual_seed(TIME + 5)
    scores = torch.randn(4, 512, device=device)
    scores[0, ::7] = float("nan")
    scores[1] = float("nan")
    scores[2, :3] = float("-0.0")
    scores[3, ::5] = float("inf")
    lengths = torch.full((4,), 512, device=device, dtype=torch.int32)
    got = select_fused.select_topk_scores(scores, lengths, 128)
    want = _reference_ordered(scores, lengths, 128)
    assert _eq_k2(got, want)


# ------------------------------------------------------- GPU: expansion from keys


def _reference_expand(ordered: torch.Tensor, members: torch.Tensor, positions: torch.Tensor,
                      window: int, sink: int, top_k: int):
    picked = keys_mod.unpack_ids(ordered)
    from lsa.core.kernels.triton.expansion import expand_in_order

    return expand_in_order(picked, members, positions.int(), window, sink, top_k)


def _synthetic_members(rows_buckets: int, cap: int, device, seed: int = TIME + 3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    members = (torch.rand(rows_buckets * cap, generator=g) * 997).long().reshape(rows_buckets, cap)
    empty = torch.rand(rows_buckets, cap, generator=g) < 0.15
    members = torch.where(empty, torch.tensor(-1), members)
    return members.to(device, dtype=torch.int32)


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("rows,width,window,sink,top_k,positions_kind", [
    (1, 1024, 512, 4, 4096, "last"),        # decode B=1, production budget
    (32, 1024, 512, 4, 4096, "ragged"),     # multi-batch ragged decode
    (8, 256, 512, 4, 4096, "early"),        # positions inside one window: window_start=0
    (4, 256, 512, 4, 4096, "boundary"),     # positions exactly at window boundaries
    (4, 128, 512, 4, 64, "last"),           # tiny top_k: walk stops inside the first tiles
    (4, 128, 8, 2, 4096, "last"),           # tiny window and sink
    (2, 4096, 512, 4, 4096, "last"),        # production width walk
])
def test_expand_from_keys_equals_reference(rows, width, window, sink, top_k, positions_kind, device):
    torch.manual_seed(TIME + width + top_k)
    scores = torch.randn(rows, width, device=device)
    lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    ordered = _reference_ordered(scores, lengths, min(1024, width))
    members = _synthetic_members(width, 16, device)
    if positions_kind == "last":
        positions = torch.full((rows,), 4096, device=device, dtype=torch.int32)
    elif positions_kind == "ragged":
        positions = torch.tensor([1, 511, 512, 513, 1023, 1024, 2047, 4095][:rows],
                                 device=device, dtype=torch.int32)
    elif positions_kind == "early":
        positions = torch.arange(rows, device=device, dtype=torch.int32)
    else:
        positions = torch.tensor([511, 512, 513, 1024][:rows], device=device, dtype=torch.int32)
    got_lists, got_lengths = select_fused.expand_in_order_from_keys(
        ordered, members, positions, window, sink, top_k)
    want_lists, want_lengths = _reference_expand(ordered, members, positions, window, sink, top_k)
    assert torch.equal(got_lists, want_lists), "read lists diverged (order matters - integer equality)"
    assert torch.equal(got_lengths, want_lengths), "lengths diverged"


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_expand_from_keys_empty_picks_contribute_nothing(device):
    """Short row: ordered keys are an EMPTY tail - the walk must skip them exactly
    like the reference's -1 picks."""
    torch.manual_seed(TIME + 4)
    scores = torch.randn(2, 64, device=device)
    lengths = torch.tensor([3, 0], device=device, dtype=torch.int32)  # 3 live, 0 live
    got = select_fused.select_topk_scores(scores, lengths, 64)        # EMPTY-heavy tail
    members = _synthetic_members(64, 16, device)
    positions = torch.tensor([500, 500], device=device, dtype=torch.int32)
    got_lists, got_lengths = select_fused.expand_in_order_from_keys(got, members, positions, 64, 4, 128)
    want_lists, want_lengths = _reference_expand(got, members, positions, 64, 4, 128)
    assert torch.equal(got_lists, want_lists) and torch.equal(got_lengths, want_lengths)


# ------------------------------------------------- GPU: composition through the switch


def _build_workspace(context: int, device: str = "cuda", seed: int = TIME):
    from lsa_rocm_lab.port.gates import make_index_space
    from lsa_rocm_lab.port.bench import build_geometry

    q, k, v = build_geometry(context, torch.bfloat16, seed=seed + context, heads_q=8, heads_kv=2, dim=64)
    config = PortConfig()
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    return q, k, v, idx_q, buckets, config


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("context", [512, 1536, 2048, 8192])
@pytest.mark.parametrize("impl", ["fused", "fused_unpack"])
def test_switched_flat_select_is_byte_exact(context, impl, device):
    q, k, v, idx_q, buckets, config = _build_workspace(context, device)
    positions = torch.arange(context, device=device)
    os.environ["LSA_SELECT_IMPL"] = impl
    try:
        got_lists, got_lengths = pipeline.flat_select(idx_q, positions, buckets, config)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    want_lists, want_lengths = pipeline.flat_select_reference(idx_q, positions, buckets, config)
    assert torch.equal(got_lists, want_lists), f"{impl} lists diverged at ctx {context}"
    assert torch.equal(got_lengths, want_lengths), f"{impl} lengths diverged at ctx {context}"
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("impl", ["fused", "fused_unpack"])
def test_switched_flat_select_decode_and_ragged_exact(impl, device):
    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    positions = torch.arange(2048, device=device)
    for phase, pos in [
        ("decode_b1", positions[-1:]),
        ("decode_b1_position0", positions[:1]),
        ("ragged_b32", torch.tensor([0, 1, 511, 512, 513, 1024, 1536, 2045, 2047] * 3 + [2047, 2047, 2047, 2047, 2044],
                                    device=device)),
    ]:
        rows = idx_q[pos.long()] if phase.startswith("ragged") else idx_q[-1:]
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            got_lists, got_lengths = pipeline.flat_select(rows, pos, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
        want_lists, want_lengths = pipeline.flat_select_reference(rows, pos, buckets, config)
        assert torch.equal(got_lists, want_lists), f"{impl}/{phase}: lists diverged"
        assert torch.equal(got_lengths, want_lengths), f"{impl}/{phase}: lengths diverged"
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
@pytest.mark.parametrize("score_dtype", ["bf16", "fp32"])
def test_switched_flat_select_scoring_dtypes(score_dtype, device):
    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    positions = torch.arange(2048, device=device)[::16]  # a prefill-slice of rows
    rows = idx_q[::16]
    os.environ["LSA_SELECT_IMPL"] = "fused"
    try:
        got_lists, got_lengths = pipeline.flat_select(rows, positions, buckets, config,
                                                      score_dtype=score_dtype)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    want_lists, want_lengths = pipeline.flat_select_reference(rows, positions, buckets, config,
                                                              score_dtype=score_dtype)
    assert torch.equal(got_lists, want_lists) and torch.equal(got_lengths, want_lengths)
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_fused_deterministic_across_runs(device):
    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    positions = torch.arange(2048, device=device)[::8]
    rows = idx_q[::8]
    os.environ["LSA_SELECT_IMPL"] = "fused"
    try:
        first = pipeline.flat_select(rows, positions, buckets, config)
        for _ in range(3):
            again = pipeline.flat_select(rows, positions, buckets, config)
            assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    del q, k, v
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", [d for d in DEVICES if d == "cuda"])
def test_fused_select_plus_read_gqa_numerical_equivalence(device):
    """End-to-end: identical lists drive the sparse read - outputs within the
    standing gate tolerance (lists themselves are already integer-equal above)."""
    from lsa_rocm_lab.port.read_kernel_tiled import sparse_read_auto

    q, k, v, idx_q, buckets, config = _build_workspace(2048, device)
    scale = q.shape[-1] ** -0.5
    pos = torch.tensor([2047, 1536, 1023, 511], device=device, dtype=torch.int64)
    rows = idx_q[pos]
    os.environ["LSA_SELECT_IMPL"] = "fused"
    try:
        lists, lengths = pipeline.flat_select(rows, pos, buckets, config)
    finally:
        os.environ.pop("LSA_SELECT_IMPL", None)
    ref_lists, ref_lengths = pipeline.flat_select_reference(rows, pos, buckets, config)
    assert torch.equal(lists, ref_lists)
    got = sparse_read_auto(q[:, pos], k, v, lists, lengths, scale)
    want = sparse_read_auto(q[:, pos], k, v, ref_lists, ref_lengths, scale)
    assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)
    del q, k, v
    torch.cuda.empty_cache()
