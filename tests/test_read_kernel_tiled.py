"""Tiled sparse read (v0.3.0/T3) - correctness equals v1 across the edge matrix.

v2 tiles TM query rows per program but must keep v1's per-row numeric path
exactly (fp32 scores/softmax, -1 padding, GQA, per-occurrence duplicate
semantics) so any speedup is an implementation gain, never a semantic change.
Edge coverage per the plan: causal visibility, GQA/MQA, fp32/fp16/bf16, PAD
and boundary indices, duplicate ids, zero-length rows (v2 defines them as a
zero output - v1's NaN is the hazard the L2 gate pins), ragged tiles, batches,
decode rows. Equivalence is asserted for every (tile_rows, chunk) config the
A/B benchmark will use, so a config that changes semantics can never win it.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("LSA_SKIP_GPU_TESTS") == "1", reason="LSA_SKIP_GPU_TESTS=1"
)


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_available(), reason="no ROCm GPU visible to torch")

from lsa_rocm_lab.port.pipeline import PortConfig, dense_read  # noqa: E402
from lsa_rocm_lab.port.read_kernel import sparse_read  # noqa: E402
from lsa_rocm_lab.port.read_kernel_tiled import sparse_read_tiled  # noqa: E402

TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}
CONFIGS = [(1, 64), (2, 64), (4, 32), (4, 64), (8, 16), (8, 32)]


def geometry(dtype, T=512, Hq=8, Hkv=2, D=64, seed=20261106, B=1, Tk=None):
    torch.manual_seed(seed)
    Tk = Tk or T
    q = (torch.randn(B, T, Hq, D, device="cuda") * 0.2).to(dtype)
    k = (torch.randn(B, Tk, Hkv, D, device="cuda") * 0.2).to(dtype)
    v = (torch.randn(B, Tk, Hkv, D, device="cuda") * 0.2).to(dtype)
    return q, k, v


def config() -> PortConfig:
    return PortConfig(cluster_block_size=512, pool_size=4, bucket_size=8, bucket_cap=16,
                      top_p=1024, top_k=4096, window_size=512, sink_size=4,
                      node_bucket_size=4, top_level=3, node_cluster_block_size=64, beam_width=64)


def causal_lists(T, device, width=None):
    """Row t lists keys 0..t then -1 padding - full causal visibility (the D2 shape)."""
    row = torch.arange(T, device=device, dtype=torch.int32)
    ids = torch.where(row[None, :] <= row[:, None], row[None, :].expand(T, T),
                      torch.tensor(-1, device=device, dtype=torch.int32))
    lengths = row + 1
    if width is not None:
        ids = torch.cat([ids, torch.full((T, width - T), -1, device=device, dtype=torch.int32)], dim=1)
    return ids.view(1, T, -1).contiguous(), lengths.view(1, T).contiguous()


@gpu
class TestEquivalenceToV1:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("tile,chunk", CONFIGS)
    def test_real_selection_all_configs(self, dtype, tile, chunk):
        from lsa_rocm_lab.port.gates import make_index_space
        from lsa_rocm_lab.port import pipeline as pl

        T = 1024
        q, k, v = geometry(dtype, T=T)
        idx_q, idx_k = make_index_space(q, k)
        buckets = pl.build_buckets_gpu(idx_k, config())
        lists, lengths = pl.flat_select(idx_q, torch.arange(T, device="cuda"), buckets, config())
        lists, lengths = lists.unsqueeze(0), lengths.unsqueeze(0)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q, k, v, lists, lengths, scale)
        tiled = sparse_read_tiled(q, k, v, lists, lengths, scale, tile_rows=tile, chunk=chunk)
        atol, rtol = TOLERANCES[dtype]
        assert torch.allclose(base.float(), tiled.float(), atol=atol, rtol=rtol), \
            float((base.float() - tiled.float()).abs().max())

    def test_full_causal_matches_dense_sdpa(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            q, k, v = geometry(dtype, T=256)
            ids, lengths = causal_lists(256, "cuda")
            scale = q.shape[-1] ** -0.5
            tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=4, chunk=64)
            dense = dense_read(q, k, v, scale)
            atol, rtol = TOLERANCES[dtype]
            assert torch.allclose(tiled.float(), dense.float(), atol=atol, rtol=rtol), dtype

    @pytest.mark.parametrize("hq,hkv", [(8, 2), (4, 1), (2, 2)])
    def test_gqa_mqa_mapping(self, hq, hkv):
        q, k, v = geometry(torch.bfloat16, T=384, Hq=hq, Hkv=hkv)
        torch.manual_seed(7)
        ids = torch.randint(0, 384, (1, 384, 1024), device="cuda", dtype=torch.int32)
        lengths = torch.full((1, 384), 1024, device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q, k, v, ids, lengths, scale)
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=4, chunk=64)
        assert torch.allclose(base.float(), tiled.float(), atol=2e-2, rtol=2e-2)

    def test_batch_and_ragged_tile(self):
        B, T = 3, 13  # 13 rows: tiles of 4 leave a ragged 1-row tile
        q, k, v = geometry(torch.bfloat16, T=T, B=B)
        torch.manual_seed(11)
        ids = torch.randint(0, T, (B, T, 128), device="cuda", dtype=torch.int32)
        lengths = torch.randint(1, 129, (B, T), device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q, k, v, ids, lengths, scale)
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=4, chunk=32)
        assert torch.allclose(base.float(), tiled.float(), atol=2e-2, rtol=2e-2)

    def test_decode_single_row(self):
        q, k, v = geometry(torch.bfloat16, T=512)
        q1, ids1 = q[:, -1:], None
        torch.manual_seed(13)
        ids = torch.randint(0, 512, (1, 1, 4620), device="cuda", dtype=torch.int32)
        lengths = torch.full((1, 1), 4620, device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q1, k, v, ids, lengths, scale)
        tiled = sparse_read_tiled(q1, k, v, ids, lengths, scale, tile_rows=4, chunk=64)
        assert torch.allclose(base.float(), tiled.float(), atol=2e-2, rtol=2e-2)
        del ids1


@gpu
class TestEdgeSemantics:
    def test_pad_and_boundary_indices(self):
        q, k, v = geometry(torch.bfloat16, T=256)
        ids = torch.full((1, 256, 512), -1, device="cuda", dtype=torch.int32)
        ids[:, :, 0] = 0          # the very first key
        ids[:, :, 1] = 255        # the very last key
        ids[:, :, 2:] = -1        # everything else padding
        lengths = torch.full((1, 256), 3, device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q, k, v, ids, lengths, scale)
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=4, chunk=64)
        assert torch.allclose(base.float(), tiled.float(), atol=2e-2, rtol=2e-2)
        # and the two-key answer is a real mix of the two keys, not zeros
        assert tiled.float().abs().max() > 0

    def test_duplicate_ids_counted_per_occurrence(self):
        q, k, v = geometry(torch.bfloat16, T=128)
        ids = torch.full((1, 128, 8), -1, device="cuda", dtype=torch.int32)
        ids[:, 0, 0] = 5
        ids[:, 0, 1] = 5  # duplicate: v1 counts it twice (weight 2p / (2p))
        lengths = torch.full((1, 128), 2, device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        base = sparse_read(q, k, v, ids, lengths, scale)
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=2, chunk=8)
        assert torch.allclose(base[0, 0].float(), tiled[0, 0].float(), atol=1e-3, rtol=1e-3)
        # a duplicated key collapses to that key's value row in fp32 math,
        # per query head through its GQA group (Hq=8 -> Hkv=2)
        for h in range(8):
            assert torch.allclose(tiled[0, 0, h].float(), v[0, 5, h // 4].float(),
                                  atol=1e-2, rtol=1e-2), h

    def test_zero_length_rows_are_zero_not_nan(self):
        q, k, v = geometry(torch.bfloat16, T=64)
        ids = torch.full((1, 64, 128), -1, device="cuda", dtype=torch.int32)
        lengths = torch.zeros(1, 64, device="cuda", dtype=torch.int32)
        scale = q.shape[-1] ** -0.5
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale)
        assert torch.isfinite(tiled).all()
        assert tiled.float().abs().max() == 0.0
        base = sparse_read(q, k, v, ids, lengths, scale)
        assert torch.isnan(base).all(), "v1 documents the NaN hazard the tiled kernel fixes"

    def test_mixed_zero_and_real_rows(self):
        q, k, v = geometry(torch.bfloat16, T=64)
        torch.manual_seed(17)
        ids = torch.randint(0, 64, (1, 64, 32), device="cuda", dtype=torch.int32)
        lengths = torch.full((1, 64), 32, device="cuda", dtype=torch.int32)
        lengths[0, 1::2] = 0  # every other row reads nothing
        scale = q.shape[-1] ** -0.5
        tiled = sparse_read_tiled(q, k, v, ids, lengths, scale, tile_rows=4, chunk=32)
        assert torch.isfinite(tiled).all()
        assert tiled[0, 1::2].float().abs().max() == 0.0
        assert tiled[0, 0::2].float().abs().max() > 0
