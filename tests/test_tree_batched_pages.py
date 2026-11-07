"""The experimental page-batched tree walk (v0.3.0/T4) stays set-equal to production.

The batched walk is not the default (it measured 1.16-1.86x the per-page walk's
time - slower - in the audited A/B evidence/receipts-v0.3.0/ab-tree-select.json;
launch savings lost to batch-max widths and extra copies), but it is kept behind
``LSA_TREE_IMPL=batched`` as evidence and a base for a future Triton beam. This
test holds it to gate C's bar - selection-set equality with the per-page walk -
so it cannot silently rot or drift away from the reference semantics.
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

from lsa_rocm_lab.port._tree import _tree_select_pages, tree_select_per_page  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402
from lsa_rocm_lab.port.pipeline import PortConfig, build_buckets_gpu, build_hierarchy_gpu  # noqa: E402


def _sets(lists: torch.Tensor, row: int) -> set:
    return {int(x) for x in lists[row].tolist() if x >= 0}


@gpu
def test_batched_walk_set_equals_per_page(monkeypatch):
    T = 2048
    torch.manual_seed(20261107)
    config = PortConfig(cluster_block_size=512, pool_size=4, bucket_size=8, bucket_cap=16,
                        top_p=1024, top_k=4096, window_size=512, sink_size=4,
                        node_bucket_size=4, top_level=3, node_cluster_block_size=64, beam_width=64)
    q = torch.randn(1, T, 8, 64, device="cuda") * 0.2
    k = torch.randn(1, T, 2, 64, device="cuda") * 0.2
    idx_q, idx_k = make_index_space(q, k)
    buckets = build_buckets_gpu(idx_k, config)
    hierarchy = build_hierarchy_gpu(idx_k, buckets, config)
    positions = torch.arange(T, device="cuda")
    from lsa_rocm_lab.port import pipeline

    per_lists, _ = pipeline.tree_select(idx_q, positions, buckets, hierarchy, config)
    monkeypatch.setenv("LSA_TREE_IMPL", "batched")
    bat_lists, _ = pipeline.tree_select(idx_q, positions, buckets, hierarchy, config)
    mismatched = [r for r in range(T) if _sets(per_lists, r) != _sets(bat_lists, r)]
    assert not mismatched, f"rows with selection-set mismatch: {mismatched[:8]}"


@gpu
def test_batched_walk_single_row_matches_per_page_row(monkeypatch):
    T = 2048
    torch.manual_seed(20261108)
    config = PortConfig(cluster_block_size=512, pool_size=4, bucket_size=8, bucket_cap=16,
                        top_p=1024, top_k=4096, window_size=512, sink_size=4,
                        node_bucket_size=4, top_level=3, node_cluster_block_size=64, beam_width=64)
    q = torch.randn(1, T, 8, 64, device="cuda") * 0.2
    k = torch.randn(1, T, 2, 64, device="cuda") * 0.2
    idx_q, idx_k = make_index_space(q, k)
    buckets = build_buckets_gpu(idx_k, config)
    hierarchy = build_hierarchy_gpu(idx_k, buckets, config)
    pos = torch.arange(T, device="cuda")
    from lsa_rocm_lab.port import pipeline

    per_lists, per_len = pipeline.tree_select(idx_q[-1:], pos[-1:], buckets, hierarchy, config)
    monkeypatch.setenv("LSA_TREE_IMPL", "batched")
    bat_lists, bat_len = pipeline.tree_select(idx_q[-1:], pos[-1:], buckets, hierarchy, config)
    # single-row decode: both impls take the same shapes -> exactly equal picks
    assert torch.equal(per_lists, bat_lists) and torch.equal(per_len, bat_len)
