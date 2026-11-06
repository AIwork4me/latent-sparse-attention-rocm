"""Matched kernel benchmarks on one Radeon GPU - dense vs block-sparse vs D-LSA Flat vs Tree.

Everything timed here runs on the GPU with synchronized clocks (CUDA events around
``torch.cuda.synchronize``): the dense SDPA baseline, the positional block-sparse
arm, and the port's LSA flat/tree pipelines. A row is only produced when both sides
of a comparison ran the same shape, dtype and workload; the speedup of a row is
``baseline_ms / candidate_ms`` and nothing else, per ``docs/PORT_CONTRACT.md``.

Stage breakdown (each timed separately, warm, and with compilation excluded by the
warmups): index build (regroup + pools + Triton Ward merge + members/centroids),
hierarchy build (Triton node-bucket pairing), select (GEMM + Triton top-k +
Triton expansion), and the sparse read (this port's Triton gather-attention).
"""

import statistics
import time

import torch

from lsa_rocm_lab.port import pipeline
from lsa_rocm_lab.port.gates import make_index_space
from lsa_rocm_lab.port.pipeline import PortConfig
from lsa_rocm_lab.port.read_kernel_tiled import sparse_read_auto as sparse_read


def sync():
    torch.cuda.synchronize()


class Clock:
    """Synchronized wall timing of GPU stages, warm - the compile lands in warmup, not the rows."""

    def __init__(self, warmup: int = 5, reps: int = 20):
        self.warmup = warmup
        self.reps = reps

    def time(self, fn) -> dict:
        for _ in range(self.warmup):
            fn()
        sync()
        samples = []
        for _ in range(self.reps):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        return summarize(samples)


def summarize(samples: list[float]) -> dict:
    """Median, p95, min, max, MAD of milliseconds - the contract's kernel metrics."""
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)
    mad = statistics.median([abs(x - median) for x in ordered])
    return {
        "samples": n,
        "p50_ms": median,
        "p95_ms": ordered[min(n - 1, int(n * 0.95))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mad_ms": mad,
    }


def stage_times(fn) -> dict:
    """One synchronized pass of a multi-stage callable - a breakdown snapshot, not a repetition."""
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return {"wall_ms": (time.perf_counter() - start) * 1000, "result": out}


def build_geometry(context: int, dtype, heads_q: int, heads_kv: int, dim: int, seed: int, device="cuda"):
    """The matched q/k/v every method reads - same seed, same shape, same dtype."""
    torch.manual_seed(seed)
    q = (torch.randn(1, context, heads_q, dim, device=device, dtype=torch.float32) * 0.2).to(dtype)
    k = (torch.randn(1, context, heads_kv, dim, device=device, dtype=torch.float32) * 0.2).to(dtype)
    v = (torch.randn(1, context, heads_kv, dim, device=device, dtype=torch.float32) * 0.2).to(dtype)
    return q, k, v


def bench_dense_prefill(q, k, v, clock: Clock) -> dict:
    """The dense causal baseline - torch SDPA, whatever ROCm attention it dispatches."""
    scale = q.shape[-1] ** -0.5
    fn = lambda: pipeline.dense_read(q, k, v, scale)
    return {"method": "dense_sdpa", "phase": "prefill", **clock.time(fn)}


def bench_dense_decode(q, k, v, clock: Clock, query_len: int = 1) -> dict:
    """Dense decode - the cache's last ``query_len`` rows against the whole cache.

    ``query_len=1`` is the serving decode step; ``dense_read`` gives it full
    attention over the cache (the v0.2.0 harness's ``is_causal=True`` masked it
    to key 0 - see gate G). ``query_len>1`` times a speculative-decoding chunk on
    the same absolute-position semantics.
    """
    scale = q.shape[-1] ** -0.5
    tail = q[:, -query_len:]
    fn = lambda: pipeline.dense_read(tail, k, v, scale)
    return {"method": "dense_sdpa", "phase": "decode", "query_len": query_len, **clock.time(fn)}


def lsa_index_build(q, k, config: PortConfig):
    """Index-space collapse + buckets: regroup, pools, Triton Ward merge, members, centroids."""
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    return idx_q, buckets


def bench_lsa(q, k, v, config: PortConfig, clock: Clock, tree: bool, phase: str, top_k: int | None = None) -> list[dict]:
    """The LSA arms - flat or tree, prefill or decode - each stage timed alone, then the step whole."""
    rows = []
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    rows.append({"method": "lsa_tree" if tree else "lsa_flat", "phase": phase, "stage": "index_build",
                 **clock.time(lambda: pipeline.build_buckets_gpu(idx_k, config))})
    hierarchy = None
    if tree:
        hierarchy = pipeline.build_hierarchy_gpu(idx_k, buckets, config)
        rows.append({"method": "lsa_tree", "phase": phase, "stage": "hierarchy_build",
                     **clock.time(lambda: pipeline.build_hierarchy_gpu(idx_k, buckets, config))})
    positions = torch.arange(q.shape[1], device=q.device)
    if phase == "decode":
        # A decode step reads one row: its query, its position, the whole cache
        positions = positions[-1:]
        idx_q = idx_q[-1:]

    if tree:
        select = lambda: pipeline.tree_select(idx_q, positions, buckets, hierarchy, config, top_k)
    else:
        select = lambda: pipeline.flat_select(idx_q, positions, buckets, config, top_k)
    lists, lengths = select()
    rows.append({"method": "lsa_tree" if tree else "lsa_flat", "phase": phase, "stage": "select",
                 **clock.time(select)})
    scale = q.shape[-1] ** -0.5
    if phase == "prefill":
        read = lambda: sparse_read(q, k, v, lists, lengths, scale)
    else:
        read = lambda: sparse_read(q[:, -1:], k, v, lists[-1:], lengths[-1:], scale)
    read()
    rows.append({"method": "lsa_tree" if tree else "lsa_flat", "phase": phase, "stage": "sparse_read",
                 **clock.time(read)})
    # The whole attention step, select + read, on the built index - what a layer pays per step
    def step():
        l, n = select()
        if phase == "prefill":
            return sparse_read(q, k, v, l, n, scale)
        return sparse_read(q[:, -1:], k, v, l[-1:], n[-1:], scale)
    rows.append({"method": "lsa_tree" if tree else "lsa_flat", "phase": phase, "stage": "select+read",
                 **clock.time(step)})
    return rows


def bench_block_sparse(q, k, v, config: PortConfig, clock: Clock, phase: str, top_k: int | None = None) -> list[dict]:
    """The positional block-sparse arm - position buckets, the same select and read."""
    from lsa_rocm_lab.port.gates import make_index_space

    idx_q, idx_k = make_index_space(q, k)
    rows = [{"method": "block_sparse", "phase": phase, "stage": "index_build",
             **clock.time(lambda: pipeline.position_buckets(idx_k, config))}]
    buckets = pipeline.position_buckets(idx_k, config)
    positions = torch.arange(q.shape[1], device=q.device)
    if phase == "decode":
        positions = positions[-1:]
        idx_q = idx_q[-1:]
    select = lambda: pipeline.flat_select(idx_q, positions, buckets, config, top_k)
    lists, lengths = select()
    rows.append({"method": "block_sparse", "phase": phase, "stage": "select", **clock.time(select)})
    scale = q.shape[-1] ** -0.5
    if phase == "prefill":
        read = lambda: sparse_read(q, k, v, lists, lengths, scale)
    else:
        read = lambda: sparse_read(q[:, -1:], k, v, lists[-1:], lengths[-1:], scale)
    read()
    rows.append({"method": "block_sparse", "phase": phase, "stage": "sparse_read", **clock.time(read)})

    def step():
        l, n = select()
        if phase == "prefill":
            return sparse_read(q, k, v, l, n, scale)
        return sparse_read(q[:, -1:], k, v, l[-1:], n[-1:], scale)
    rows.append({"method": "block_sparse", "phase": phase, "stage": "select+read", **clock.time(step)})
    return rows
