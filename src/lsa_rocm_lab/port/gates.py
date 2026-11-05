"""Correctness gates - the port is only as true as these are strict.

Every gate runs the ROCm port path and the upstream pure-torch oracle on identical
inputs and reports a machine-readable verdict. Per ``docs/PORT_CONTRACT.md`` the
gates distinguish *exact set equality* (selection machinery vs oracle, scored in
fp32) from *numerical tolerance* (the sparse read vs the quadratic oracle) from
*intersection/recall statistics* (production bf16 scoring vs the fp32 oracle), and
never suppress a mismatch: a gate reports the numbers it measured and PASS/FAIL.

Gate index:
  A  bucket build        port Triton Ward vs ``reference.build_buckets``      partition equality + centroids
  A2 hierarchy build     port Triton pairing vs ``reference.build_hierarchy`` children equality
  B1 flat select         fp32-scored port select vs ``reference.read``        exact key-set equality per row
  B2 flat select (bf16)  production-scoring select vs oracle selection         key-set recall >= threshold
  C  tree select         port beam vs ``reference.read_hierarchy``             exact key-set equality per row
  D1 sparse read         port Triton read vs ``topk_attention`` oracle         allclose per dtype
  D2 full select         sparse read over all keys vs dense SDPA               allclose (the design's anchor)
  E  decode step         one decode position vs whole-prefill select+read      exact selection + tolerance
  F  GQA geometry        Hq != Hkv through ``index_space.regroup``             D1 on a grouped shape
"""

import json
import sys

import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.ward import cluster_node_buckets as triton_ward_cluster_node_buckets
from lsa_rocm_lab.port import HIP_PRECISION, pipeline, triton_ward
from lsa_rocm_lab.port.pipeline import PortConfig
from lsa_rocm_lab.port.read_kernel import sparse_read

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def make_index_space(q: torch.Tensor, k: torch.Tensor):
    """Collapse GQA query/key heads into the shared index space, upstream's ``regroup``.

    ``regroup`` reads ``[B, H, T, D]``; the benchmark tensors are ``[B, T, H, D]``.
    Returns (index_queries [T, d], index_keys [T, d]) of the first batch row.
    """
    from lsa.core.index_space import regroup

    index_queries, index_keys = regroup(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3))
    return index_queries[0], index_keys[0]


def set_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-row intersection statistics of two -1-padded id lists [R, W]."""
    a, b = a.long(), b.long()
    equal_rows = 0
    recalls, precisions = [], []
    for r in range(a.shape[0]):
        ra = set(a[r][a[r] >= 0].tolist())
        rb = set(b[r][b[r] >= 0].tolist())
        if ra == rb:
            equal_rows += 1
        if rb:
            recalls.append(len(ra & rb) / len(rb))
        if ra:
            precisions.append(len(ra & rb) / len(ra))
    n = a.shape[0]
    return {
        "rows": n,
        "rows_set_equal": equal_rows,
        "rows_set_equal_frac": equal_rows / max(n, 1),
        "recall_mean": sum(recalls) / len(recalls) if recalls else 1.0,
        "recall_min": min(recalls) if recalls else 1.0,
        "precision_mean": sum(precisions) / len(precisions) if precisions else 1.0,
    }


def _oracle_buckets(keys: torch.Tensor, config: PortConfig):
    """The torch oracle's buckets, on the CPU - its designed habitat (its internal
    ``torch.full``/``torch.empty`` default to CPU, so it cannot run on a GPU as-is).
    Cross-device ulp differences can flip near-ties; the gates measure that instead
    of hiding it (see ``set_stats``), and upstream's own near-tie test sets the
    precedent: < 2% co-membership drift with the Ward objective equal to 1e-5."""
    from lsa.core import reference

    return reference.build_buckets(
        keys.cpu().float(),
        config.pool_size,
        config.bucket_size,
        config.bucket_cap,
        config.cluster_block_size,
    )


def _oracle_buckets_on(device_buckets, device):
    """Put oracle CPU buckets on the GPU in the port's Buckets shape (both dtypes)."""
    return pipeline.Buckets(
        device_buckets.bucket_centroids.to(device, torch.bfloat16),
        device_buckets.bucket_members.to(device),
        device_buckets.bucket_sizes.to(device),
        centroids_f32=device_buckets.bucket_centroids.to(device),
    )


def gate_a_bucket_build(keys: torch.Tensor, config: PortConfig) -> dict:
    """A: the Triton Ward build's buckets vs the torch oracle's, on identical keys."""
    port = pipeline.build_buckets_gpu(keys, config)
    oracle = _oracle_buckets(keys, config)
    same_partition = True
    mismatch_row = -1
    pm, om = port.members.cpu().long(), oracle.bucket_members.long()
    for row in range(port.rows):
        pa = pm[row]
        pa = pa[pa >= 0]
        oa = om[row]
        oa = oa[oa >= 0]
        if not torch.equal(pa.sort().values, oa.sort().values):
            same_partition = False
            mismatch_row = row
            break
    centroid_err = float((port.centroids_f32.cpu() - oracle.bucket_centroids).abs().max())
    ok = same_partition and centroid_err < 1e-5
    return {
        "gate": "A bucket build (Triton Ward vs torch oracle)",
        "rows": port.rows,
        "partition_equal": bool(same_partition),
        "first_mismatch_row": mismatch_row,
        "centroid_max_abs_err": centroid_err,
        "status": "PASS" if ok else "FAIL",
    }


def _oracle_pairing_on_device(key_sums: torch.Tensor, node_bucket_size: int) -> torch.Tensor:
    """Upstream's pairing wrapper (``pair_into_node_buckets``) on the GPU, its ``ward_rounds``
    oracle the only algorithmic piece - the form upstream's own kernel test compares
    against with ``torch.equal``, device to device."""
    from lsa.core.ward import ward_rounds

    blocks, buckets, index_dim = key_sums.shape
    labels = torch.arange(buckets, device=key_sums.device).expand(blocks, buckets)
    sums = key_sums
    for _ in range(node_bucket_size.bit_length() - 1):
        count = sums.shape[1]
        pairs = ward_rounds(sums, torch.ones(blocks, count, device=sums.device), count // 2, 2)[0]
        labels = pairs.gather(1, labels)
        sums = torch.zeros(blocks, count // 2, index_dim, device=sums.device).scatter_add_(
            1, pairs[..., None].expand_as(sums), sums
        )
    return labels


def gate_a2_hierarchy_build(keys: torch.Tensor, config: PortConfig) -> dict:
    """A2: the Triton node-bucket pairing vs the torch oracle's, on the same device."""
    port_buckets = pipeline.build_buckets_gpu(keys, config)
    members = port_buckets.members.long()
    key_sums = (keys.float()[members.clamp_min(0)] * (members >= 0)[..., None]).sum(1)
    block_size = config.node_cluster_block_size
    status = "PASS"
    detail = {"levels": 0}
    while True:
        blocks = key_sums.shape[0] // block_size
        if not blocks or detail["levels"] >= config.top_level:
            break
        count = blocks * block_size
        view = key_sums[:count].view(blocks, block_size, -1).contiguous()
        want = _oracle_pairing_on_device(view, config.node_bucket_size)
        got, _ = pipeline._pair_to_size(view, config.node_bucket_size)
        if not torch.equal(got, want):
            status = "FAIL"
            detail["pairing_equal"] = False
            break
        order = (
            torch.argsort(got, dim=1, stable=True)
            + block_size * torch.arange(blocks, device=keys.device)[:, None]
        ).flatten()
        key_sums = key_sums[order].view(-1, config.node_bucket_size, key_sums.shape[1]).sum(1)
        detail["levels"] += 1
    detail["gate"] = "A2 hierarchy pairing (Triton vs torch, same device)"
    detail["status"] = status
    return detail


def gate_b_flat_select(keys, queries, positions, config: PortConfig, score_dtype: str, threshold: float = 0.99) -> dict:
    """B1 (fp32, exact) / B2 (bf16 production, recall): the port flat select vs ``reference.read``."""
    from lsa.core import reference

    oracle = _oracle_buckets(keys, config)
    selected, _ = reference.read(
        queries.cpu().float(), positions.cpu(), oracle, config.top_p, config.top_k,
        config.sink_size, config.window_size,
    )
    gpu_buckets = _oracle_buckets_on(oracle, keys.device)  # same bucket bits, on the GPU
    port_lists, _ = pipeline.flat_select(
        queries, positions, gpu_buckets, config, score_dtype=score_dtype
    )
    stats = set_stats(port_lists.cpu(), selected.cpu())
    if score_dtype == "fp32":
        name = "B1 flat select exact (fp32 scoring)"
        ok = stats["rows_set_equal"] == stats["rows"]
    else:
        # bf16 production scoring rounds scores near ties into different top-p picks
        # than the fp32 oracle - a measurement gate, with the worst row published
        name = "B2 flat select recall (bf16 production scoring; mean >= 0.999)"
        ok = stats["recall_mean"] >= 0.999 and stats["precision_mean"] >= 0.999
    return {"gate": name, **stats, "status": "PASS" if ok else "FAIL"}


def gate_c_tree_select(keys, queries, positions, config: PortConfig) -> dict:
    """C: the port's beam against the reference walk over the SAME tree.

    The CPU oracle walks the port's own GPU-built hierarchy (moved to the CPU), so
    the gate isolates the port's beam semantics from the pairing's build, which
    gate A2 holds to the torch oracle on the same device.
    """
    from lsa.core.hierarchy import read_hierarchy
    from lsa.core.reference import Buckets as ReferenceBuckets

    from lsa_rocm_lab.port._tree import to_reference

    gpu_buckets = pipeline.build_buckets_gpu(keys, config)
    gpu_hierarchy = pipeline.build_hierarchy_gpu(keys, gpu_buckets, config)
    ref_tree = to_reference(gpu_hierarchy, config)
    ref_tree = type(ref_tree)(
        [c.cpu() for c in ref_tree.node_centroids],
        [c.cpu() for c in ref_tree.children],
        [m.cpu() for m in ref_tree.node_means],
        ref_tree.bucket_means.cpu(),
        ref_tree.node_bucket_size,
        ref_tree.node_cluster_block_sizes,
    )
    ref_buckets = ReferenceBuckets(
        gpu_buckets.centroids_f32.cpu(),
        gpu_buckets.members.cpu(),
        gpu_buckets.sizes.cpu(),
        config.block_bucket_count,
        config.cluster_block_size,
    )
    walk_selected, _ = read_hierarchy(
        queries.cpu().float(), positions.cpu(), ref_buckets, ref_tree,
        config.beam_width, config.start_nodes, config.top_p, config.top_k,
        config.sink_size, config.window_size,
        config.cluster_block_size, config.block_bucket_count,
    )
    port_lists, _ = pipeline.tree_select(queries, positions, gpu_buckets, gpu_hierarchy, config)
    stats = set_stats(port_lists.cpu(), walk_selected)
    ok = stats["rows_set_equal"] == stats["rows"]
    return {"gate": "C tree select exact (port beam vs reference walk, same tree)",
            **stats, "status": "PASS" if ok else "FAIL"}


def _oracle_ids(lists: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Read lists with padding the quadratic oracle takes safely.

    ``topk_attention``'s ``keep_set`` scatters ``ids.clamp_min(0)`` with the value
    ``ids != PAD``: a PAD entry (-1) races a real id 0 at index 0, and every row's
    window contains id 0 - the scatter outcome is undefined on a GPU. Replacing
    padding with the row's own position is race-free (the scatter writes True at an
    index the window already made True) and causality-neutral, so the oracle reads
    exactly the listed keys.
    """
    rows = torch.arange(lists.shape[0], device=lists.device, dtype=torch.int32)[:, None]
    return torch.where(lists >= 0, lists, rows.expand_as(lists))


def gate_d_sparse_read(q, k, v, config: PortConfig) -> dict:
    """D1: the Triton sparse read vs upstream's quadratic ``topk_attention`` on identical lists."""
    from lsa.core.attention import topk_attention

    B, T, Hq, D = q.shape
    scale = D**-0.5
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    lists, lengths = pipeline.flat_select(idx_q, torch.arange(T, device=q.device), buckets, config)
    port = sparse_read(q, k, v, lists.unsqueeze(0), lengths.unsqueeze(0), scale)
    oracle = topk_attention(q, k, v, _oracle_ids(lists, lengths).unsqueeze(0).int(), window=0, sink=0, scale=scale)
    atol, rtol = TOLERANCES[q.dtype]
    err = float((port.float() - oracle.float()).abs().max())
    ok = bool(torch.allclose(port.float(), oracle.float(), atol=atol, rtol=rtol))
    return {
        "gate": f"D1 sparse read vs quadratic oracle ({q.dtype})",
        "max_abs_error": err,
        "atol": atol,
        "status": "PASS" if ok else "FAIL",
    }


def gate_d2_full_select(q, k, v) -> dict:
    """D2: the design's anchor - a full selection reproduces dense causal SDPA."""
    from lsa.core.attention import topk_attention

    B, T, Hq, D = q.shape
    scale = D**-0.5
    # Row t lists keys 0..t (causal by construction, as the expansion emits), padded
    # with t itself for the oracle's race (see _oracle_ids) and -1 for the kernel
    row = torch.arange(T, device=q.device, dtype=torch.int32)
    causal = row[None, :] <= row[:, None]  # key j visible to query t
    ids_kernel = torch.where(causal, row[None, :].expand(T, T), torch.tensor(-1, device=q.device, dtype=torch.int32))
    lengths = (row + 1).to(torch.int32)
    port = sparse_read(q, k, v, ids_kernel.view(1, T, T).contiguous(), lengths.view(1, T), scale)
    dense = pipeline.dense_read(q, k, v, scale)
    oracle_ids = torch.where(causal, row[None, :].expand(T, T), row[:, None].expand(T, T))
    oracle = topk_attention(q, k, v, oracle_ids.contiguous().view(1, T, T), window=0, sink=0, scale=scale)
    atol, rtol = TOLERANCES[q.dtype]
    err_sdpa = float((port.float() - dense.float()).abs().max())
    err_oracle = float((port.float() - oracle.float()).abs().max())
    ok = bool(torch.allclose(port.float(), dense.float(), atol=atol, rtol=rtol)) and bool(
        torch.allclose(port.float(), oracle.float(), atol=atol, rtol=rtol)
    )
    return {
        "gate": f"D2 full select == dense SDPA ({q.dtype})",
        "max_abs_vs_sdpa": err_sdpa,
        "max_abs_vs_quadratic_oracle": err_oracle,
        "atol": atol,
        "status": "PASS" if ok else "FAIL",
    }


def gate_e_decode_step(q, k, v, config: PortConfig) -> dict:
    """E: a decode step's select+read equals the whole-prefill row at the same position."""
    B, T, Hq, D = q.shape
    scale = D**-0.5
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    positions = torch.arange(T, device=q.device)
    whole_lists, whole_lengths = pipeline.flat_select(idx_q, positions, buckets, config)
    whole = sparse_read(q, k, v, whole_lists.unsqueeze(0), whole_lengths.unsqueeze(0), scale)
    last = T - 1
    step_lists, step_lengths = pipeline.flat_select(
        idx_q[last : last + 1], positions[last : last + 1], buckets, config
    )
    step = sparse_read(q[:, last : last + 1], k, v, step_lists.unsqueeze(0), step_lengths.unsqueeze(0), scale)
    selection_equal = bool(torch.equal(step_lists[0].cpu(), whole_lists[last].cpu()))
    err = float((step.float() - whole[:, last : last + 1].float()).abs().max())
    return {
        "gate": "E decode step == whole prefill row",
        "selection_equal": selection_equal,
        "max_abs_error": err,
        "status": "PASS" if selection_equal and err < 5e-3 else "FAIL",
    }


def gate_f_gqa(q, k, v, config: PortConfig) -> dict:
    """F: a grouped shape (Hq != Hkv) reads through the right KV head."""
    result = gate_d_sparse_read(q, k, v, config)
    return {"gate": "F GQA " + result["gate"], **{k_: v_ for k_, v_ in result.items() if k_ != "gate"}}


def run_all_gates(seed: int = 20261105) -> list[dict]:
    """Run the gate suite on one GPU; every gate's verdict is returned, none is swallowed."""
    device = "cuda"
    torch.manual_seed(seed)
    results = []
    # A small but fully nested geometry: block 128, 16 buckets of 8, top_p 16, top_k 64
    config = PortConfig(
        cluster_block_size=128, bucket_size=8, bucket_cap=16, top_p=16, top_k=64,
        window_size=128, sink_size=4, node_bucket_size=4, top_level=2,
        node_cluster_block_size=64, beam_width=4,
    )
    T, Hkv, D = 512, 2, 64
    Hq = 2 * Hkv
    q = torch.randn(1, T, Hq, D, device=device) * 0.2
    k = torch.randn(1, T, Hkv, D, device=device) * 0.2
    v = torch.randn(1, T, Hkv, D, device=device) * 0.2
    idx_q, idx_k = make_index_space(q, k)
    positions = torch.arange(T, device=device)

    results.append(gate_a_bucket_build(idx_k, config))
    results.append(gate_a2_hierarchy_build(idx_k, config))
    results.append(gate_b_flat_select(idx_k, idx_q, positions, config, "fp32"))
    results.append(gate_b_flat_select(idx_k, idx_q, positions, config, "bf16"))
    results.append(gate_c_tree_select(idx_k, idx_q, positions, config))
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        results.append(gate_d_sparse_read(q.to(dtype), k.to(dtype), v.to(dtype), config))
        results.append(gate_d2_full_select(q.to(dtype), k.to(dtype), v.to(dtype)))
    results.append(gate_e_decode_step(q.float(), k.float(), v.float(), config))
    q8 = torch.randn(1, T, 8, D, device=device) * 0.2
    results.append(gate_f_gqa(q8.bfloat16(), k.bfloat16(), v.bfloat16(), config))
    return results


if __name__ == "__main__":
    verdicts = run_all_gates()
    for line in verdicts:
        print(json.dumps(line), flush=True)
    sys.exit(0 if all(v["status"] == "PASS" for v in verdicts) else 1)
