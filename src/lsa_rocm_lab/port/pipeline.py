"""The LSA pipelines on ROCm - bucket build, flat/tree select, expansion, sparse read.

Every stage either calls an upstream Triton kernel (ward merge, node-bucket pairing,
top-k listing, expansion), a torch GPU op the production path also uses (the bf16
centroid GEMM with fp32 out, normalize/index_add), or this port's Triton sparse-read
kernel. Nothing here falls back to a CPU loop or a dense mask silently: the flat
select at benchmark scale is the production short/medium-context composition (dense
GEMM + Triton top-k + Triton expansion), and the tree beam is a GPU-torch
vectorization of upstream's ``read_hierarchy`` walk, gated set-equal against it.

Provenance: algorithm and Triton kernels are upstream Apache-2.0 at the pinned SHA
(see ``lsa_rocm_lab.port``); the sparse read kernel and this composition are this
project's, also Apache-2.0.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lsa_rocm_lab.port import (
    HIP_PRECISION,
    triton_expansion,
    triton_top_k,
    triton_ward,
)
from lsa_rocm_lab.port.read_kernel_tiled import sparse_read_auto as sparse_read
from lsa.core.kernels.triton.keys import unpack_ids


@dataclass(frozen=True)
class PortConfig:
    """The design's knobs, upstream's defaults first, the benchmark's variations beside them."""

    cluster_block_size: int = 512  # = window: a block closes when the window slides past it
    pool_size: int = 4
    bucket_size: int = 8
    bucket_cap: int = 16
    top_p: int = 1024
    top_k: int = 4096
    window_size: int = 512
    sink_size: int = 4
    node_bucket_size: int = 4
    top_level: int = 3
    node_cluster_block_size: int = 64
    # The tree's beam; production GLM runs width 2048 over 16 levels at million-key
    # contexts - benchmark scales run the same walk proportionally smaller
    beam_width: int = 64
    start_nodes: int | None = None

    @property
    def block_bucket_count(self) -> int:
        return self.cluster_block_size // self.bucket_size


@dataclass
class Buckets:
    """A cache's buckets on the GPU, the port-side twin of upstream's reference ``Buckets``."""

    centroids: torch.Tensor  # [rows, index_dim] bf16 - what production scoring reads
    members: torch.Tensor  # [rows, bucket_cap] int32, absolute positions, -1 padded
    sizes: torch.Tensor  # [rows] int32
    centroids_f32: torch.Tensor | None = None  # [rows, index_dim] fp32 - the exactness gates' oracle-parity copy

    @property
    def rows(self) -> int:
        return self.centroids.shape[0]


def build_buckets_gpu(index_keys: torch.Tensor, config: PortConfig) -> Buckets:
    """Cluster every closed cluster block into buckets - pools, the Triton Ward merge, members, centroids.

    Mirrors ``lsa.core.reference.build_buckets`` stage for stage with the GPU kernels
    in the upstream Triton modules; the only numeric difference is the Gram dot at
    ``ieee`` fp32 (the torch oracle is fp32 throughout, so this is exact-er than
    upstream's tf32x3 default, which is NVIDIA-only).

    Args:
        index_keys: Unit-row index keys [T, index_dim] fp32 or bf16.
        config: The design.

    Returns:
        The buckets of the ``T // cluster_block_size`` closed blocks.
    """
    keys = index_keys.float()
    block, cap, bpb = config.cluster_block_size, config.bucket_cap, config.block_bucket_count
    blocks = keys.shape[0] // block
    index_dim = keys.shape[1]
    closed = keys[: blocks * block].view(blocks, block, index_dim)

    # pools [G, P, d] and their sizes; pool_size divides the block by config validation
    pools = closed.reshape(blocks, block // config.pool_size, config.pool_size, index_dim).sum(2)
    pool_sizes = torch.full(pools.shape[:2], float(config.pool_size), device=keys.device)

    # The Triton Ward merge - upstream kernel, ieee Gram dot on ROCm
    labels = triton_ward.ward_merge(pools, pool_sizes, bpb, cap, HIP_PRECISION)[0]

    # Key-level labels [G, block], then the member table [G * bpb, cap] in block-local positions
    key_labels = labels.repeat_interleave(config.pool_size, dim=1)
    members = triton_ward.bucket_members_from_labels(key_labels, bpb, cap)
    # Block-local made absolute, as the reference does (padding stays -1)
    offsets = (torch.arange(blocks, device=keys.device) * block).int()[:, None, None]
    members = torch.where(members >= 0, members + offsets, members).reshape(blocks * bpb, cap)

    sizes = (members >= 0).sum(1, dtype=torch.int32)

    # Centroids: the normalized sum of each bucket's index keys (equal to the pool-key
    # sum path the reference takes), kept bf16 as the kernels' scoring reads them
    bucket_of_key = key_labels + bpb * torch.arange(blocks, device=keys.device)[:, None]
    sums = torch.zeros(blocks * bpb, index_dim, device=keys.device)
    sums.index_add_(0, bucket_of_key.flatten(), closed.reshape(-1, index_dim))
    unit = F.normalize(sums, dim=-1)
    centroids = unit.to(torch.bfloat16)

    return Buckets(centroids, members, sizes, centroids_f32=unit)


@dataclass
class NodeLevel:
    """One hierarchy level's node buckets on the GPU."""

    centroids: torch.Tensor  # [N, index_dim] fp32
    children: torch.Tensor  # [N, node_bucket_size] int64 - rows of the level below
    key_sums: torch.Tensor  # [N, index_dim] fp32 (for the level above's pairing)


@dataclass
class Hierarchy:
    """The node-bucket levels over a cache's buckets, built with the Triton pairing kernel."""

    levels: list[NodeLevel] = field(default_factory=list)
    bucket_key_sums: torch.Tensor | None = None  # [rows, index_dim] fp32, the pairing input


def _pair_to_size(sums: torch.Tensor, size: int, precision: str = HIP_PRECISION):
    """Pair a level's key sums into groups of exactly ``size`` - the Triton pairing
    kernel iterated ``log2(size)`` times, as production's ``pair_into_node_buckets``
    iterates it (one kernel call is one halving to pairs)."""
    blocks, pools, index_dim = sums.shape
    labels = torch.arange(pools, device=sums.device).expand(blocks, pools)
    running = sums.contiguous()
    for _ in range(size.bit_length() - 1):
        count = running.shape[1]
        pairs = triton_ward.cluster_node_buckets(running, precision)
        labels = pairs.gather(1, labels)
        running = torch.zeros(blocks, count // 2, index_dim, device=sums.device).scatter_add_(
            1, pairs[..., None].expand_as(running), running
        )
    return labels, running


def build_hierarchy_gpu(index_keys: torch.Tensor, buckets: Buckets, config: PortConfig) -> Hierarchy:
    """Pair every full node cluster block of every level - upstream's Triton node-bucket pairing.

    Mirrors ``lsa.core.hierarchy.build_hierarchy``: a level's blocks of key sums pair
    by the Ward cost into node buckets of exactly ``node_bucket_size`` children, whose
    centroid is the normalized sum beneath them. Children rows are in node-bucket
    order, block after block, as the reference numbers them.
    """
    keys = index_keys.float()
    members = buckets.members.long()
    key_sums = (keys[members.clamp_min(0)] * (members >= 0)[..., None]).sum(1)
    hierarchy = Hierarchy(bucket_key_sums=key_sums)
    size, block_size = config.node_bucket_size, config.node_cluster_block_size
    while True:
        blocks = key_sums.shape[0] // block_size
        if not blocks or len(hierarchy.levels) >= config.top_level:
            break
        count = blocks * block_size
        labels, node_sums = _pair_to_size(key_sums[:count].view(blocks, block_size, -1), size)
        # Rows in the order of their node buckets, then block after block (the reference's order)
        order = (
            torch.argsort(labels, dim=1, stable=True)
            + block_size * torch.arange(blocks, device=keys.device)[:, None]
        ).flatten()
        key_sums = node_sums.reshape(-1, key_sums.shape[1])
        hierarchy.levels.append(
            NodeLevel(F.normalize(key_sums, dim=-1), order.view(-1, size), key_sums.clone())
        )
    return hierarchy


def visible_counts(positions: torch.Tensor, buckets: Buckets, config: PortConfig) -> torch.Tensor:
    """Buckets each query sees - the closed blocks before its own, as the reference's read takes."""
    return (positions // config.cluster_block_size * config.block_bucket_count).clamp_max(buckets.rows).int()


def flat_select(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    config: PortConfig,
    top_k: int | None = None,
    score_dtype: str = "bf16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each query's keys - the production short/medium-context composition.

    The bf16 centroid GEMM with fp32 out (as the served path), upstream's Triton
    top-k listing, a sort of the packed keys to best-first order (their int64 layout
    is order-preserving), and upstream's Triton rank-order expansion.
    """
    top_k = config.top_k if top_k is None else top_k
    if positions.shape[0] != index_queries.shape[0]:
        raise ValueError(
            f"a position a query: {index_queries.shape[0]} queries, {positions.shape[0]} positions"
        )
    counts = visible_counts(positions, buckets, config)
    if score_dtype == "fp32":
        # Bit-parity scoring for the exactness gate: fp32 queries on the fp32 centroids
        scores = torch.mm(index_queries.float(), buckets.centroids_f32.t())
    else:
        queries = index_queries.to(torch.bfloat16)
        # fp32 straight out of the GEMM, as the served path asks (a bf16 product cast
        # after costs five times as much, per upstream)
        scores = torch.mm(queries, buckets.centroids.t(), out_dtype=torch.float32)
    listing = triton_top_k.TopK(config.top_p, exact=False)
    keys, _ = listing.listed(scores, counts)
    keys = torch.sort(keys, dim=1, descending=True).values
    picked = unpack_ids(keys[:, : config.top_p])
    return triton_expansion.expand_in_order(
        picked, buckets.members, positions.int(), config.window_size, config.sink_size, top_k
    )


def tree_select(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    config: PortConfig,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each query's keys by a beam down the levels - GPU-torch walk of ``read_hierarchy``.

    The beam semantics are upstream's: the levels as the query's page saw them (the
    ``growth_at`` replay), candidates beside the carried children scored by centroid,
    the best ``beam_width`` a level kept, the last beam's children plus the uncovered
    buckets picked to ``top_p``, and the same expansion. Vectorized over queries;
    gated set-equal against the reference walk by ``gates.py``.
    """
    from lsa_rocm_lab.port._tree import tree_select_batched

    return tree_select_batched(index_queries, positions, buckets, hierarchy, config, top_k)


def dense_read(q, k, v, scale):
    """The dense causal baseline - torch SDPA, whatever attention ROCm PyTorch dispatches.

    The query rows are the *tail* ``q_len`` positions of the ``kv_len``-long cache
    (a decode step reads ``q_len == 1``, a chunked prefill ``1 < q_len < kv_len``).
    ``is_causal=True`` alone is wrong for every ``q_len != kv_len`` shape: PyTorch
    aligns that mask to the top-left, so a single decode query would attend key 0
    only. Instead: the square case keeps the fused causal flag, one row attends the
    whole cache unmasked, and a chunk gets an explicit bottom-right-aligned causal
    mask - key j is visible to chunk row i iff j <= (kv_len - q_len) + i, the
    absolute positions' causality (gates G and tests/test_dense_read.py pin this).
    """
    heads_q, heads_kv = q.shape[2], k.shape[2]
    q_len, kv_len = q.shape[1], k.shape[1]
    if q_len > kv_len:
        raise ValueError(f"query rows {q_len} exceed the cache {kv_len}: no causal read exists")
    queries, keys, values = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    gqa = heads_q != heads_kv
    if q_len == kv_len:
        out = F.scaled_dot_product_attention(queries, keys, values, is_causal=True, scale=scale, enable_gqa=gqa)
    elif q_len == 1:
        # The one query sits at the cache's last position: everything is visible
        out = F.scaled_dot_product_attention(queries, keys, values, scale=scale, enable_gqa=gqa)
    else:
        causal = torch.arange(kv_len, device=q.device)[None, :] <= torch.arange(
            kv_len - q_len, kv_len, device=q.device
        )[:, None]
        out = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=causal, scale=scale, enable_gqa=gqa
        )
    return out.permute(0, 2, 1, 3)


def lsa_read(q, k, v, read_lists, lengths, scale):
    """The sparse read over selected lists - this port's Triton gather-attention kernel."""
    return sparse_read(q, k, v, read_lists, lengths, scale)


def position_buckets(index_keys: torch.Tensor, config: PortConfig) -> Buckets:
    """The positional block-sparse baseline's buckets - runs of consecutive keys, upstream's research option.

    The paper's Block Sparse arm: same bucket machinery, same select and read, the
    clustering replaced by position - what a PISA-style selector groups by.
    """
    keys = index_keys.float()
    block, size, cap = config.cluster_block_size, config.bucket_size, config.bucket_cap
    blocks = keys.shape[0] // block
    bpb = config.block_bucket_count
    index_dim = keys.shape[1]
    closed = keys[: blocks * block]
    # [blocks, bpb, bucket_size] of block-local positions, -1 padded to the cap
    local = torch.arange(block, device=keys.device).view(bpb, size).expand(blocks, bpb, size)
    members = F.pad(local, (0, cap - size), value=-1)
    offsets = (torch.arange(blocks, device=keys.device) * block).int()[:, None, None]
    members = torch.where(members >= 0, members + offsets, members).reshape(blocks * bpb, cap).int()
    sizes = (members >= 0).sum(1, dtype=torch.int32)
    bucket_of_key = (
        torch.arange(block, device=keys.device) // config.bucket_size
        + bpb * torch.arange(blocks, device=keys.device)[:, None]
    )
    sums = torch.zeros(blocks * bpb, index_dim, device=keys.device)
    sums.index_add_(0, bucket_of_key.flatten(), closed.reshape(-1, index_dim))
    unit = F.normalize(sums, dim=-1)
    return Buckets(unit.to(torch.bfloat16), members, sizes, centroids_f32=unit)
