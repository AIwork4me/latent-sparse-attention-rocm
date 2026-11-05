"""The sparse read on ROCm - a Triton gather-attention kernel. Not in upstream.

Upstream's own sparse read (``lsa.core.attention.topk_attention``) is the blocked
quadratic oracle: it materialises a full score row per query block and masks it. In
serving the read is delegated to FA3 / FlashMLA / TileLang attention backends keyed
by exact list lengths, none of which exist on ROCm. This kernel is the port's
replacement: one program per (query row, head) walks that row's read list in blocks,
gathers K/V at the listed ids, and keeps a flash-style online softmax - so no
[Tq, Tk] score matrix ever exists, exactly like upstream's kernel-side intent.

Inputs match the contract of upstream's expansion output: read lists
``[B, Tq, width]`` int32 holding the expansion, the window, then the sinks, -1
padded (``width = top_k + window_size + sink_size``), plus keys-read counts per row.
GQA is honoured: each query head reads its group's KV head.

The kernel is causal by construction (a list only holds positions <= its query),
applies ``scale`` in fp32, accumulates the online softmax in fp32 and casts the
output back to the query dtype, as the oracle does.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_read_kernel(
    q_ptr, k_ptr, v_ptr, ids_ptr, lengths_ptr, out_ptr,
    scale,
    TK,  # keys per batch, runtime
    TQ: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
    D: tl.constexpr, WIDTH: tl.constexpr, NQ: tl.constexpr, ID_BLOCK: tl.constexpr,
):
    """One program reads one (query row, head) over its listed ids with an online softmax."""
    pid = tl.program_id(0)
    head = pid % HQ
    row = pid // HQ  # query row across the whole batch: row = batch * TQ + t
    batch = row // TQ
    kv_head = head // (HQ // HKV)

    offs_d = tl.arange(0, D)
    # q [D]; scores are taken in fp32 as the oracle does
    q = tl.load(q_ptr + row * (HQ * D) + head * D + offs_d).to(tl.float32)

    # A finite huge-negative running max keeps every -inf subtraction NaN-free:
    # exp(-3e38 - score) underflows to 0 and alpha is exactly 1 while nothing is live.
    m_i = tl.full((), -3.0e38, tl.float32)
    l_i = 0.0
    acc = tl.zeros((D,), tl.float32)

    length = tl.load(lengths_ptr + row)
    # num_stages=1: the ROCm Triton backend's default software pipelining miscompiles
    # this masked gather loop (nondeterministic illegal memory access at low occupancy,
    # e.g. the 8-program decode launch) - found by stress bisection on gfx1151,
    # Triton 3.8.0+git669b31ac.rocm10.1.0. Serialized, unpipelined, it is stable.
    for start in tl.range(0, NQ, ID_BLOCK, num_stages=1):
        offs_n = start + tl.arange(0, ID_BLOCK)
        valid = offs_n < length
        # -1 padding selects nothing; clamped to 0 and masked out of the softmax
        ids = tl.load(ids_ptr + row * WIDTH + offs_n, mask=valid, other=-1)
        kept = valid & (ids >= 0)
        safe = tl.where(kept, ids, 0)
        # k, v [ID_BLOCK, D] gathered at the listed key positions of this batch
        base = (batch.to(tl.int64) * TK + safe) * (HKV * D) + kv_head * D
        k = tl.load(k_ptr + base[:, None] + offs_d[None, :], mask=kept[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base[:, None] + offs_d[None, :], mask=kept[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(kept, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + row * (HQ * D) + head * D + offs_d, out.to(out_ptr.dtype.element_ty))


def sparse_read(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    read_ids: torch.Tensor,
    lengths: torch.Tensor,
    scale: float | None = None,
    block: int = 64,
) -> torch.Tensor:
    """Attention over each query row's read list - the ROCm sparse read.

    Args:
        q: Queries [B, Tq, Hq, Dh], contiguous.
        k: Keys [B, Tk, Hkv, Dh], contiguous.
        v: Values [B, Tk, Hkv, Dh], contiguous.
        read_ids: Read lists [B, Tq, width] int32, -1 padded (expansion, window, sinks).
        lengths: Keys each row reads [B, Tq] int32.
        scale: Score multiplier; None takes Dh ** -0.5.
        block: Ids gathered per inner iteration.

    Returns:
        Attention output, same shape/dtype/device as ``q``.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "the sparse read is a GPU kernel"
    B, Tq, Hq, D = q.shape
    Tk, Hkv = k.shape[1], k.shape[2]
    assert Hq % Hkv == 0, "query heads must group evenly onto KV heads"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    width = read_ids.shape[-1]
    flat_ids = read_ids.reshape(-1, width).contiguous()
    flat_lengths = lengths.reshape(-1).contiguous()
    out = torch.empty_like(q)
    nq = triton.cdiv(width, block) * block
    _sparse_read_kernel[(B * Tq * Hq,)](
        q, k, v, flat_ids, flat_lengths, out,
        D**-0.5 if scale is None else scale,
        Tk,
        TQ=Tq, HQ=Hq, HKV=Hkv, D=D, WIDTH=width, NQ=nq, ID_BLOCK=block,
        num_warps=4,
    )
    return out
