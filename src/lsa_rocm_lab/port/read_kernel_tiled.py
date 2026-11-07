"""The tiled sparse read (v0.3.0/T3) - row-tiled Triton gather-attention. Not in upstream.

The v1 kernel (``read_kernel.py``) runs one program per (query row, head): at
32k prefill that is 262k tiny programs each walking its whole read list in a
``num_stages=1`` serialized loop (software pipelining miscompiles the masked
gather on this ROCm Triton build), and T2 profiling attributes 299 ms - 77% of
the flat prefill step - to that single kernel, issue/latency-bound rather than
DRAM-bound (K+V is L2-resident at these sizes).

v2 tiles TM query rows into one program. Every row keeps its own read list
(the expansion is per-row), so the numeric path per row is v1's exactly: fp32
scores, fp32 online softmax, fp32 accumulator, output cast to the query dtype.
The wins are structural: q loaded once per tile, [TM, CN, D] gathers issuing
wider transactions, TM independent id-chains per program (memory-level
parallelism the pipelining ban removed), and TM-fold fewer programs to
schedule. ``tl.dot`` is deliberately NOT used: it needs one shared K matrix
for the whole M-axis, and per-row expanded lists (bucket-scattered members)
do not provide one - a forced tl.dot would silently change semantics.

Also fixes v1's zero-length hazard: a row whose list holds nothing (length 0,
all -1) produced 0/0 = NaN; v2 defines it as a zero output (no keys, no
contribution), and the L2 gate pins that contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_read_tiled_kernel(
    q_ptr, k_ptr, v_ptr, ids_ptr, lengths_ptr, out_ptr,
    scale,
    TK,
    TQ,  # per-batch query rows, runtime
    ROWS,  # total rows (B * Tq), runtime
    HQ: tl.constexpr, HKV: tl.constexpr,
    D: tl.constexpr, WIDTH: tl.constexpr, NQ: tl.constexpr,
    TM: tl.constexpr, CN: tl.constexpr,
):
    """One program reads TM query rows of one head over their listed ids."""
    pid = tl.program_id(0)
    head = pid % HQ
    tile = pid // HQ  # first row of the tile, in flat (batch * Tq + t) coordinates

    offs_m = tl.arange(0, TM)
    rows = tile * TM + offs_m  # [TM] query rows
    row_ok = rows < ROWS
    batch = rows // TQ  # batch index of each row
    kv_head = head // (HQ // HKV)

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, CN)
    # q [TM, D]; scores are taken in fp32 as the oracle does
    q = tl.load(q_ptr + rows[:, None] * (HQ * D) + head * D + offs_d[None, :],
                mask=row_ok[:, None], other=0.0).to(tl.float32)

    # A finite huge-negative running max keeps every -inf subtraction NaN-free
    m_i = tl.full((TM,), -3.0e38, tl.float32)
    l_i = tl.zeros((TM,), tl.float32)
    acc = tl.zeros((TM, D), tl.float32)

    lengths = tl.load(lengths_ptr + rows, mask=row_ok, other=0)
    for start in tl.range(0, NQ, CN, num_stages=1):
        cols = start + offs_n  # [CN] slot in the width
        valid = (cols[None, :] < lengths[:, None]) & row_ok[:, None]  # [TM, CN]
        ids = tl.load(ids_ptr + rows[:, None] * WIDTH + cols[None, :],
                      mask=valid, other=-1)
        kept = valid & (ids >= 0)
        safe = tl.where(kept, ids, 0)
        base = (batch[:, None].to(tl.int64) * TK + safe) * (HKV * D) + kv_head * D
        k = tl.load(k_ptr + base[:, :, None] + offs_d[None, None, :],
                    mask=kept[:, :, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base[:, :, None] + offs_d[None, None, :],
                    mask=kept[:, :, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[:, None, :], axis=2) * scale  # [TM, CN]
        scores = tl.where(kept, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(v * p[:, :, None], axis=1)
        m_i = m_new

    # zero-length rows: no keys read -> zero output (never 0/0 NaN)
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    tl.store(out_ptr + rows[:, None] * (HQ * D) + head * D + offs_d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


def sparse_read_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    read_ids: torch.Tensor,
    lengths: torch.Tensor,
    scale: float | None = None,
    tile_rows: int = 4,
    chunk: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Attention over each query row's read list - the tiled ROCm sparse read.

    Drop-in twin of ``read_kernel.sparse_read`` (same contract, same -1/-padding
    and GQA semantics) with the row tile ``TM=tile_rows`` and id chunk ``CN=chunk``
    as tunables. Returns attention output, same shape/dtype/device as ``q``.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "the sparse read is a GPU kernel"
    B, Tq, Hq, D = q.shape
    Tk, Hkv = k.shape[1], k.shape[2]
    assert Hq % Hkv == 0, "query heads must group evenly onto KV heads"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    width = read_ids.shape[-1]
    flat_ids = read_ids.reshape(-1, width).contiguous()
    flat_lengths = lengths.reshape(-1).contiguous()
    # Defense in depth (review F3): a caller claiming more keys than the row
    # holds would walk past the ids into unrelated memory in BOTH kernels; the
    # expansion contract keeps length <= width, but the wrapper clamps anyway
    # rather than trusting it. No-op for every production input.
    flat_lengths = flat_lengths.clamp_max(width)
    out = torch.empty_like(q)
    rows_total = B * Tq
    nq = triton.cdiv(width, chunk) * chunk
    grid = (triton.cdiv(rows_total, tile_rows) * Hq,)
    _sparse_read_tiled_kernel[grid](
        q, k, v, flat_ids, flat_lengths, out,
        D**-0.5 if scale is None else scale,
        Tk,
        Tq,
        rows_total,
        HQ=Hq, HKV=Hkv, D=D, WIDTH=width, NQ=nq,
        TM=tile_rows, CN=chunk,
        num_warps=num_warps,
    )
    return out


# Production dispatch (v0.3.0/T3): the A/B on gfx1151 (results/campaigns/
# campaign-20261106T231500Z/ab-read-kernel.json, audited copy in evidence/)
# measured TM4/CN32 best-or-within-1.2% at 8k/16k/32k prefill (1.24/1.18/1.14x
# vs v1) and TM1/CN64 at parity for the few-row decode launch, where bigger
# tiles only reduce parallelism. 32 rows is the decode/batch-decode boundary.
DECODE_ROW_LIMIT = 32


def sparse_read_auto(q, k, v, read_ids, lengths, scale=None):
    """The production sparse read entry: config chosen from the launch shape.

    Same contract as ``read_kernel.sparse_read`` (which stays as the A/B
    reference and is equality-gated against this by gate H every receipt run).
    """
    rows_total = q.shape[0] * q.shape[1]
    if rows_total <= DECODE_ROW_LIMIT:
        return sparse_read_tiled(q, k, v, read_ids, lengths, scale, tile_rows=1, chunk=64)
    return sparse_read_tiled(q, k, v, read_ids, lengths, scale, tile_rows=4, chunk=32)
