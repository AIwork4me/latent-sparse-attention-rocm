"""v0.4.0 fused selection candidates - the measured select-fusion kernels.

Two port-side Triton kernels attack the two measured select bottlenecks
(docs/v0.4.0/BASELINE_AND_BOTTLENECKS.md): the ATen int64 top-k fan-out that is
~65% of the 32k prefill select (21 launches) and the ``unpack_ids`` chain (13
launches, ~10%), plus the decode path's launch-bound 34-launch sequence.

``select_topk_scores`` (F2 stage A) fuses upstream's pack + the order-giving
top-k into ONE kernel: it reads the fp32 score row directly (upstream's whole
listing path packs every column anyway when ``columns <= 4 * top_p``), packs
score bits + column in-register exactly as ``pack_key`` does, and emits the
``min(top_p, columns)`` greatest keys sorted descending via ``tl.topk`` -
byte-identical to ``best_first_keys`` applied to the packed listing, without
materializing the ``[rows, columns]`` int64 key matrix at all.

``expand_in_order_from_keys`` (F1) is upstream's ``_expand_in_order`` walk with
the id unpack moved into the kernel: it reads the ordered int64 keys and
computes ``key_bucket`` in-register, eliminating the separate ``unpack_ids``
tensor and its 13 elementwise launches. The walk loop, its masks, the
window/sink tail (``_append_window_and_sink``) and the early budget are
upstream's, imported unmodified - the pinned upstream is never edited.

Equivalence is pinned byte-exactly against the v0.3.1 reference chain by
tests/test_select_fusion.py (ties, EMPTY keys, k == width, short rows, ragged
rows, window/sink boundaries, non-power-of-two widths, production shapes);
the wide-row sampled listing (``columns > 4 * top_p``) is NOT replicated and
falls back to the reference composition loudly - never a silent blanket path.
"""

import warnings

import torch
import triton
import triton.language as tl

from lsa.core.kernels.triton.expansion import _append_window_and_sink, _row_budget
from lsa.core.kernels.triton.keys import EMPTY, key_bucket, ordered_bits

EMPTY_KEY: tl.constexpr = tl.constexpr(EMPTY)

# The whole-listing contract upstream's TopK(exact=False) takes for every
# benchmark/production shape here (4096 buckets at 32k context = 4 * top_p).
# Wider score matrices use the sampled listing, which this kernel does not
# replicate - the wrapper dispatches those to the reference composition.
WHOLE_FACTOR = 4


@triton.jit
def _select_topk_scores(
    scores_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    out_ptr: tl.tensor,
    score_stride: tl.tensor,
    out_stride: tl.tensor,
    column_count: tl.tensor,
    BLOCK: tl.constexpr,
    K2: tl.constexpr,
) -> None:
    """One row's scores to its ``K2`` greatest packed keys, sorted descending.

    Packs in-register (score order bits high, ``0x7FFFFFFF - column`` low, the
    empty key where a column is past the row's length) and keeps the ``K2``
    greatest via ``tl.topk`` - keys are unique per column, so the sorted result
    equals ``torch.topk(..., sorted=True)`` exactly, EMPTY keys padding the tail
    when a row lists fewer than ``K2`` live columns.
    """
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    length = tl.load(lengths_ptr + row)
    live = (column < column_count) & (column < length)
    score = tl.load(scores_ptr + row * score_stride + column, mask=live, other=0.0)
    packed = (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - column).to(tl.int64)
    key = tl.where(live, packed, EMPTY_KEY)
    picks = tl.topk(key, K2, descending=True)
    at = tl.arange(0, K2)
    tl.store(out_ptr + row * out_stride + at, picks)


def select_topk_scores(scores: torch.Tensor, lengths: torch.Tensor, top_p: int) -> torch.Tensor:
    """The production listing + best-first keys, fused, on identical inputs.

    Args:
        scores: Score matrix [R, N] fp32 (the GEMM's fp32 output either way).
        lengths: Columns each row sees [R] int32, at most N.
        top_p: Best keys per row (the design's ``top_p``).

    Returns:
        Ordered packed keys [R, K2] int64, K2 = next_pow2(min(top_p, N)) - exactly
        ``best_first_keys(listing.listed(scores, lengths)[0], top_p)`` (EMPTY keys
        pad the tail of short rows, as ``torch.topk`` would return them too).
    """
    rows, columns = scores.shape
    if columns > WHOLE_FACTOR * top_p:
        # The sampled-listing contract: deliberate, visible fallback to the exact
        # reference composition (this shape never occurs in the benchmark ladder).
        warnings.warn(
            f"select_topk_scores: {columns} columns > {WHOLE_FACTOR} * top_p {top_p} "
            "- sampled listing contract, using the reference pack+topk",
            stacklevel=2,
        )
        from lsa.core.kernels.triton import keys as keys_mod

        ids = torch.arange(columns, device=scores.device, dtype=torch.int32).expand(rows, columns)
        keys = keys_mod.pack_keys(scores, ids)
        return torch.topk(keys, k=min(top_p, columns), dim=1, sorted=True).values
    if scores.device.type != "cuda":
        raise ValueError(f"select_topk_scores is a ROCm kernel, got device {scores.device}")
    if scores.dtype != torch.float32:
        raise ValueError(f"scores must be the GEMM's fp32 output, got {scores.dtype}")
    keep = min(top_p, columns)
    k2 = triton.next_power_of_2(keep)
    block = triton.next_power_of_2(columns)
    out = torch.empty(rows, k2, device=scores.device, dtype=torch.int64)
    _select_topk_scores[(rows,)](
        scores,
        lengths,
        out,
        scores.stride(0),
        out.stride(0),
        columns,
        BLOCK=block,
        K2=k2,
        num_warps=8 if block >= 4096 else 4,
    )
    return out


@triton.jit
def _expand_in_order_keys(
    keys_ptr: tl.tensor,
    bucket_members_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    slot_table_ptr: tl.tensor,
    slot_rows_ptr: tl.tensor,
    keys_stride: tl.tensor,
    out_stride: tl.tensor,
    slot_stride: tl.tensor,
    TOP_P: tl.constexpr,
    BUCKET_CAP: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> None:
    """Upstream's ``_expand_in_order`` walk reading packed keys, unpacking in-register.

    Identical loop, masks and window/sink tail; the only change is the pick load -
    an int64 key whose bucket is ``key_bucket(key)`` and whose EMPTY value plays
    the reference's ``-1`` pad. See upstream ``expansion.py`` for the contract.
    """
    row = tl.program_id(0).to(tl.int64)
    position = tl.load(positions_ptr + row)
    top_k, top_p = _row_budget(position, TOP_K, TOP_P, EARLY_POSITION, EARLY_TOP_K, EARLY_TOP_P)
    window_start = tl.maximum(0, position + 1 - WINDOW_SIZE)
    slot_row = 0
    if MAP_SLOTS:
        slot_row = tl.load(slot_rows_ptr + row).to(tl.int64)
    pad = tl.arange(0, PAD_BLOCK)
    tl.store(out_ptr + row * out_stride + pad, -1, mask=pad < WIDTH)
    tl.debug_barrier()
    slot = tl.arange(0, TILE * BUCKET_CAP)
    filled = 0
    for first in range(0, TOP_P, TILE):
        pick = first + slot // BUCKET_CAP
        key = tl.load(
            keys_ptr + row * keys_stride + pick, mask=(pick < top_p) & (filled < top_k), other=EMPTY_KEY
        )
        bucket = tl.where(key == EMPTY_KEY, -1, key_bucket(key))
        ids = tl.load(
            bucket_members_ptr + bucket.to(tl.int64) * BUCKET_CAP + slot % BUCKET_CAP,
            mask=bucket >= 0,
            other=-1,
        )
        valid = (bucket >= 0) & (ids >= SINK_SIZE) & (ids < window_start)
        at = filled + tl.cumsum(valid.to(tl.int32)) - 1
        stored = valid & (at < top_k)
        if MAP_SLOTS:
            ids = tl.load(slot_table_ptr + slot_row * slot_stride + ids, mask=stored, other=-1)
        tl.store(out_ptr + row * out_stride + at, ids, mask=stored)
        filled += tl.sum(valid.to(tl.int32))
    filled = tl.minimum(filled, top_k)
    length = _append_window_and_sink(
        out_ptr + row * out_stride,
        slot_table_ptr + slot_row * slot_stride,
        position,
        window_start,
        filled,
        WINDOW_SIZE,
        SINK_SIZE,
        LOCAL_BLOCK,
        MAP_SLOTS,
    )
    tl.store(lengths_ptr + row, length)


def expand_in_order_from_keys(
    ordered_keys: torch.Tensor,
    bucket_members: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read lists from ordered packed keys - upstream ``expand_in_order``, unpack fused.

    Args:
        ordered_keys: Packed keys [R, W] int64, best first, EMPTY-padded tail.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        positions: Each row's position [R] int32.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded, and their
        lengths [R] int32 - byte-identical to the reference unpack+expand pair.
    """
    if ordered_keys.device.type != "cuda":
        raise ValueError(f"expand_in_order_from_keys is a ROCm kernel, got device {ordered_keys.device}")
    top_p = ordered_keys.size(1)
    bucket_cap = bucket_members.size(1)
    width = top_k + window_size + sink_size
    out = torch.empty(ordered_keys.size(0), width, device=ordered_keys.device, dtype=torch.int32)
    lengths = torch.empty(ordered_keys.size(0), device=ordered_keys.device, dtype=torch.int32)
    decoding = ordered_keys.size(0) <= 32
    tile = max(1, (4096 if decoding else 1024) // bucket_cap)
    _expand_in_order_keys[(ordered_keys.size(0),)](
        ordered_keys,
        bucket_members,
        positions,
        out,
        lengths,
        bucket_members,
        positions,
        ordered_keys.stride(0),
        width,
        0,
        TOP_P=top_p,
        BUCKET_CAP=bucket_cap,
        TILE=tile,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        TOP_K=top_k,
        WIDTH=width,
        PAD_BLOCK=triton.next_power_of_2(width),
        LOCAL_BLOCK=triton.next_power_of_2(window_size + sink_size),
        MAP_SLOTS=False,
        EARLY_POSITION=0,
        EARLY_TOP_K=0,
        EARLY_TOP_P=0,
        num_warps=16 if decoding else 4,
    )
    return out, lengths
