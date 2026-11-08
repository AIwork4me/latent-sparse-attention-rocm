# v0.4.0 / B2 — Candidate Designs and Decision Record

**Inputs:** `BASELINE_AND_BOTTLENECKS.md` (measured), the upstream contract study below, and a Triton-primitive probe (`tl.sort`/`tl.topk` on int64, single-CTA, gfx1151, this session).

## The exact contract any candidate must reproduce (mission B0)

Reference = v0.3.1 `pipeline.flat_select` (immutable oracle):

```
scores = mm(q_bf16, centroids.t(), out_dtype=fp32)            # production GEMM — untouched
keys, counts = TopK(top_p, exact=False).listed(scores, lengths)  # whole path: pack ALL N cols to int64, EMPTY past length
ordered = torch.topk(keys, k=min(top_p, N), sorted=True).values # k greatest keys, descending; keys unique ⇒ deterministic
picked = unpack_ids(ordered)                                    # int32 bucket ids, -1 where EMPTY
lists, lens = expand_in_order(picked, members, positions, window, sink, top_k)
```

Invariants the fused path must keep **byte-exactly** (gated, not argued): packed-key best-first order; score-tie → lower column (the pack's low word guarantees uniqueness, so no two keys ever tie); visible-count clamp; EMPTY/sentinel handling when a row lists fewer than k; expansion order (bucket at a time, member order within bucket, first `top_k` valid members, members outside `[sink, window_start)` skipped but window/sink appended last); final list lengths. Every benchmark context takes the listing's *whole* path (`columns ≤ 4·top_p`; 4096 at 32k) — the sampled path (`columns > 4·top_p`) is NOT replicated by any candidate and must dispatch to reference.

Upstream evidence that in-kernel selection is the intended direction: `expand_by_threshold` "stands in for their top-k, its sort and the walk"; `top_k.py`'s exact select finds the k-th key "a bit at a time from the top" in-kernel; serving's prefill uses the threshold expansion. None of these preserve the best-first *order* our port's contract (and mission B0) pins — so we select in-kernel but keep the sorted-order walk.

## Probe: Triton 3.8 `tl.sort` / `tl.topk` on int64 (exact vs `torch.topk`, 200-rep device timing)

| tile | k | exact? | `tl.topk` | `torch.topk` (1 row) |
|---:|---:|---|---:|---:|
| 64 | 64 | ✅ | 7.3 µs | 10.1 µs |
| 256 | 256 | ✅ | 7.1 µs | 19.7 µs |
| 1024 | 1024 | ✅ | 7.0 µs | 26.0 µs |
| 4096 | 1024 | ✅ | 45.1 µs | 29.1 µs |

Interpretation: exact and fast on narrow tiles; at 4096→1024 a single CTA loses to ATen's radix select *for one row* — but the reference's 63 ms at 32k prefill is not one row: it is 32768 rows through a 21-launch int64 fan-out, and `tl.topk` runs row-per-CTA with 2 CTAs/CU (32 KiB LDS each) — throughput-bound, launch-free, and it also deletes the pack materialization and the unpack chain. Decode (launch-bound) wins regardless of tile width. The probe's per-row 45 µs is latency, not throughput; the A/B decides.

## Candidates

### F1 — `expand_in_order_from_keys` (unpack elimination; upstream walk verbatim, keys read directly)
- **Change:** port-side Triton kernel identical to upstream `_expand_in_order` except it loads the ordered **int64 keys** and computes `bucket = 0x7FFFFFFF − (key & 0xFFFFFFFF)` in-kernel; `key == EMPTY` ⇔ the reference's `picked == −1`. Upstream's `_append_window_and_sink` imported and reused unmodified.
- **Removes:** the whole `unpack_ids` stage — 13 launches, ~10.5 % of 32k prefill select, ~0.03–0.06 ms of every decode select.
- **Semantic risk: none by construction** (pure relocation of the unpack arithmetic; same walk order, same masks). Register/LDS: unchanged vs upstream walk.
- **Role:** independently shippable micro-win, the fallback if F2 disappoints, and F2's walk stage.

### F2 — `fused_select_expand` (one kernel from scores to read lists; "F2a" = tl.topk select)
- **Change:** after the production GEMM, ONE port-side Triton kernel per row: load fp32 scores tile (`BLOCK = next_pow2(N)`), pack in-register (`ordered_bits`/key layout imported from upstream `keys.py`), `k2 = next_pow2(min(top_p, N))`, `picks = tl.topk(keys, k2, descending=True)`, then the F1 walk over `idx < k_eff` (`k_eff = min(min(top_p, N), live_count)` — exactly the reference's effective picks after EMPTY masking).
- **Removes:** `_pack_scores`' [R, N] int64 materialization (1 GiB at 32k prefill), the entire ATen top-k fan-out (21 launches, 65 % of prefill select), unpack (13 launches), the separate expansion launch. Post-GEMM select: **1 launch**.
- **Exactness argument:** keys are unique (column in the low word) ⇒ `tl.topk(sorted)` returns exactly `torch.topk(sorted=True).values[:, :k_eff]`; walk verbatim ⇒ byte-identical lists/lengths. Verified by gate + exhaustive adversarial suite (B3), not by argument alone.
- **Register/LDS (estimated, to be read off the compiled kernel):** keys tile 4096×8 B = 32 KiB + members tile ⇒ ~2 CTAs/CU; bitonic network on int64 → larger kernel (compile seconds, cached). Prefill parallelism = R CTAs (32768 at 32k) — plenty.
- **Known risk:** per-row sort latency at N=4096 (probe: 45 µs) could cap prefill throughput if occupancy disappoints — the A/B decides; **F2b** (upstream-style digit-radix threshold select + sort only the ≤1024 picks) is the documented next lever if F2a's prefill is still sort-bound. Not implemented now — no complexity before measurement.

### F3 — shape-aware dispatch (the policy layer, only if measurements demand it)
`LSA_SELECT_IMPL=reference|fused` selects the implementation; inside `fused`, dispatch is **explicit and logged**: `columns ≤ 4·top_p` (the whole-listing contract the kernel replicates) → fused kernel; anything wider, non-finite guard trips, or missing device → reference path, loudly (never a blanket `except`). If the A/B shows wins on some shapes and regressions on others, the dispatch narrows by measured shape (visible in code + tests), per the mission.

## Decision

Implement **F1 and F2** (F2's walk reuses F1's kernel body), gate both against the byte-exact oracle across the B3 adversarial matrix, then run the B4 interleaved A/B. Promotion (which value `LSA_SELECT_IMPL=fused` maps to, and whether dispatch is shape-narrowed) is decided by receipts, not here. F2b recorded as follow-on research, not in scope unless F2a fails its own bar.

**Negative-result obligation:** whichever candidate loses is kept here with its measured numbers; nothing is silently dropped.
