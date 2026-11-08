# M2 findings — long-context scaling ladder, 64k → 256k (bf16, gfx1151)

Receipt: `tracking/overnight_v1/receipts/m2-ladder-bf16.json` sha256 `d0bf6477…`
Harness: `scripts/ctx_scale_ladder.py` (fail-closed: projection precheck, chunked
reference oracle, per-arm OOM classification; chunked==monolithic oracle asserted at 8k).
Single seed (424243), warmup 5 / reps 20, interleaved arms, production entry points.

## Correctness (the good news)

- **15/15 measured phase rows fused == reference byte-exact + 3-repeat deterministic**,
  across 64k/96k/128k/192k/256k decode (b1 + ragged) and 64k/96k/128k prefill — the
  wide/sampled fallback contract holds at every real size, not just the adversarial tests' shapes.
- OOM shapes (192k/256k prefill) are recorded as `OOM` with the failing allocation
  (`19–38 GiB` asks vs `3–14 GiB` free on the 100 GiB GTT) — never a PASS, never zero.

## Performance (the honest news)

Contexts ≥ 65536 are the **wide/sampled path** (`columns > 4·top_p`): the fused kernel A
no longer applies; fused runs its length-respecting pack+topk fallback. Measured result:
**fused is systematically SLOWER than reference on the wide path** and the gap widens with
context:

| ctx | prefill select | prefill s+r | decode b1 s+r | decode ragged s+r |
|---:|---:|---:|---:|---:|
| 64k | 0.335× | 0.669× | 1.045× | 0.878× |
| 96k | 0.256× | 0.610× | 0.938× | 0.843× |
| 128k | 0.212× | 0.560× | 0.916× | 0.828× |
| 192k | — (OOM) | — (OOM) | 0.843× | 0.786× |
| 256k | — (OOM) | — (OOM) | 0.815× | 0.699× |

**Production guidance (evidence-backed): `LSA_SELECT_IMPL=fused` should stay opt-in and
scoped to the ≤32k ladder where it wins 1.18–1.23×; beyond 64k keep `reference`.**
This independently supports PR #8's narrow promotion scope. No kernel was changed tonight
(mission: optimize only profiled bottlenecks — the wide fallback's cost is now a measured,
registered candidate for a future F-series experiment).

## Memory envelope (UMA 100 GiB GTT ceiling, 32 GiB VRAM)

| ctx | decode peak | prefill peak (both arms live) | prefill verdict |
|---:|---:|---:|---|
| 64k | 0.26 GiB | 18.3 GiB | measured |
| 96k | 0.47 GiB | 40.1 GiB | measured |
| 128k | 0.62 GiB | 70.3 GiB | measured (reference fit via GTT) |
| 192k | 0.85 GiB | — | **OOM** (reference arm allocation 19–38 GiB fails) |
| 256k | 0.86 GiB | — | **OOM** (even fused cannot fit the fp32 score matrix) |

- Decode select+read is memory-trivial across the whole ladder (< 1 GiB): long-context
  decode is feasible today; the binding constraint is prefill's O(ctx²/8) scores/keys.
- 256k prefill needs a chunked/streamed prefill design (the mission's separate optional
  lane) — not blind full-matrix allocation.

## Harness defects (recorded, not rerun)

- Lane-close rule fired late: a fully-OOM prefill row is classified `FAILED` (because
  `lane_failed` is set by any non-PASS phase), so the `status == "OOM"` lane-close check
  missed 192k and 256k prefill attempted once more (each ~30 s, correctly classified OOM).
  No correctness impact; the extra 256k attempt yielded the useful fused-side OOM data point.
- 128k reference arm ran despite a ~28 GiB projection because free VRAM after 96k release
  was ~60 GiB (85% rule passed); it fit at 70.3 GiB peak — projection vs GTT elasticity
  noted for future budgeting.
