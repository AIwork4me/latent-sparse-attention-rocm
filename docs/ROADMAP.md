# Open-source roadmap — latent-sparse-attention-rocm

Living plan. Each item names its acceptance gate; nothing is "done" until the
gate passes on the target GPU.

## Track 1 — read-kernel performance (highest leverage)

The sparse read is the port's hot kernel and is correctness-first v1: one
program per (query row, head), elementwise dot, gather loads.

1. **Tile queries into the read kernel** — a program handles a block of query
   rows sharing the page's list prefix, converting the gather loop into
   `tl.dot` tensor-core tiles on unioned ids. Gate: D1/D2 stay green; decode
   p50 at 32k improves ≥ 2×.
2. **Coalesced K/V layout for the read** — gather rows by sorted ids within a
   chunk (the expansion already emits bucket-contiguous ids); measure the
   gather hit rate with `rocprof`.
3. **Fused decode** — score+select+read in one kernel for the single-row step
   (upstream's `scan_select` fused path and `paged.py` are the template).
   Gate: decode select+read beats the two-kernel form at every context.

## Track 2 — scale

4. **128k/256k contexts** — dense SDPA baselines through subprocess isolation
   (OOM-classified), LSA arms expected to keep their decode advantage; publish
   the prefill crossover curve as context grows.
5. **Fused scan-select above 32,768 buckets** — upstream's Triton
   `scan_select` path (currently unreachable below ~1M keys at these bucket
   sizes); requires the cutlass-free front door composition. Gate: upstream
   `test_scan_select.py` semantics reproduced through the port's gates.
6. **Shapes (bucket spread scoring)** — upstream's shape kernels show a
   marginal RDNA numerical divergence (PORT_NOTES §7); re-derive tolerances on
   gfx1151 before enabling shaped scoring.

## Track 3 — integration

7. **SGLang on ROCm APU** — blocked upstream of us: needs an SGLang ROCm build
   for gfx1151-class devices or a Triton-backend arm of the upstream plugin.
   When it exists: `--json-model-override-args '{"lsa": {...}}'`, capture the
   armed banner, E2E receipts per PORT_CONTRACT. Until then E2E stays BLOCKED.
8. **Model-specific distilled projections (D-LSA maps)** — a training campaign
   (upstream `lsa.training.distillation`) for a checkpoint that fits the APU;
   maps are checkpoint-bound and never transferable.

## Track 4 — upstream relationship

9. Daily read-only watch (`scripts/upstream_watch.py`, `upstream-watch.yml`)
   opens a deduplicated review issue per new HEAD; classification rubric:
   algorithm change / kernel change / bug fix / benchmark comparability /
   AMD portability. Baseline SHA moves only after the ROCm regression workflow
   passes on the new SHA.
10. Upstream contributions worth sending: the decode-phase query/position
    mismatch pattern (if generalizable), the Triton-HIP `num_stages` gather
    miscompile report, and an optional lazy-import of the cutedsl front doors
    (would unblock `test_scan_select.py` on AMD).

## Track 5 — release quality

11. Fresh-machine reproducibility: cold-venv CI job (self-hosted) building the
    environment from the README verbatim.
12. Parquet export of stage rows; per-stage roofline anchors (memory-bound vs
    compute-bound) for the read kernel.
