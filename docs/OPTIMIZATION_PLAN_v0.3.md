# v0.3.0 Work Plan — Genuine ROCm Kernel Optimization

Status: **plan (Draft PR, staged behind PR #1 — do not merge before it)**
Base: `v0.2.1-scientific-validity` · Opened: 2026-11-06 · Tracking: issue #4
(seed study: issue #2), `tracking/MILESTONES.yaml` v0.3.0.

## Objective

Close the 32k prefill deficit (0.19×) and improve decode (flat 2.1× / block-sparse
3.9× medians) **without any correctness regression**, on matched baselines, with
auditor-approved receipts. Honest no-gain is an acceptable result; a non-equivalent
baseline or skipped gate is not.

## Baseline (audited v0.2.1 receipts, do not re-derive by hand)

From `evidence/repetitions-20261105T225807Z/` and
`evidence/kernel-receipt-overnight-bf16.json` (all auditor-APPROVED):

- Decode select+read @32k bf16: dense ≈1.27 ms; block-sparse ≈3.9×; flat ≈2.1×
  (spread 1.5–2.4 across processes); tree ≈0.9×.
- Prefill select+read @32k: ≈0.19× — the sparse read dominates (untiled,
  one-program-per-row gather, `read_kernel.py`).
- Stage rows show `sparse_read` and `select` are the cost centers; index/hierarchy
  build are amortizable and already timed separately.

## Tasks (order matters; each lands as its own commit + tests)

### T1 — Multi-seed decode study (issue #2; gates all later claims)
`kernel_adapter --seed S` for ≥3 seeds at 8k/16k/32k bf16 decode, ≥3 process
repetitions each; auditor-approve everything; publish per-seed spread in
BENCHMARKS.md. Output: a variance band every T2–T4 gain must exceed to count.

### T2 — Profiling attribution (no code changes to kernels yet)
Per stage (Ward build, top-k listing, expansion, sparse read, tree beam):
warm-vs-cold separation (already excluded compile via warmups — now also record
first-call times), `torch.profiler` kernel table, clocks via `rocm-smi --showclock`
sampling in `scripts/telemetry.sh`, and existing stage rows. Deliverable:
`docs/OPTIMIZATION_v0.3.md` §bottleneck table with kernel IDs.

### T3 — Tiled multi-query sparse read
`read_kernel.py`: batch query rows per program, `tl.dot` where shapes allow,
coalesced K/V gathers (sort ids block-locally before load), multi-query blocks.
Constraints: preserve -1 padding, duplicate ids, sinks/window semantics, GQA,
fp16/bf16/fp32 tolerances (gates D1/D2/E/F stay the bar), arbitrary positions,
0/1/boundary lengths (add the L2 zero-length guard). A/B against the v0.2.1 kernel
in the same process, same inputs; success = prefill ratio materially up with
16/16 gates green.

### T4 — Kernelized tree beam (feasibility-scoped)
Replace the python-composed beam loop with Triton/GPU-torch kernels where the
win is proven; batched decode B={1,2,4,8}; incremental index maintenance. Gate C
(set-equality vs reference walk) is the correctness bar; scan-select/shape-scoring
stay opt-in with separately gated accuracy.

## Acceptance gates (release-blocking)

1. 16/16 gates PASS unchanged (plus any new negative tests).
2. Every comparison from auditor-APPROVED receipts on matched inputs; ≥3 seeds /
   ≥3 process reps on strategic shapes; p50/p95/MAD + memory reported.
3. No correctness regression; gains stated with variance bands from T1.
4. `docs/OPTIMIZATION_v0.3.md` baseline→candidate waterfall; Draft PR updated;
   independent review before any merge; owner approval to merge.

## Out of scope (later milestones)

64k+ contexts (v0.4.0), real-model quality (v0.5.0), serving (v0.6.0+).
