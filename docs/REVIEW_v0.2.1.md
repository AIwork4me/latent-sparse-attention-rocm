# Independent review — v0.2.1 scientific validity closure

**Reviewer role:** independent reproducibility auditor + senior ROCm kernel
engineer (agent session, 2026-11-06 UTC)
**Subject:** `AIwork4me/latent-sparse-attention-rocm` @ branch
`v0.2.1-scientific-validity` (base: `5f2918e`, the v0.2.0 release state)
**Method:** full read of the harness/port code at both revisions, live
reproduction of every defect on the target machine (Radeon 8060S / gfx1151,
ROCm 10.1 pip wheels, torch 2.14.0+rocm10.1.0, Triton 3.8.0-rocm10.1.0), a full
re-run campaign, and inspection of every v0.2.0 receipt against its report.

---

## A. Findings (each reproduced before it was fixed)

### F1 — CRITICAL: the Dense-Decode baseline was semantically wrong (and slow)

**Where (v0.2.0):** `src/lsa_rocm_lab/port/pipeline.py`, `dense_read` —
`is_causal=True` passed unconditionally; `port/bench.py`, `bench_dense_decode`
sliced `q[:, -1:]` against the full KV cache.

**Reproduction (this machine):** for `q_len=1, kv_len=4096` (bf16) the returned
row differs from the square-causal prefill's last row by **max abs 2.44** — the
top-left-aligned mask leaves exactly one key visible (key 0), so the baseline
computed softmax over a single value. The mis-dispatched causal path also took
**3.40 ms** p50 at 32k where the correct full-attention decode takes ~1.2 ms —
the defective baseline was ~2.8× too slow, inflating every v0.2.0 decode
speedup by roughly that factor.

**Consequence:** v0.2.0 headline decode speedups (5.04× flat, 7.86×
block-sparse, 2.18× tree at 32k) were invalid in both numerator semantics and
denominator timing. Corrected (median of 3 independent runs, §C): **2.1×, 3.9×,
0.9×**.

**Fix:** `dense_read` now defines query rows as the tail `q_len` positions of
the cache (square → fused causal; 1 row → unmasked; chunk → explicit
bottom-right-aligned mask; `q_len > kv_len` → ValueError). Pinned by
`tests/test_dense_read.py` (decode, chunks M∈{2,3,16,127}, square regression,
reject-longer) and release-blocking **gate G** (3 dtypes) inside the manifest.

### F2 — CRITICAL: `--skip-gates` produced a *claimable* receipt

**Where (v0.2.0):** `kernel_adapter.py` — `gates = [] if args.skip_gates else
run_all_gates(...)` then `correctness_pass = all(g["status"]=="PASS" ...)`;
`all([])` is vacuously `True`, so a skipped suite wrote
`"status": "PASS", "correctness_pass": true` and `run.py`'s validator accepted
it (`status=='PASS' and correctness_pass is True`), setting
`claimable_lsa_speedup: true`.

**Fix:** a torch-free gate manifest (`lsa_rocm_lab.gate_manifest`) is enforced
at both ends: the adapter writes `status: UNVERIFIED / correctness_pass: false /
gates_skipped: true` when gates are skipped, and validates the exact id set +
all-PASS verdicts when they run; `run.validate_receipt` independently re-checks
the manifest and rejects UNVERIFIED, empty, short, renamed, duplicated or
failed gate lists (verified live: an UNVERIFIED receipt from a real
`--skip-gates` run is refused with "not claimable"). Regression tests:
`tests/test_evidence_gates.py` (13 cases).

### F3 — HIGH: kernel-stage failures did not fail the campaign

**Where (v0.2.0):** `run.py` returned `0 if preflight_ok and correctness_ok
else 1` — a FAILed `lsa_rocm_kernel` stage (the speedup evidence itself) still
exited 0.

**Fix:** `run.campaign_exit_code` fails on any mandatory stage FAIL/TIMEOUT,
kernel included whenever an adapter ran; BLOCKED without an adapter remains a
legitimate non-claimable outcome. Tests: `tests/test_runner_contract.py` (9
cases).

### F4 — HIGH: receipts vs final report did not reconcile (v0.2.0 FINAL_REPORT)

1. §3 claimed "~40 min wall each" for the two profiles; the run trees'
   `stages.jsonl` sum to **~2.5 min per profile** (elapsed_s: 131 s / 137 s).
2. §6 claimed "fp16 matrix also recorded in receipts"; both v0.2.0 receipts are
   **bf16-only**.
3. The "quick" profile receipt is a **full 32k / reps-20** matrix identical in
   scope to overnight — both trees were direct `run.py` invocations
   (`run_meta.json` command), not the scripted quick pilot (2048/reps-5).
   v0.2.1's pilot genuinely runs 2048/reps-10 and validates.

**Fix:** FINAL_REPORT rewritten as v0.2.1 from generated tables; v0.2.0 numbers
preserved and marked SUPERSEDED (`docs/RESULTS_v0.2.0.md`,
`evidence/v0.2.0-archive/`); every table now cites run id, repo HEAD, upstream
SHA, device, seed, command and receipt path (`scripts/render_tables.py` refuses
unvalidated receipts).

### F5 — MEDIUM: self-hosted CI never ran; the local runner was registered to another repository

**Evidence:** `gh api …/actions/runners` returned `total_count: 0` while the
regression run for `5f2918e` sat queued 7+ hours; the machine's
`~/actions-runner/.runner` pointed at `AIwork4me/Qwen3-TTS-ROCm`. (Also: the
runner's `bin` symlink pattern resolves the .NET root through the *original*
directory — the first registration attempt failed with "already configured"
until the symlinks were retargeted locally.)

**Fix:** a dedicated runner install (`~/actions-runner-lsa`, name
`lsa-rocm-gfx1151`, labels `rocm,gfx1151`) registered to this repository and
online; the stale queued run cancelled. The workflow now (a) locates the exact
ROCm venv outside the workspace, (b) sets `PYTHONPATH=$workspace/src` so the
gates test the **PR's** code rather than the main checkout's editable install,
(c) runs the triaged upstream suite and the 16 gates, and (d) validates the CI
receipt with the same fail-closed validator. Known limitation: no passwordless
sudo on this host, so the runner runs via `nohup` rather than as a systemd
service (`sudo ./svc.sh install` from `~/actions-runner-lsa` is the follow-up
for reboot persistence).

### F6 — MEDIUM: the 7 upstream failures were mis-classified in v0.2.0

The v0.2.0 report said "(a) 4 Ward tests exercising tf32/tf32x3 input
precision" — true as far as it goes (all four call `ward_merge` /
`cluster_node_buckets` at the **default** tf32x3 precision; HIP's Triton
rejects `input_precision='tf32x3'` at compile: `must be one of ('ieee',
'bf16x3', 'bf16x6')`), but two of them (`test_every_block_ends_at_its_count_under_the_cap`,
`test_a_cap_that_leaves_no_pair_reports_the_block_short`) are cap-semantics
tests that merely *default* to tf32x3 — nothing about their assertions is
precision-related, and their substance is covered on ROCm at `ieee` precision
by `test_the_kernel_merges_as_the_torch_rounds_do` (passes) and gates A/A2.
The three shape-kernel failures are marginal RDNA numerics (measured: rel
7.07e-3 vs rtol 1e-3 on one of 128 variances; abs 2.0275e-3 vs atol 2e-3 on 1
of 6400 scores; last-bit bf16 differences breaking a `torch.equal` contract)
on a path the port does not use.

**Fix:** each of the 7 reproduced individually with exact error text and
classified in `evidence/upstream-test-classification.json` (4 ×
`UNSUPPORTED_PRECISION`, 3 × `NUMERIC_DIVERGENCE_UNUSED_PATH`, with per-test
port-coverage notes); `scripts/run_upstream_kernel_tests.sh` now triages via
`lsa_rocm_lab.upstream_triage` — classified failures stay fully visible, any
**unclassified** failure exits nonzero (203 pass / 7 classified / rc 0
verified live). Nothing is hidden: the classification file, not the exit code,
is what carries the reasons.

### F7 — LOW: campaign scripts had two latent breakages (found by the re-run)

1. `scripts/overnight.sh` invoked bare `python`, absent on this host → the
   "overnight" leg of the first v0.2.1 chain attempt exited 127 without
   running. Fixed to `$ROOT/.venv/bin/python`.
2. The quick pilot's `LSA_REPS=5` trips the receipt validator's `samples >= 10`
   evidence bar — v0.2.0 never hit this because its "quick" run silently used
   reps 20. Pilot now uses reps 10 (the documented minimum).

### F8 — LOW: receipts carried no code provenance

v0.2.0 receipts could not be tied to the code that produced them. v0.2.1
receipts record `run_id`, `repo_head_sha`, `gpu_name`, `gpu_arch`, `torch`,
exact `command`, contexts, phases; run trees record run_id/profile/hostname;
`run.py` threads `LSA_RUN_ID` into adapter receipts.

### F9 — INFO: benchmark methodology limits that remain (honest open items)

- Single-GPU, single-run matrices for non-strategic shapes; the three-run
  repetition evidence covers decode at 8k/16k/32k only (the headline claims).
  Flat @32k run-to-run spread is wide (1.52–2.41×) and is published as such.
- Benchmarks time attention-layer steps on prebuilt indexes (per the port
  contract); index build is reported separately and not amortized in the
  headline speedup.
- `dense_read` decode timing depends on PyTorch's dispatch for the unmasked
  1-row path (efficient attention); backend selection is recorded via torch
  version, not pinned per-kernel.

## B. What was verified green (this machine, this branch)

| Check | Result |
|---|---|
| Port gates | **16/16 PASS** (`python -m lsa_rocm_lab.port.gates`, ids exactly the manifest) |
| Repository tests | **50/50 PASS** (`pytest tests/ -v`) |
| Upstream kernel tests | 203 pass / 7 classified / 0 unclassified (triage rc 0) |
| Campaign (pilot + overnight bf16) | all stages PASS, `claimable_lsa_speedup: true`, rc 0 |
| fp16 matrix | all stages PASS, receipt validated |
| 3× independent repetitions | 16/16 gates each; aggregation validated all receipts |
| Receipt validation | every published receipt passes `run.validate_receipt` incl. gate manifest |
| Fail-closed negative paths | UNVERIFIED receipt rejected; empty/short/renamed/duplicate/failed gate lists rejected (unit tests + live) |

## C. Corrected headline numbers (see `docs/BENCHMARKS.md` for all tables)

Decode select+read, 32k ctx, bf16, three independent runs (per-run speedups;
median bold):

| arm | run 1 / 2 / 3 | median |
|---|---|---:|
| block-sparse | 3.93× / 3.57× / 4.17× | **3.93×** |
| D-LSA Flat | 2.41× / 1.52× / 2.11× | **2.11×** |
| D-LSA Tree | 0.89× / 0.74× / 0.92× | **0.89×** |

At 16k: block-sparse 1.81×, flat **0.89× (slower than dense)**, tree 0.46×.
At 8k: block-sparse 1.07×, flat 1.02×. Prefill at 32k remains 0.19×. The
"LSA beats dense decode everywhere ≥4k" story of v0.2.0 does not survive the
corrected baseline; the real advantage at these scales is block-sparse at
large contexts, with flat competitive only at 32k.

## D. GO/NO-GO

**GO — v0.2.1 as a scientific-validity release.** The decode semantics are
repaired and pinned by a release-blocking gate; the evidence chain is
fail-closed end to end (adapter, validator, campaign exit code, CI triage);
every published number is generated from validated receipts with full
provenance; the superseded claims are preserved and explained; the regression
workflow runs green on a correctly-registered self-hosted gfx1151 runner.

**NO-GO — carrying any v0.2.0 performance claim forward, and any new
performance work before the ROADMAP tracks land.** With the corrected
baseline, D-LSA Flat decode is slower than dense below ~16k and ~2.1× at 32k
(with wide variance); the tree arm is slower everywhere at these scales.
Kernel optimization (tiled read, kernelized beam) should proceed from these
corrected numbers, and future claims should quote repetition medians with
spread, as this release does.

## E. Residual risks / follow-ups

1. Runner durability: install as a systemd service when sudo is available
   (`sudo ./svc.sh install && sudo ./svc.sh start` in `~/actions-runner-lsa`).
2. Extend repetitions to prefill and to fp16 strategic shapes (evidence bar,
   not a correctness gap).
3. Re-derive shape-kernel tolerances on gfx1151 (ROADMAP Track 2.6) before any
   use of shaped scoring; the three divergences stay classified until then.
4. The upstream-watch cron (02:17 UTC daily) had not fired for the first time
   at review time; a manual dispatch verified behavior (upstream HEAD ==
   reviewed SHA `5ebea732`, no spurious issue). Watch its first natural run.
