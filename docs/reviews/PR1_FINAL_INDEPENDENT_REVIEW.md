# PR #1 — Final Independent Review (v0.2.1 scientific validity)

**Reviewer role:** independent (not the PR author's `docs/REVIEW_v0.2.1.md`, which is
treated as a claim, not proof).
**Reviewed as returned:** head `e243ba4e03f996e8650b4eb143dd678eff13c643`
(14 commits, 65 changed files) against base `main` `5f2918e3fd3441d81b07ed8e9b8c8c47ae5689cb`.
**Date:** 2026-11-06. **Repository state:** public; runner routing verified live.

---

## 1. Verification actually performed (this review)

| Check | Result | Evidence |
|---|---|---|
| `git diff --check` (whitespace/conflict markers) | clean | local run at `e243ba4` |
| Full unit + negative + adversarial suite on gfx1151 | **133 passed**, 0 failed | `.venv/bin/python -m pytest tests/ -v` (2026-11-06) |
| 16/16 port correctness gates on the GPU at head | **PASS** (fresh run, incl. G:fp32/fp16/bf16 dense-decode == prefill tail) | `lsa_rocm_lab.port.gates` output, this session |
| Triaged upstream Triton kernel tests | **203 pass / 7 classified / 0 unclassified**, exit 0 | `results/upstream-triage-20261106T004630Z.json` |
| Trusted GPU workflow (dispatch, confirmed ref) | **success 1m35s**; its receipt re-audited locally: APPROVED, 12 comparisons recomputed | run `42139297335`, artifact `ci-kernel-receipt-gh42139297335` |
| Fat-finger guard (dispatch, mismatched `confirm_ref`) | **fails closed in 20s** before any GPU work | run `42139650668` |
| Public-PR routing (live push to the PR branch) | `pull_request` event ran **only** the GitHub-hosted CPU workflow; no new self-hosted PR run | runs `42139138701/42139629501` vs last self-hosted PR run `42128067427` (pre-fix) |
| CPU-PR workflow (GitHub-hosted, zero secrets) | **success 59s** after fixes | run `42139623108` |
| Independent receipt audit of all 6 claim receipts | **APPROVED, 111 comparisons recomputed** from raw stage rows | `evidence/audit-v0.2.1.json` |
| Fresh post-fix receipt at head `2139758` | adapter PASS → validator PASS → auditor APPROVED (12 comparisons) | `evidence/receipt-2139758-postfix-verification.json`, `evidence/audit-2139758-postfix.json` |
| Upstream pin integrity | checkout at `5ebea732d639…`; `checkout_upstream.sh` now pins the reviewed SHA by default | `tracking/UPSTREAM_BASELINE.json` |
| Headline figures independently reconstructed | 32k bf16 decode medians across 3 runs: block-sparse ≈3.9×, flat ≈2.1× (1.5–2.4 spread), tree ≈0.9×; prefill ≈0.19× — recomputed by the auditor, not copied from prose | `evidence/repetitions-20261105T225807Z/` |

**Not executed / out of scope for this PR (honest ledger):** model-level accuracy
(v0.5.0), SGLang E2E (BLOCKED, documented), >32k contexts (v0.4.0), multi-seed study
(see L3 below).

---

## 2. Findings

### Fixed in this PR during review (were P0 blockers)

**F10 / BLOCKER→FIXED — public-PR code could execute on the persistent gfx1151 host.**
The pre-fix `rocm-regression.yml` carried an unfiltered `pull_request` trigger to
`runs-on: [self-hosted, rocm, gfx1151]`. This repository is **public** and the runner
executes as the interactive user `amd`, whose `$HOME` contains **live gh CLI
credentials** (`~/.config/gh/hosts.yml`) — a single malicious fork PR was a credential
compromise. Fix (commit `2139758`): `pull_request` removed from the GPU workflow;
new GitHub-hosted `cpu-pr.yml` (ephemeral, `contents: read`, zero secrets, loud GPU
skips) is the only PR-facing workflow; GPU = push-to-main / confirmed dispatch only,
SHA-pinned actions, concurrency-queued, no secrets; `scripts/check_workflow_security.py`
(R1–R5) fails CI on regressions (it caught the unpinned `upstream-watch.yml` checkout
itself); `docs/SECURITY_RUNNER_POLICY.md` records the threat model, live host audit
and owner hardening items. **Live validation** in the table above (both positive and
negative paths). Residual owner actions (runner service account, credential rotation,
environment reviewers) are listed in the policy §4 — they are defense-in-depth, not
gates for this PR.

**F11 / BLOCKER→FIXED — campaign exit code could green a failed campaign.**
Three holes: (a) a configured-and-attempted kernel adapter that exited 78 (BLOCKED)
left `campaign_exit_code` at 0; (b) an explicitly configured E2E adapter's failure was
ignored entirely; (c) `scripts/campaign.sh` printed `pilot rc=1` and always exited 0.
Fix (`2139758`): attempted kernel/E2E adapter on **anything but verified PASS** fails
the campaign (30 parameterized tests); a PASS record with no adapter run is rejected
as fabricated; `claimable_lsa_speedup` computed by stage name; `execute_stage()`
extracted and covered for crashes/timeouts/partial receipts/missing upstream SHA/
pre-log failures (15 tests); `campaign.sh` aggregates and propagates every phase's
exit (8 shell tests, sandboxed).

**F12 / BLOCKER→FIXED — no independent check between receipts and published claims.**
Fix (`2139758`): `lsa_rocm_lab.evidence.auditor` trusts nothing the adapter wrote —
it re-verifies the 16-gate manifest, provenance SHAs (pinned upstream, repo HEAD,
gfx1151 device, traceable run id, exact CLI), validates every stage row (finite
positive timings, ≥10 samples, p50∈[min,max], single seed/dtype/geometry, sparse
budgets recorded, dense decode `query_len==1`), then **recomputes every comparison**
from raw rows (`baseline_p50/candidate_p50`, tolerance 1e-9), rejecting altered
speedups, swapped baselines, duplicate/missing/ambiguous rows, NaN/Inf, replayed run
ids, corrupted `SHA256SUMS`, and precomputed-index scope inflation. 30 adversarial
fixture tests. `render_tables.py` and GPU CI now refuse non-APPROVED receipts; all 6
v0.2.1 claim receipts APPROVED (111 comparisons). `BENCHMARKS.md`/`FINAL_REPORT.md`
regenerated through that gate; v0.2.0 preserved as SUPERSEDED.

**M1 / MEDIUM→FIXED — CI upstream checkout floated to upstream main.**
`checkout_upstream.sh` defaulted to `origin/main`: an upstream push would have moved
CI onto an unreviewed SHA while `upstream.sha` silently tracked it. Found during this
review; fixed in `e243ba4` (default = reviewed SHA from `tracking/UPSTREAM_BASELINE.json`;
`UPSTREAM_REF` remains the explicit override for approved compatibility runs).

**CI1 / MEDIUM→FIXED — first CPU-PR runs failed collection** (no `PYTHONPATH`, no
triton, no upstream checkout for the port's import chain). Fixed in `e243ba4`; live
run green in 59s.

### Open findings (none blocking merge; tracked)

**L1 / LOW — `bench.py` docstring says "CUDA events" but timing is synchronized
`time.perf_counter`.** The measurement is correct (sync on both sides of each rep);
only the docstring's mechanism claim is wrong. Repro: `src/lsa_rocm_lab/port/bench.py:3-4`
vs `Clock.time` (`bench.py:37-48`). Fix: one-line docstring correction in v0.3.0.

**L2 / LOW — `read_kernel.sparse_read` returns NaN for a zero-length read list**
(`l_i == 0` → `0/0`). Unreachable in the current harness (every expansion includes
`sink_size ≥ 4`), but the kernel is an API surface. Repro: call `sparse_read` with
`lengths` all zero. Fix (v0.3.0): clamp `l_i` to a floor or skip rows with
`length ≤ 0`. `src/lsa_rocm_lab/port/read_kernel.py:76`.

**L3 / LIMITATION (recorded per mission) — the "three independent decode repetitions"
are three processes sharing ONE seed (20261105).** They demonstrate run-to-run
timing variance, not seed-space robustness. `kernel_adapter.py` exposes `--seed`;
a ≥3-seed study at strategic shapes is scheduled as the first v0.3.0 benchmark task
(tracking/MILESTONES.yaml `v0.3.0/seed-study`).

**L4 / LOW — p95 equals max for 20-sample rows** (`ordered[min(n-1, int(n*0.95))]`
→ index 19). Conservative, not wrong; worth an interpolated p95 when v0.4.0 raises
sample counts. `src/lsa_rocm_lab/port/bench.py:60`.

**L5 / INFO — `p95==max` and single-geometry matrix (8Q/2KV/64d) are the receipts'
whole scope.** Correctly labeled everywhere (`select+read on a prebuilt index`,
`not TTFT/TPOT or E2E`); no claim inflation found. The auditor now enforces the
labeling (MISLABELED_SCOPE).

### Security & supply chain

- Workflows: SHA-pinned actions, `persist-credentials: false`, least permissions,
  no secrets on self-hosted jobs, concurrency-bounded GPU; guard R1–R5 in CI on
  every PR and push (see F10). Host exposure and rotation needs documented without
  exposing secret values (`docs/SECURITY_RUNNER_POLICY.md` §2/§4).
- License: Apache-2.0 provenance preserved (upstream read-only at pinned SHA; port
  files carry notices); no upstream files modified; no vendored binaries.

---

## 3. Gate matrix

| Gate | Status |
|---|---|
| Code Reviewed | ✅ this document (all 65 files; core numerics re-derived, adversarial tests run) |
| GPU Tested (gfx1151, ROCm 10.1 / torch 2.14.0+rocm10.1.0) | ✅ 133 tests, 16/16 gates, upstream 203/7/0, trusted CI run green |
| Accuracy Measured (model-level) | ❌ not in scope — v0.5.0 |
| E2E Verified (serving) | ❌ BLOCKED (SGLang MI-class; feasibility lane documented) — v0.6.0 |
| Production Qualified | ❌ — v1.0.0 hard gates |

## 4. Verdict

**GO (merge-ready) for v0.2.1 as a scientific-validity release** — every P0 blocker
from the prior review round (F1–F9, closed in-repo earlier) and this round's F10/F11/F12
plus M1 are fixed, tested, and validated on live infrastructure; every published
number is receipt-backed and independently recomputed; failures fail closed.

**NO-GO** for anything beyond: no TTFT/TPOT/E2E/model-quality claims, no >32k
extrapolation, no performance-win narrative beyond the tables (tree slower than
dense; flat slower ≤16k; prefill 0.19×). If the owner merges, tag `v0.2.1` only
after the post-merge main SHA passes the trusted GPU workflow.

**Decision left to the repository owner:** PR #1 remains Draft; no merge performed.
