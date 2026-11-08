# CONTINUATION — exact resume point for the next session

Updated: 2026-11-08 (R1.1 post-release hygiene executed on branch `chore/r1.1-post-release-hygiene`)
Branch context: **main @ `1ab0e0b`** — v0.3.1 RELEASED. PR #1 and PR #5 MERGED (merge commits `7b0b06f`, `958d948`); tags `v0.2.1`→`7b0b06f`, `v0.3.1`→`0f1abeb`; both Releases published (**bodies corrected to FINAL publication language 2026-11-08, tags untouched** — `docs/r1.1/RELEASE_METADATA_AUDIT.md`). Legacy branches retained (`v0.2.1-scientific-validity`, `v0.3.0-kernel-optimization`) — do not delete; they are the rollback anchors.
Upstream pin: `5ebea732` (unchanged; every receipt).

## R1.1 executed (2026-11-08, branch `chore/r1.1-post-release-hygiene`)

- **A1** both published Release bodies corrected (DRAFT wording → FINAL + exact tag targets; v0.3.1 test counts now lead with merged-tree 192/141; obsolete version-bump instruction marked EXECUTED). Archive + diffs: `docs/r1.1/release-archive/`.
- **A2** superseded queued GPU run `42381678702` cancelled; run `42381769814` @ current main HEAD intentionally kept as the post-migration validation vehicle (`docs/r1.1/ACTIONS_QUEUE_AUDIT.md`).
- **A3** runner migration fully preflighted nonprivileged; venv copy+repair **empirically validated** (torch+Triton kernel compile from relocated venv); corrected systemd unit fixes two R1-draft defects (`ProtectHome` hides the new HOME; `DeviceAllow` inert without `DevicePolicy=closed`); owner runbook `docs/r1.1/ops/migrate-runner.sh` + `rollback-runner.sh`. **Status OWNER_ACTION_REQUIRED** (`docs/r1.1/RUNNER_MIGRATION_VALIDATION.md`).
- **A4** branch/tag protection + rulesets proposal (P1–P4) prepared with exact commands, **nothing applied** (`docs/r1.1/REPO_SECURITY_AUDIT.md`).
- Baseline freeze ledger: `docs/r1.1/BASELINE_AUDIT.md`. v0.4.0 branch `feature/v0.4.0-measured-select-fusion` exists from the same verified main.


## Release ledger (exact)

| Object | SHA | Evidence |
|---|---|---|
| PR #1 merge | `7b0b06f` (head `c12e8b5` = `79412f8` + Option-A hygiene) | post-merge @ merge tree: CPU 117/16/0, 16/16 gates, 133 GPU tests, evidence 24/24 |
| v0.2.1 | tag → `7b0b06f`, published 14:17:36Z | releases/tag/v0.2.1 |
| PR #5 merge | `958d948` (branch head `99902d7` = `c7661c2` + merge-of-main; SHA256SUMS conflict resolved as verified superset) | §7 hard gate @ `958d948`: CPU 141/46/0, **17/17 gates default AND batched**, 192 gfx1151 tests, evidence 71/71, recompute 153/153, guard PASS, main CI success |
| v0.3.1 | bump `0f1abeb`, tag → `0f1abeb`, published 14:26:43Z (Latest) | releases/tag/v0.3.1 |
| Repo controls | gpu-regression env **required_reviewers=AIwork4me**; **sha_pinning_required=true**; guard hardened (hosted-label allowlist + R6 + pinned-source step); rocm-regression active, runner OFFLINE, queue empty | API-verified |

## Awaiting owner console actions (only remaining items)

1. **A3 runner migration** — sudo required at the console; the R1.1-validated owner runbook supersedes the R1 §2.1 paste block: `sudo bash docs/r1.1/ops/migrate-runner.sh` (preflight-verified, empirically validated venv-repair recipe, corrected unit, rollback `rollback-runner.sh`). Runner stays OFFLINE until then (env approval already gates any dispatch). Then the kept queued run `42381769814` validates end-to-end.
2. After migration: **rotate the `amd` gh credential** (`gh auth logout`).
3. **A4 protection proposal** (P1 ruleset main-protected, P2 tag protection `v*`, P3 external-contributor approval toggle, P4 runner-group scoping) — exact commands in `docs/r1.1/REPO_SECURITY_AUDIT.md`; apply from the owner console when approved.

## ONE next most important action

Owner executes A3 (then rotation). Engineering next lever only on owner go-ahead:
fused pack→expand select pass — `docs/OPTIMIZATION_v0.3.md` §h (measured order).

## Resume commands

```bash
cd ~/Desktop/lsa-rocm-eval-starter && git checkout main && git pull && git log --oneline -3
.venv/bin/python -m pytest tests/ -q                                   # 192 on gfx1151
PYTHONPATH=src .venv/bin/python -m lsa_rocm_lab.port.gates | tail -1   # 17/17
python3 scripts/check_workflow_security.py                             # guard PASS
cd evidence && sha256sum -c SHA256SUMS | grep -cv ": OK$"              # 0
gh release list                                                        # v0.3.1 Latest, v0.2.1
gh run list --limit 5                                                  # CI posture
```

## Risks / gotchas (carried; full list in git history of this file)

- Profiler-inflated GPU totals on this ROCm build — attribute by kernel-table shares, cost by synced p50.
- `pkill -f` self-match — kill by PID. Kill leftover GPU python before benchmarking.
- A/B scripts: receipt methodology (warmup 5 / reps 20), gate equivalence before timing.
- Receipts pin their code SHA — ANY perf-path source change ⇒ fresh receipts before new claims. Post-release: claims are frozen at `60c004b` receipts; new work lands on new branches with new evidence.
- The workflow-security guard now uses a hosted-label allowlist + R6 + a pinned-source re-run in cpu-pr.yml (pin `2af35b9`); bump the pin only via reviewed guard changes.
- pip stalls on download.pytorch.org (0 B/s stretches); aliyun mirror ~7× faster for generic packages; torch CPU wheel only from the pytorch index.
