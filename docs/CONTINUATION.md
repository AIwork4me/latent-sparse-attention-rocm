# CONTINUATION — exact resume point for the next session

Updated: 2026-11-07 ~14:40 UTC (post-release; owner approved chain 1–5, executed)
Branch context: **main @ `8f2d13b`** — v0.3.1 RELEASED. PR #1 and PR #5 MERGED (merge commits `7b0b06f`, `958d948`); tags `v0.2.1`→`7b0b06f`, `v0.3.1`→`0f1abeb`; both Releases published. Legacy branches retained (`v0.2.1-scientific-validity`, `v0.3.0-kernel-optimization`) — do not delete; they are the rollback anchors.
Upstream pin: `5ebea732` (unchanged; every receipt).

## Release ledger (exact)

| Object | SHA | Evidence |
|---|---|---|
| PR #1 merge | `7b0b06f` (head `c12e8b5` = `79412f8` + Option-A hygiene) | post-merge @ merge tree: CPU 117/16/0, 16/16 gates, 133 GPU tests, evidence 24/24 |
| v0.2.1 | tag → `7b0b06f`, published 14:17:36Z | releases/tag/v0.2.1 |
| PR #5 merge | `958d948` (branch head `99902d7` = `c7661c2` + merge-of-main; SHA256SUMS conflict resolved as verified superset) | §7 hard gate @ `958d948`: CPU 141/46/0, **17/17 gates default AND batched**, 192 gfx1151 tests, evidence 71/71, recompute 153/153, guard PASS, main CI success |
| v0.3.1 | bump `0f1abeb`, tag → `0f1abeb`, published 14:26:43Z (Latest) | releases/tag/v0.3.1 |
| Repo controls | gpu-regression env **required_reviewers=AIwork4me**; **sha_pinning_required=true**; guard hardened (hosted-label allowlist + R6 + pinned-source step); rocm-regression active, runner OFFLINE, queue empty | API-verified |

## Awaiting owner console actions (only remaining items)

1. **A3 runner migration** — sudo required at the console; paste-ready block: `docs/release-closure/R1_RUNNER_SECURITY.md` §2.1 (user `lsa-runner`, `/opt` install, hardened systemd unit, venv copy, one-line workflow path PR). Rollback §2.3. Runner stays OFFLINE until then (env approval already gates any dispatch).
2. After migration: **rotate the `amd` gh credential** (`gh auth logout`).
3. Optional UI toggle: Settings → Actions → require approval for all external contributors.

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
