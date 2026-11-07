# R1 Baseline — frozen state ledger (release-closure mission R1)

Date: 2026-11-07 (~20:30 local / 12:30 UTC) · Auditor: R1 release-engineering session (independent re-verification, no prior report trusted without checking)
Verdict per reference: **every reference SHA below re-verified against live GitHub + local state; none changed.**

## 1. Frozen SHA ledger (verified)

| Reference | Expected | Observed (2026-11-07T12:30Z) | Match |
|---|---|---|---|
| main | `5f2918e3fd3441d81b07ed8e9b8c8c47ae5689cb` | `5f2918e` (origin/main) | ✅ |
| PR #1 HEAD | `79412f8a8b4fcb09f09f4a41b6eb07d075dbe3b1` | `79412f8` (origin/v0.2.1-scientific-validity) | ✅ |
| PR #5 HEAD | `261805c98c41230c2bfc32533a78873857058fdf` | `261805c` (origin/v0.3.0-kernel-optimization; local HEAD identical, tree clean) | ✅ |
| Measured production code (all v0.3.1 perf receipts) | `60c004b5630771082ac9ac3a2877262cae1f4ed4` | present; ancestor of `0908842` | ✅ |
| Last production-code correction | `090884245e1dc6bcff30dd508470675cf2d99e18` | present; ancestor of PR #5 HEAD | ✅ |
| CPU portability test fix | `5a13bf07a5fe859edfb709e3a311fef11dd00ae5` | present; ancestor of PR #5 HEAD | ✅ |
| Upstream pin | `5ebea732d639576d2a7579b0de40e6252f9001f5` | `tracking/UPSTREAM_BASELINE.json` == `third_party/latent-sparse-attention` HEAD == every receipt's `upstream_sha` | ✅ |

Ancestry (verified with `git merge-base --is-ancestor`): `60c004b` ⊂ `0908842` ⊂ `261805c`; PR #1 HEAD `79412f8` ⊂ PR #5 HEAD `261805c` (stacking intact → merge-commit of PR #1 preserves PR #5 review diff).

## 2. Pull requests

| | PR #1 | PR #5 |
|---|---|---|
| Title | v0.2.1: scientific validity closure — decode baseline repair, fail-closed evidence, re-run benchmarks | v0.3.0 kernel optimization + v0.3.1 topk select — implemented, evidence-complete, owner review requested (Draft) |
| State | OPEN, **Draft** | OPEN, **Draft** |
| Base | `main` | `v0.2.1-scientific-validity` (stacked, by design) |
| Mergeable | MERGEABLE / CLEAN | MERGEABLE |
| CI | `static-and-unit` SUCCESS ×2 (GitHub-hosted CPU) | SUCCESS ×2 at `261805c` ([run 42371502625](https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42371502625), [run 42371506680](https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42371506680)) |
| Review decision | none requested/recorded | none requested/recorded |

Tags: **0** · Releases: **0** (nothing published).

## 3. Repository settings relevant to release

- Default branch `main`; merge-commit, squash and rebase all **allowed** (merge-commit preferred for PR #1 per §4 of the mission).
- `delete_branch_on_merge: false` (history preserved).
- Actions: `allowed_actions: all`, **`sha_pinning_required: false`** ← policy §4 item 2 not yet applied (owner-gated).
- Environments: `gpu-regression` exists with **no protection rules, no required reviewers, no branch policy** ← policy §4 item 1 not yet applied (owner-gated).
- Workflows (all active): `cpu-pr.yml` (pull_request+push → GitHub-hosted), `rocm-regression.yml` (push:main + workflow_dispatch w/ ref confirmation → self-hosted gfx1151, `environment: gpu-regression`), `upstream-watch.yml` (cron+dispatch → GitHub-hosted, issues:write via ephemeral `github.token`).
- Static guard `scripts/check_workflow_security.py` (R1–R5) re-run this session: **PASS**; runs inside both CI workflows.

## 4. Host / runner snapshot (read-only preflight, details in R1_RUNNER_SECURITY.md)

- Three runner installs under `/home/amd`: `actions-runner` (Qwen3-TTS-ROCm, systemd `github-runner.service`, running), `actions-runner-ovamd` (OpenVINO-Notebooks-on-AMD, systemd `github-runner-ovamd.service`, running), `actions-runner-lsa` (**this repo — OFFLINE**: no `Runner.Listener` process; config/credentials files present; last worker log 2026-11-06 08:50 UTC).
- All runners execute as the interactive user `amd`; `~/.config/gh/hosts.yml` (live credential) EXISTS in that `$HOME`; no passwordless sudo; `amd` ∈ `video`,`render` (GPU device access); `/dev/kfd` root:render 0660.
- No GPU compute processes at preflight time (host clean).

## 5. Runner-control verdict (drives GPU dispatch policy for R1)

Required controls (env required-reviewers, SHA-pinning enforcement, runner/user isolation) are **not yet established** → **RUNNER HARDENING BLOCKED** (per mission §2.3). Consequence applied for the rest of R1: **no GitHub-dispatched self-hosted GPU jobs**; all GPU validation uses the trusted local execution path on this host (the same path every historical receipt used), with explicit provenance recorded per artifact.

## 6. Baseline acceptance gate

- PASS: all reference SHAs unchanged; ancestry intact; both PRs Draft with green CPU CI; no tags/releases; local tree clean at `261805c`.
- This ledger is the frozen basis for every subsequent R1 phase; any drift discovered later re-triggers §1 re-evaluation before dependent steps.
