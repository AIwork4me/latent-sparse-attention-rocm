# A4 — Repository Security & Branch Protection Audit (proposal only — NOTHING applied)

**Date:** 2026-11-08 · **Actor:** autonomous session (gh authenticated as the owner account — protection-policy changes still require **explicit owner approval** per the mission boundary; this document only proposes exact commands).

## 1. Observed state (verified read-only today)

| Surface | State | Assessment |
|---|---|---|
| `main` branch protection | **NONE** (API 404) | GAP — force-push/deletion of main possible with a push-capable token; published history immutability relies on discipline alone |
| Tag protection | **NONE** (API 404) | GAP — `v*` tags (and therefore release↔tag bindings) are rewritable/deletable |
| Rulesets | none | — (proposal below uses rulesets, the current GitHub mechanism) |
| Required checks | none enforced | `static-and-unit` (GitHub-hosted CPU) green on every recent main push but not required |
| `gpu-regression` environment | required reviewers **present** | ✅ (R1 owner packet item 1 — executed) |
| Repo-wide SHA pinning | `sha_pinning_required: true` | ✅ (R1 owner packet item 2 — executed) |
| Workflow guard R1–R6 | present, **live run PASS** ("no untrusted route to a persistent runner, all actions SHA-pinned, no secrets on non-hosted jobs") | ✅ |
| Pinned-source guard in `cpu-pr.yml` | present — re-runs guard from reviewed SHA `2af35b9` against the PR's workflows (fork cannot neuter it) | ✅ |
| Public fork PR isolation | `cpu-pr.yml` is the only `pull_request` workflow; hosted runners, `contents: read`, zero secrets, loud GPU skips | ✅ |
| Runner scope | runner in `Default` pool — **not** scoped to this repo (org/user-level setting) | GAP → owner UI (R1 packet item 6, restated below) |
| External contributor approval | not configured | GAP (optional) |
| Collaborators with push | `AIwork4me` (admin) only | single-maintainer model confirmed |
| `persist-credentials: false` on all checkouts; concurrency + `timeout-minutes` | verified in both workflows | ✅ |

## 2. Proposed configuration (minimal, single-maintainer, nothing here breaks the owner's own workflow)

Design constraints: the owner is the **only** merger; CPU CI is fast and hosted; GPU CI needs the offline runner + environment approval and therefore must NOT be a required merge check; the owner must retain the ability to merge their own PRs (admin, bypass allowed) and to hotfix-release (tags stay creatable by admins).

### P1 — Ruleset "main-protected" (branch `main`)

```bash
gh api -X POST repos/AIwork4me/latent-sparse-attention-rocm/rulesets \
  -f name=main-protected -f target=branch -f enforcement=active \
  -F 'conditions[ref_name][include][]=refs/heads/main' \
  -F 'rules[][type]=deletion' \
  -F 'rules[][type]=non_fast_forward' \
  -F 'rules[][type]=required_status_checks' \
  -F 'rules[][parameters][required_status_checks][strict]=false' \
  -F 'rules[][parameters][required_status_checks][do_not_enforce_on_create]=false' \
  -F 'rules[][parameters][required_status_checks][checks][][context]=static-and-unit' \
  -F 'bypassactors[]…' # see note
```
Equivalent UI path: Settings → Rules → Rulesets → New branch ruleset → target `main` → Require status checks (`static-and-unit`, non-strict so the release-day pattern "push bump commit then tag after CI" still works) → Block deletions → Block non-fast-forward.

Effects: force-push and deletion of `main` become impossible for normal tokens (admin bypass retained by default — deliberately, so the single maintainer is never locked out); merges require green hosted CPU CI. `strict=false` keeps "branch up to date" off — required for a one-person repo merging sequential PRs without churn. The GPU job `gates-and-kernels` is deliberately NOT required (runner offline ⇒ would deadlock all merges; its gate is the environment approval, kept).

Note on bypass: by default repo admins bypass branch rulesets — for a single-maintainer repo this is the desired fail-open-for-owner posture; if the owner prefers, add themselves explicitly and set `enforcement=evaluate` first to observe for a week before `active`.

### P2 — Tag protection for release tags

```bash
gh api -X POST repos/AIwork4me/latent-sparse-attention-rocm/tags/protection \
  -f pattern='v*' -F 'allow_deletion=false' -F 'allow_force_push=false' 2>/dev/null \
  || gh api -X POST repos/AIwork4me/latent-sparse-attention-rocm/rulesets \
       -f name=release-tags -f target=tag -f enforcement=active \
       -F 'conditions[ref_name][include][]=refs/tags/v*' \
       -F 'rules[][type]=non_fast_forward' -F 'rules[][type]=deletion'
```
(Tag-protection REST supports pattern + deletion/force flags on current GitHub; the ruleset fallback covers tag targets if the legacy endpoint is unavailable.) Effect: `v0.2.1`/`v0.3.1` and future release tags can never be rewritten or deleted — converts the mission invariant "historical tags are immutable" into an enforced control. Rollback: DELETE the rule / ruleset.

### P3 — External-contributor first-run approval (UI only, no REST field)

Settings → Actions → General → "Require approval for all external contributors". Fork PR workflows never run until approved — complements the guard's structural isolation. Rollback: uncheck.

### P4 — Runner group scoping (UI; R1 packet item 6 restated)

Settings → Actions → Runners → group of `lsa-rocm-gfx1151` → "Selected repositories" = `latent-sparse-attention-rocm` only + require approval for external contributors. Structural fix for the F1 residual (a future mis-scoped runner can't be leased by another repo's workflow). Rollback: switch group back to "All repositories".

## 3. What is deliberately NOT proposed

- Requiring PR reviews (≥1 approval) — would block the sole maintainer from merging their own PRs; the required hosted check + admin-protected history is the proportionate control.
- Making `gates-and-kernels` required — runner is intentionally offline pending migration.
- `strict=true` on status checks / linear history / signed commits — process weight with no matching threat for a one-person scientific repo.
- Restricting `allowed_actions` to a list — `sha_pinning_required` already enforces the property that matters.

## 4. Verification commands (after the owner applies P1/P2)

```bash
gh api repos/AIwork4me/latent-sparse-attention-rocm/rulesets --jq '.[] | {name, enforcement}'
gh api repos/AIwork4me/latent-sparse-attention-rocm/tags/protection
# negative test (expect rejection): git push origin +refs/heads/main:refs/heads/main --force (on a throwaway clone)
# positive test: open a draft PR from a throwaway branch → merge button blocked until static-and-unit green
```

**Status: proposal complete — OWNER_ACTION_REQUIRED to apply P1–P4.** Nothing was modified.
