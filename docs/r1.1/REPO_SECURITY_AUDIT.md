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

Revised 2026-11-08 (handoff-review session): the first draft's `bypassactors[]…` placeholder was
not a runnable request, and its claim "admin bypass retained by default" was **factually wrong —
rulesets do NOT auto-bypass repository admins** (unlike classic branch protection; bypass must be
granted explicitly via `bypass_actors`, [docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)).
The command below is a complete, valid JSON-body request (auditable, no `gh` magic-field expansion):

```bash
gh api -X POST repos/AIwork4me/latent-sparse-attention-rocm/rulesets --input - <<'EOF'
{
  "name": "main-protected",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "required_status_checks",
     "parameters": {"strict": false, "do_not_enforce_on_create": false,
                    "checks": [{"context": "static-and-unit"}]}}
  ],
  "bypass_actors": []
}
EOF
```

Bypass decision for a single-maintainer repo (choose ONE before applying):

- **Option A — no bypass (recommended, as written above; `bypass_actors: []`)**: `main` becomes
  immutable for everyone **including the owner**. The emergency escape is disabling/editing the
  ruleset itself — always available to a repo admin, and each such change lands in the audit log,
  which is exactly the evidence trail an immutability policy wants. No silent path around history.
- **Option B — explicit owner bypass**: add before the closing `]`:
  `{"actor_id": <id>, "actor_type": "User", "bypass_mode": "always"}` where
  `<id> = gh api user --jq .id` evaluated at apply time. Keeps the owner's force-push ability
  (e.g. disaster repair) at the cost of a silent-bypass path existing.

Equivalent UI path: Settings → Rules → Rulesets → New branch ruleset → target `main` → Require status checks (`static-and-unit`, non-strict so the release-day pattern "push bump commit then tag after CI" still works) → Block deletions → Block non-fast-forward.

Effects: force-push and deletion of `main` become impossible for every token/pat (Option A: for everyone; rollback = disable the ruleset, audited); merges require green hosted CPU CI. `strict=false` keeps "branch up to date" off — required for a one-person repo merging sequential PRs without churn. The GPU job `gates-and-kernels` is deliberately NOT required (runner offline ⇒ would deadlock all merges; its gate is the environment approval, kept).

Note on bypass: ~~by default repo admins bypass branch rulesets~~ — **incorrect as of the 2026-11-08 revision** (rulesets never auto-bypass admins; see the Option A/B decision above). If unsure, apply with `"enforcement": "evaluate"` first to observe a week of matching activity, then flip to `active`.

### P2 — Tag protection for release tags

Revised 2026-11-08: the legacy tag-protection REST endpoint (`POST /repos/{o}/{r}/tags/protection`)
was **removed after 2024-08-30** (sunset notice 2024-05-29; existing protections migrated to
rulesets) — the first draft led with a dead operation. The current mechanism is a tag-target
ruleset (rules `deletion` + `non_fast_forward`; `creation` deliberately NOT restricted so the
owner can still cut release tags):

```bash
gh api -X POST repos/AIwork4me/latent-sparse-attention-rocm/rulesets --input - <<'EOF'
{
  "name": "release-tags",
  "target": "tag",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"}
  ]
}
EOF
```
Effect: `v0.2.1`/`v0.3.1` and future release tags can never be rewritten or deleted — converts the mission invariant "historical tags are immutable" into an enforced control. Rollback: `gh api -X DELETE repos/AIwork4me/latent-sparse-attention-rocm/rulesets/<id>` (id from the verification listing below).

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
# both rulesets present and active (lists id + enforcement — the id is the rollback handle):
gh api repos/AIwork4me/latent-sparse-attention-rocm/rulesets --jq '.[] | {id, name, target, enforcement}'
# per-ruleset detail (rules actually registered):
gh api "repos/AIwork4me/latent-sparse-attention-rocm/rulesets/<id>" --jq '{name, rules, conditions}'
# negative tests (expect rejection): on a throwaway clone —
#   force-push main:    git push origin +refs/heads/main:refs/heads/main --force
#   delete a v* tag:    git push origin :refs/tags/v0.3.1   (throwaway tag first!)
# positive test: open a draft PR from a throwaway branch → merge blocked until static-and-unit green
```
(The legacy `GET /repos/{o}/{r}/tags/protection` check is gone with the endpoint — verify tags via the `release-tags` ruleset above.)

**Status: proposal complete — OWNER_ACTION_REQUIRED to apply P1–P4.** Nothing was modified.
