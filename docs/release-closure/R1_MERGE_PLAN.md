# R1 Merge & Release Plan — owner-ready, zero steps executed without approval

Date: 2026-11-07 · Every action below that touches GitHub state (merge, retarget, tag, release, settings) or the host (runner migration) is **owner-gated**. Prerequisite audits: R1_BASELINE (state frozen), R1_RUNNER_SECURITY (RUNNER HARDENING BLOCKED — controls pending), R1_PR1_REVIEW, R1_PR5_REVIEW, R1_EVIDENCE_AUDIT.

## 0. Approval ledger (what R1 will ask, in order)

| # | Action | Risk | Reversible? | Gate required before asking |
|---|---|---|---|---|
| A1 | Add required reviewers to `gpu-regression` env | none | yes (remove reviewers) | none (R1_RUNNER_SECURITY §3) |
| A2 | `sha_pinning_required=true` | none (workflows comply) | yes | none |
| A3 | Execute runner migration (R1_RUNNER_SECURITY §2.1) + rotate gh cred | medium (host) | yes (§2.3 rollback) | A1+A2 done |
| A4 | Merge PR #1 (merge commit) | normal merge | revert commit | R1_PR1_REVIEW = GO/CONDITIONAL GO + owner says the exact SHA |
| A5 | Tag `v0.2.1` + GitHub Release (after post-merge receipt) | public | delete tag/release | §4 checklist below |
| A6 | Retarget PR #5 → main | none | switch base back | A4 done |
| A7 | Merge PR #5 (merge commit) | normal merge | revert commit | §8 checklist below |
| A8 | Tag `v0.3.1` + GitHub Release (after merged-tree validation §7) | public | delete tag/release | §7 record complete |

## 1. Controlled merge of PR #1 (owner approval A4) — prefer MERGE COMMIT

Pre-flight (re-run at execution time; abort on any drift from R1_BASELINE):
```bash
gh pr view 1 --json headRefOid,isDraft,mergeable,mergeStateStatus   # expect 79412f8…, true, MERGEABLE, CLEAN
git fetch origin && git rev-parse origin/main                        # expect 5f2918e3…
gh run list --branch v0.2.1-scientific-validity --limit 2            # both SUCCESS
git merge-base origin/main origin/v0.2.1-scientific-validity          # record
```
Execute (owner names the SHA aloud/on-record): `gh pr merge 1 --merge --subject "Merge PR #1: v0.2.1 scientific-validity closure (79412f8)"`
**Never** squash/rebase/force-push (PR #5 stacks on this history — merge-commit preserves the ancestor relationship so PR #5's diff stays reviewable).

Post-merge (immediately, record everything):
1. `MAIN_SHA=$(git rev-parse origin/main)` — the actual merge commit; verify `git merge-base --is-ancestor 79412f8 $MAIN_SHA`.
2. Fresh tree checks on `$MAIN_SHA`: `git worktree add --detach ~/wt-postmerge1 $MAIN_SHA` → CPU CI suite (expect 117 passed/16 skipped), static guard, **GPU 16-gate manifest run on gfx1151 (trusted local path) → must be 16/16**, full pytest on GPU host.
3. Receipt provenance: this is a NEW execution at `$MAIN_SHA` — label run id `r1-postmerge1-<ts>Z`, record tree SHA (`git rev-parse $MAIN_SHA^{tree}`), ROCm/torch/triton versions, upstream sha. **Historical `60c004b`/`82c891e` receipts stay pinned to their SHAs and are never relabeled.**
4. If (and only if) all green: v0.2.1 release notes + provenance + checksums from R1_RELEASE_NOTES_v0.2.1.md; tag step is A5.

Tag/Release (A5 — separate explicit approval): `git tag -a v0.2.1 -m "…" $MAIN_SHA && git push origin v0.2.1` then `gh release create v0.2.1 --draft --notes-file docs/release-closure/R1_RELEASE_NOTES_v0.2.1.md` (**draft first**; publish is another owner click). Rollback: `git push origin :refs/tags/v0.2.1 && git tag -d v0.2.1`; `gh release delete v0.2.1`.

## 2. Retarget PR #5 → main (A6, after A4)

```bash
gh api -X PATCH repos/AIwork4me/latent-sparse-attention-rocm/pulls/5 -f base=main   # REST (GraphQL pr-edit is deprecated)
```
Then verify: `base=main`; `mergeable`; PR #1-only commits vanished from the review diff (`gh pr diff 5 --name-only` no longer lists PR #1-exclusive files — PR #1 commits become ancestors); v0.3.0/v0.3.1 changes intact (`5a13bf0`,`0908842`,`60c004b` still in `main..head`); CPU CI re-runs green on the retargeted PR. Rollback: same PATCH with `base=v0.2.1-scientific-validity` (valid while that branch exists — `delete_branch_on_merge: false` keeps it).

## 3. Final merged-tree GPU validation of PR #5 (§7 — hard gate, after A7)

At the actual post-merge main SHA `$M` (record commit + tree SHA + full env identity): CPU CI suite; **17/17 gates default AND `LSA_TREE_IMPL=batched`** on gfx1151 (fresh run ids `r1-postmerge5-<ts>Z`); local pytest (expect 186); `sha256sum -c evidence/SHA256SUMS` (71/71); `scripts/r1_verify_receipts.py` over receipts-v0.3.1-final + audit (153/153). Raw stdout/stderr + exit codes archived under `docs/release-closure/runs/`. Any failure ⇒ NO-GO for the tag; report and stop.

## 4. Controlled merge of PR #5 + v0.3.1 release (A7/A8)

Pre-flight: PR #5 HEAD == owner-named SHA (expect `261805c9…` or its post-retarget successor), base=main, CI green, reviewDecision approved by owner, no unresolved CRITICAL/HIGH finding from R1 adversarial review. Execute `gh pr merge 5 --merge`; capture merge SHA; run §3; then tag `v0.3.1` (annotated) + **draft** GitHub Release from R1_RELEASE_NOTES_v0.3.1.md (publish = separate owner click). v0.3.1 is the single release for this PR (it contains v0.3.0 + v0.3.1 work); no separate v0.3.0 tag. Rollbacks: revert merge commit; delete tag/release before publish.

## 5. What R1 deliberately does NOT do without new authorization

No dispatch of `rocm-regression.yml` while RUNNER HARDENING BLOCKED stands (trusted local path only); no workflow edits beyond the reviewed one-line venv-path PR that ships with the migration (A3); no upstream moves (pin `5ebea732` stays); no new kernel optimization; no history rewrites; no secret values read/rotated by R1 itself.
