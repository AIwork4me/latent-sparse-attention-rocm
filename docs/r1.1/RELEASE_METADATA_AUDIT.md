# A1 — Release Metadata Audit (v0.2.1 / v0.3.1 published bodies corrected)

**Executed:** 2026-11-08 ~10:05 +0800 (02:05 UTC), autonomous session under mission R1.1 authorization
("correct the already-published v0.2.1/v0.3.1 GitHub Release descriptions while preserving scientific claims").
**Archive:** `docs/r1.1/release-archive/` — `*-body-before.md` (fetched pre-edit), `*-body-after.md` (intended), `*.diff`.

## Method

1. Fetched both published bodies via `gh release view <tag> --json body` → archived verbatim.
2. Audited every count/SHA against the authoritative in-repo records (`docs/release-closure/R1_PR1_REVIEW.md`, `R1_PR5_REVIEW.md`, `R1_FINAL_REPORT.md`) — not against the bodies' own claims.
3. Edited only defective metadata language; **all scientific claims, numbers, bands, negative results and the topk 1.13×-receipt-backed-vs-1.24×-development framing preserved verbatim**.
4. Applied via `gh release edit <tag> --notes-file`, re-fetched published bodies, diffed against intent.

## Defects found and corrected

### v0.2.1 (3 edits)
| # | Defect | Correction |
|---|---|---|
| 1 | Header read "DRAFT — publication requires owner approval; tag target … to be recorded" on a **published** release | "FINAL — published 2026-11-07T14:17:36Z under owner approval; tag `v0.2.1` → PR #1 merge commit `7b0b06f…`" |
| 2 | Verifier note said `scripts/r1_verify_receipts.py` "lives on the v0.3.x branch (PR #5) … run it from a checkout of that branch" — obsolete since PR #5 merged to main | Now: script added to main by PR #5, post-dates the v0.2.1 tag tree — run from a main/v0.3.1 checkout |
| 3 | Provenance heading "(filled at tag time)" | "(recorded at publication)" |

Verified-correct and intentionally **unchanged**: post-merge counts at the merge tree (CPU CI 117/16/0 — confirmed by `R1_FINAL_REPORT.md` §75 as *post*-merge, not mixed), 16/16 gates, 133 gfx1151 tests, 24/24 evidence, 111/111 audit recomputes, SUPERSEDED v0.2.0 banner, corrected dense-baseline defect description, all performance bands.

### v0.3.1 (4 edits)
| # | Defect | Correction |
|---|---|---|
| 1 | Same DRAFT header on the published Latest release | "FINAL — published 2026-11-07T14:26:43Z, Latest, under owner approval; tag `v0.3.1` → version-bump commit `0f1abeb…`" |
| 2 | Correctness section led with **pre-merge** counts (PR #5 HEAD: 186 gfx1151 tests, 135 CPU CI) as if current | Leads with merged-tree `958d948` authority: 17/17 both impls re-verified, **192 gfx1151 tests at the merged tree** (186 noted as pre-merge), **CPU CI 141/46/0** (135 noted as pre-merge) |
| 3 | "(filled at tag time)" | "(recorded at publication)" |
| 4 | Pre-tag hygiene bullet was a live instruction ("version bump to 0.3.1 (currently 0.2.0) … verify present") — already executed before tagging | "EXECUTED before tagging: bumped to 0.3.1 + CHANGELOG entries in commit `0f1abeb` (this tag's target)" |

Preserved verbatim: 4-SHA provenance distinctions (perf code `60c004b` / PR merge `958d948` / tag `0f1abeb` / post-merge verification `958d948`), 71/71 checksums, 153/153 recomputes, receipt-backed **1.13× decode-select-stage** as the primary topk claim with 1.24× explicitly labeled a non-retained development observation, negative results (page-batched beam, tree 0.93×, SGLang BLOCKED_ON_REASSESS), 42-receipt counting rule, hardware/software pin.

## Post-edit verification (re-fetched from GitHub)

- `diff after vs published`: byte-identical except one trailing blank line at EOF on each (GitHub API normalization artifact) — confirmed on both releases.
- `grep -c DRAFT` on published bodies: **0 / 0**.
- Tags after edits: `v0.3.1 → 0f1abeb6a88c84123d06fde7654a01d5254153d2`, `v0.2.1 → 7b0b06f14578a5b9217d01510621bd38ab78a091` — **unchanged** (annotated tag objects `d1679ad…`/`732228c…` untouched, no tag rewritten, no release deleted).
- Release targets/assets: no assets existed; release↔tag bindings unchanged.

**Acceptance: MET** — both public Releases carry final publication language and correct historical provenance; tags unchanged.
