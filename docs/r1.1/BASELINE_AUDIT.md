# R1.1 / v0.4.0 Baseline Audit — Frozen before any modification

**Audit timestamp:** 2026-11-08 ~10:00 +0800 (02:00 UTC)
**Audited by:** autonomous engineering session (Codex mission R1.1 → v0.4.0)
**Working tree:** `/home/amd/Desktop/lsa-rocm-eval-starter`, clean at audit start (`git status` empty).

## 1. Commands executed (verbatim intent)

```
git status; git log --oneline -8; git remote -v; git branch -a
git fetch origin --tags --prune          # FAILED — see deviations §6
git tag -l --format='%(refname:short) %(objecttype) %(objectname) %(*objectname)'
git cat-file -t <sha>                    # for each audit SHA
git rev-parse origin/main
gh pr list --state all --limit 10
gh release list --limit 10
gh api repos/AIwork4me/latent-sparse-attention-rocm/tags
gh api repos/AIwork4me/latent-sparse-attention-rocm/branches/main
gh api repos/AIwork4me/latent-sparse-attention-rocm/branches/main/protection
gh api repos/AIwork4me/latent-sparse-attention-rocm/environments
gh api repos/AIwork4me/latent-sparse-attention-rocm/actions/permissions
gh run list --limit 15
gh api repos/AIwork4me/latent-sparse-attention-rocm/actions/runs/<id>{,/jobs}
rocm-smi --showid --showclocks --showtemp --showmeminfo vram
rocm-smi --showpidgpus ; ps aux | grep .venv/bin/python
grep -m1 version pyproject.toml ; cat tracking/UPSTREAM_BASELINE.json
.venv/bin/python -c "import torch, triton; ..."
```

## 2. SHA identity and ancestry verification

| Reference | Expected | Observed | Verdict |
|---|---|---|---|
| PR #1 merge | `7b0b06f` | commit `7b0b06f14578a5b9217d01510621bd38ab78a091`, reachable in main history | MATCH |
| PR #5 merge | `958d948` | commit `958d948`, parent chain `99902d7` → main | MATCH |
| v0.2.1 annotated tag | → `7b0b06f` | tag object `732228c26a330794741222181aa66657191cfa7b` → commit `7b0b06f…` (local **and** `gh api …/tags`) | MATCH |
| v0.3.1 annotated tag | → `0f1abeb` | tag object `d1679ad56a5d0755178d99e06363ef7bb81789a9` → commit `0f1abeb6a88c84123d06fde7654a01d5254153d2` (local **and** remote API) | MATCH |
| Observed main | `1ab0e0b` | local `origin/main` = remote API `branches/main` = `1ab0e0bd32679b11f51015f6a2db8c811292fefd` | MATCH, remote unchanged |
| v0.3.1 perf code | `60c004b` | commit `60c004b5630771082ac9ac3a2877262cae1f4ed4` "perf(v0.3.1): topk replaces the select's full-width sort + strided slice" (2026-11-07 01:10 +0800) | MATCH |

No uncommitted work existed; nothing preserved-then-discarded. No force operations performed.

## 3. Pull requests and releases

- PR #1 (v0.2.1 scientific-validity closure) — **MERGED** 2026-11-05T23:10:22Z (merge `7b0b06f`).
- PR #5 (v0.3.0 kernel optimization + v0.3.1 topk select) — **MERGED** 2026-11-06T00:55:40Z (merge `958d948`).
- No open PRs.
- Release v0.2.1 — published 2026-11-07T14:17:36Z, tag `v0.2.1`@`7b0b06f`.
- Release v0.3.1 — published 2026-11-07T14:26:43Z, **Latest**, tag `v0.3.1`@`0f1abeb`.

## 4. CI / runner / security state at freeze

| Item | Observed |
|---|---|
| Latest CPU CI on main | run `42381769795` @ `1ab0e0b` — **success** (2026-11-07T14:28:42Z) |
| GPU run `42381678702` | status `waiting` @ `9a8b87a` (superseded HEAD); job `gates-and-kernels` waiting on offline self-hosted runner labels `[self-hosted, linux, rocm, gfx1151]` |
| GPU run `42381769814` | status `pending` @ `1ab0e0b` (current main HEAD), **no jobs created** — awaiting `gpu-regression` environment approval; cannot execute until owner approves |
| Self-hosted runner | intentionally OFFLINE pending R1 migration (docs/release-closure/R1_RUNNER_SECURITY.md) |
| `gpu-regression` environment | `required_reviewers` protection rule present |
| Actions permissions | `enabled`, `allowed_actions: all`, **`sha_pinning_required: true`** |
| Branch protection on `main` | **NONE** (API 404 "Branch not protected") — recorded; A4 addresses |
| Repo settings | `allow_forking: true`, `delete_branch_on_merge: false` |

## 5. Environment identity

- `pyproject.toml` version = **0.3.1**; upstream pin `tracking/UPSTREAM_BASELINE.json` = `Prefrontal-AI/latent-sparse-attention` @ `5ebea732d639576d2a7579b0de40e6252f9001f5` (reviewed 2026-11-06).
- venv: torch `2.14.0+rocm10.1.0`, triton `3.8.0`, `torch.cuda.is_available() == True`.
- GPU: single `AMD Radeon Graphics` device id `0x1586` rev `0xd1` (gfx1151 / Radeon 8060S), 32 GiB VRAM, idle (no KFD PIDs), 47.0 °C edge, mclk L2 1000 MHz / sclk L1 1509 MHz at audit time, VRAM used 1.20 GiB (display + desktop).
- Host: load avg 4.23/1.66/0.77 (CPU-only, transient), RAM 88 GiB available, root fs 96 % used (76 GiB free — recorded as a constraint, no cleanup of other projects' data permitted).
- No `.venv/bin/python` orphan processes (checked per prior campaign lesson).

## 6. Deviations and notes

1. `git fetch origin` failed repeatedly with `GnuTLS recv error (-110)` while `gh api` over HTTPS succeeded. Remote state was therefore verified through the GitHub REST API (`branches/main`, `tags`) instead of the git transport; local `origin/main` ref (fetched 2026-11-07) matches the API-reported remote HEAD exactly. All push operations later in this mission may need retry/transport diagnosis; no fetch-dependent assumption is made beyond what the API confirmed.
2. Main has no branch protection rule — this was not previously recorded as a live control; the effective controls are the `gpu-regression` environment reviewers and repo-level SHA pinning. A4 proposes a minimal protection config for owner approval.
3. Disk at 96 % — GPU campaigns in Mission B must avoid multi-GiB scratch growth in `/tmp` and reuse `results/`/`evidence/` locations deliberately.

## 7. Baseline ledger snapshot

- Baseline commit for both mission branches: `1ab0e0bd32679b11f51015f6a2db8c811292fefd` (`main`, verified == remote).
- Branches created from it: `chore/r1.1-post-release-hygiene`, `feature/v0.4.0-measured-select-fusion` (independently; v0.4.0 does not stack on R1.1).
- Historical evidence areas `evidence/v0.2.0-archive`, `receipts-v0.3.0*`, `receipts-v0.3.1-final`, `evidence/SHA256SUMS` are treated immutable; new evidence goes to `evidence/v0.4.0/` only.

**Verdict: BASELINE VERIFIED — proceed.**
