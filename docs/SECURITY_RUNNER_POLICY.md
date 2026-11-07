# Self-Hosted GPU Runner Security Policy (gfx1151)

Status: **enforced for workflow routing since PR #1 (F10)** — owner action items listed at the end.
Audited: 2026-11-06, live repository + host inspection (read-only).

## 1. Threat model

This repository is **public** (`AIwork4me/latent-sparse-attention-rocm`). Any
internet user can open a pull request. Pull-request code is **untrusted**: a
fork PR's `.github/workflows/*`, `scripts/*` and test code are attacker-controlled
artifacts whose execution goal would be:

1. Read the runner user's `$HOME` — notably `~/.config/gh/hosts.yml` (live
   GitHub CLI credential), SSH keys, browser profiles, other repositories.
2. Tamper with the shared ROCm venv (`~/Desktop/lsa-rocm-eval-starter/.venv`)
   or sibling checkouts to poison later "trusted" runs (persistent host ⇒
   persistence for the attacker).
3. Pivot to the other self-hosted runner projects on the same host.
4. Burn GPU/disk/network resources, or exfiltrate any secret a workflow hands
   to the job.

**Human review approval of a PR does not sandbox its code.** Approval only
makes GitHub willing to run it; the runner-side exposure is identical. This is
why routing — not review — is the control.

## 2. Host exposure audit (2026-11-06, facts)

| Fact | Observation | Risk |
|---|---|---|
| Repo visibility | PUBLIC | untrusted PRs reachable |
| Runner process | `~/actions-runner-lsa/bin/Runner.Listener` (name `lsa-rocm-gfx1151`, labels `rocm,gfx1151`), **online**, started with `nohup ./run.sh` as user `amd` | jobs run as the interactive desktop user |
| Runner user | `amd` — same user that owns the gh CLI config, the working tree, the venv and two sibling runner installs (`~/actions-runner` for AIwork4me/Qwen3-TTS-ROCm, `~/actions-runner-ovamd`) | full user-level read/write; **no sandbox** |
| Credentials in `$HOME` | `~/.config/gh/hosts.yml` present (live token) | a single untrusted job = credential compromise |
| Privilege escalation | no passwordless sudo (`sudo -n -l` asks for a password) | limits root compromise (good) |
| Repository Actions settings | `allowed_actions: all`, `sha_pinning_required: false` (pre-fix) | floating action tags possible |
| Prior routing | `rocm-regression.yml` triggered on **`pull_request`** (any fork PR) → self-hosted gfx1151 | **the F10 P0 hole, now closed** |

## 3. Enforced routing rules (repository controls)

1. **Public PRs run GitHub-hosted CPU only.** `cpu-pr.yml` is the only
   `pull_request`-triggered workflow: ephemeral `ubuntu-latest`, permissions
   `contents: read`, zero secrets, GPU tests skip loudly (`LSA_SKIP_GPU_TESTS=1`
   / no GPU visible), concurrency-cancelled per ref.
2. **The persistent gfx1151 runner is reachable only from the trusted GPU
   workflow.** `rocm-regression.yml` triggers on `push` to `main` and on
   `workflow_dispatch` with an explicit **ref confirmation** (fat-finger guard)
   and a recorded reason. `environment: gpu-regression` allows the owner to add
   required reviewers to every dispatch.
3. **`pull_request_target` and `workflow_run`→self-hosted are prohibited** and
   machine-checked. `scripts/check_workflow_security.py` (rules R1–R5) fails CI
   on: any untrusted-event trigger routed to a self-hosted runner; any
   `pull_request_target`; any `uses:` not pinned to a 40-hex commit SHA (or
   digest-pinned `docker://`); any `${{ secrets.* }}` referenced from a
   self-hosted job. It runs in **both** CI workflows, so a PR that tries to add
   an unsafe route fails its own CPU checks.
4. **All actions are SHA-pinned** (`actions/checkout@11d5960…` v4.3.0,
   `actions/setup-python@a26af69…` v5.6.0,
   `actions/upload-artifact@ea165f8…` v4.6.2). The runner jobs never receive
   repo secrets (`persist-credentials: false` on checkout; no `secrets.` use).
5. **GPU evidence is uploaded, not published**: receipts ride as artifacts
   (90-day retention) with SHA256SUMS written by the campaign runner; the
   independent auditor (F12) must re-approve before any number reaches a
   headline table.
6. **Concurrency**: `gpu-regression` group, no cancel-in-progress — one heavy
   campaign at a time; CPU checks cancel superseded runs per ref.

### Residual risks the repository cannot code away

- A maintainer clicking "approve and run" on a **fork PR that adds a new
  workflow file** would let that workflow run (GitHub first-time-contributor
  approval). Policy: maintainers must never approve workflow-file changes from
  forks; the R1–R5 guard makes such a PR self-fail, but the guard only runs
  *after* approval. The `environment` approval on GPU dispatches is the second
  lock.
- The runner executes trusted code as `amd` with read access to `~/.config/gh`.
  Trusted code is reviewed, but defense-in-depth items below shrink this.

## 4. Owner action items (reversible, no secret values needed)

1. **Add required reviewers to the `gpu-regression` environment** (GitHub →
   Settings → Environments → gpu-regression → Required reviewers). Zero-risk;
   gates every dispatch.
2. **Require SHA pinning repo-wide**: `gh api -X PUT
   repos/AIwork4me/latent-sparse-attention-rocm/actions/permissions -f
   sha_pinning_required=true` (all workflows already comply; rollback = same
   call with `false`).
3. **Move the runner out of the interactive user** (needs sudo, owner-executed):
   dedicated `actions-runner` OS user owning `~/actions-runner-lsa`, systemd
   unit (`svc.sh install`), `$HOME` without gh credentials, no shell profile
   inheritance. Until then, treat any job on the runner as `amd`-equivalent.
4. **Rotate the gh CLI credential** after the runner user is separated (or run
   `gh auth logout` / remove `~/.config/gh/hosts.yml` from that `$HOME`).
5. Optional hardening: Settings → Actions → "Require approval for all external
   contributors" (UI-only), and set `allowed_actions: selected` with an
   allowlist once the workflow set is stable.

## 5. Validation performed

- Guard unit tests: `tests/test_workflow_security.py` (10 tests) — adversarial
  fixtures for every rule, live-workflow assertion, CLI exit codes.
- Live `.github/workflows` passes R1–R5 after this change (the check itself
  caught the unpinned `upstream-watch.yml` checkout during development).
- Post-push validation: the `pull_request` event must appear **only** on
  `cpu-pr.yml` runs; the GPU workflow must show only `push`(main)/`workflow_dispatch`
  runs. Recorded in `docs/reviews/PR1_FINAL_INDEPENDENT_REVIEW.md`.
