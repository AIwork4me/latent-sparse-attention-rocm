# R1 Runner Security — preflight, control gaps, migration design (NOT executed)

Date: 2026-11-07 · Scope: `~/actions-runner-lsa` self-hosted gfx1151 runner + repository Actions controls.
Standing verdict: **RUNNER HARDENING BLOCKED** — the required controls (§2.3) are not yet established on the GitHub side, so **no GitHub-dispatched self-hosted GPU job will be sent during R1**. All R1 GPU validation runs through the trusted local execution path (same host, same venv, explicit provenance per artifact).

## 1. Read-only preflight (facts, no secret values read or printed)

### 1.1 Runner processes and installs

| Install | Repo/purpose | Process state | Launch mode |
|---|---|---|---|
| `/home/amd/actions-runner` | AIwork4me/Qwen3-TTS-ROCm | `Runner.Listener` running (PID 336554, ~4.6 d) | systemd `github-runner.service` (system) |
| `/home/amd/actions-runner-ovamd` | OpenVINO-Notebooks-on-AMD | `Runner.Listener` running (PID 346183, ~4.5 d) | systemd `github-runner-ovamd.service` (system) |
| `/home/amd/actions-runner-lsa` | **AIwork4me/latent-sparse-attention-rocm** (this repo) | **OFFLINE** — no `Runner.Listener` process; `.runner`/`.credentials`/`.credentials_rsaparams` present; last `_diag` worker log 2026-11-06 08:50 UTC | historically `nohup ./run.sh` (not systemd) |

The two sibling runners are **out of scope** for any R1 action; they must remain untouched.

### 1.2 User, credentials, devices

- All runners execute as the interactive desktop user `amd` (uid 1000; groups include `sudo` (password required — `sudo -n` fails), `video`, `render`, `docker`).
- Credentials present in `$HOME` (existence only): `~/.config/gh/hosts.yml` (live gh CLI token — the top host-side risk), `~/.gitconfig`, `~/.ssh/`.
- GPU device nodes: `/dev/kfd` `root:render 0660`, `/dev/dri/renderD128` + `/dev/dri/card1` `root:video|render 0660`; `amd` reaches them via `render`/`video` groups — a dedicated runner user needs exactly these two groups and nothing else.
- GPU compute processes at preflight: none (host clean; matches the benchmark-hygiene rule).
- No passwordless sudo → limited root-escalation path (unchanged since the 2026-11-06 audit).

### 1.3 Repository Actions controls (read-only API, 2026-11-07)

| Control (mission §2.3) | Observed | Status |
|---|---|---|
| CPU CI only for untrusted PRs | `cpu-pr.yml` is the only `pull_request` workflow; GitHub-hosted, `contents: read`, zero secrets, `LSA_SKIP_GPU_TESTS=1` | ✅ PASS |
| No untrusted PR → self-hosted | **initially "✅ PASS" — falsified by the adversarial review (finding F1): the guard only matched the literal `self-hosted` label, so custom label sets (`[linux, rocm, gfx1151]`), `${{ matrix.runner }}` expressions and `workflow_call` composition bypassed it (all reproduced). FIXED in the R1 adversarial-fixes commit: hosted-label allowlist (only `ubuntu-*`/`macos-*`/`windows-*` are hosted; everything else, incl. expressions and unknown labels, is treated as persistent-host), new rule R6 (workflow_call + non-hosted jobs requires a reviewed allowlist marker), R5 extended to workflow-level `env:`/`defaults:` (finding F2); guard now catches every reproduced bypass (16/16 unit tests incl. 6 new attack fixtures)**; real workflows still pass | ⚠️→✅ FIXED (code); **runner-group scoping remains an owner action (§3 item 6)** |
| Guard integrity in fork PRs | **finding F3: the guard ran from the PR's own checkout, so a fork PR could neuter it in the same PR. FIXED: `cpu-pr.yml` now re-runs the guard from `origin/main` (reviewed copy) against the PR's workflows** | ⚠️→✅ FIXED |
| `pull_request_target` prohibited | guard R2; none present | ✅ PASS |
| SHA-pinned actions | all `uses:` pinned to 40-hex SHAs (checkout/setup-python/upload-artifact) | ✅ PASS (files) |
| Repo-wide SHA-pin enforcement | `sha_pinning_required: false`, `allowed_actions: all` | ⚠️ GAP (owner setting) |
| `persist-credentials: false` | on every checkout incl. GPU workflow | ✅ PASS |
| Minimal workflow permissions | `contents: read` (+`issues: write` only for the hosted upstream-watch, via ephemeral `github.token`) | ✅ PASS |
| No secrets to self-hosted jobs | guard R5; none present | ✅ PASS |
| **Protected `gpu-regression` environment** | environment exists; **protection_rules = [], reviewers = none, branch policy = none** | ❌ **GAP (owner setting)** |
| Required review before trusted GPU dispatch | depends on the environment reviewers — absent | ❌ GAP |
| Explicit ref confirmation on dispatch | `confirm_ref` fat-finger guard in-workflow (verified in file + by run 42139650668 negative test) | ✅ PASS |
| Concurrency + bounded timeouts | `gpu-regression` group no-cancel + `timeout-minutes: 120`; CPU per-ref cancel | ✅ PASS |

### 1.4 Guard validation (adversarial, safe local execution)

- Guard unit suite `tests/test_workflow_security.py`: **10/10 PASS** (ROCm venv python and CPU venv python).
- R1-independent adversarial fixtures (this session, `/tmp/adv-workflows`, not committed): `pull_request`→self-hosted, `pull_request_target` anywhere, `workflow_run`→self-hosted, floating `@v4` tag, floating `docker://` tag, `secrets.*` in a self-hosted job → guard reported **6/6 violations with correct rule attribution (R1×2, R2, R4×2, R5)** and a clean look-alike of the real GPU workflow passed. Exit code 1 with findings as designed.

## 2. Minimum-privilege migration design (prepared; **requires owner authorization — not executed**)

Goal: run this repo's runner as a dedicated non-interactive OS user with no access to `amd`'s credential store, minimum device rights, systemd-managed, fully reversible, without touching the two sibling runners or losing GPU access.

Design decisions:

- New system user `lsa-runner` (name verified available), `nologin` shell, home `/home/lsa-runner` (empty by construction → no gh credential), groups `video,render` only (verified sufficient for `/dev/kfd` + `/dev/dri`).
- Runner relocated by **copy** (not move) to `/opt/actions-runner-lsa` — the old `~/actions-runner-lsa` stays intact as the rollback path.
- The ROCm venv the GPU workflow uses (`~/Desktop/lsa-rocm-eval-starter/.venv`, 9.1 GiB; 76 GiB free) is **copied to `/opt/lsa-rocm/.venv`** and `rocm-regression.yml`'s "Locate the ROCm venv" step points there (one-line, guard-covered workflow change in a normal PR). The runner user never needs `/home/amd` at all.
- systemd unit (drafted; `systemd-analyze verify` **PASS** on syntax with stand-in ExecStart — the only pre-migration warning was the not-yet-existing install path, which the migration creates): hardening = `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`, `ProtectHome=true` (hides `/home/amd`), `ReadWritePaths=/opt/actions-runner-lsa /opt/lsa-rocm`, `DeviceAllow=/dev/kfd rw /dev/dri/renderD128 rw /dev/dri/card1 rw`, `Restart=on-failure`.
- Runner work dir stays inside `/opt/actions-runner-lsa/_work` (repo checkout happens there; the persistent venv lives outside the workspace exactly as today).

### 2.1 Exact migration commands (owner-authorized execution only)

```bash
# A. preflight (already done, repeatable, non-destructive)
id lsa-runner 2>/dev/null && echo "name taken - STOP" || echo "name free"
df -h /opt | tail -1            # need >= 12 GiB
ls -l /dev/kfd /dev/dri/renderD128 /dev/dri/card1

# B. create the dedicated user (no sudo group, no shell)
sudo useradd -r -m -d /home/lsa-runner -s /usr/sbin/nologin -G video,render lsa-runner

# C. copy runner install + venv (originals untouched = rollback)
sudo mkdir -p /opt/actions-runner-lsa /opt/lsa-rocm
sudo rsync -a /home/amd/actions-runner-lsa/ /opt/actions-runner-lsa/
sudo rsync -a /home/amd/Desktop/lsa-rocm-eval-starter/.venv/ /opt/lsa-rocm/.venv/
sudo rm -rf /opt/actions-runner-lsa/_work /opt/actions-runner-lsa/_diag   # stale amd-owned runtime state
sudo chown -R lsa-runner:lsa-runner /opt/actions-runner-lsa /opt/lsa-rocm

# D. systemd service (unit content: this file §2.2; ExecStart=/opt/actions-runner-lsa/bin/runsvc.sh)
sudo cp github-runner-lsa.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now github-runner-lsa.service
systemctl status github-runner-lsa.service --no-pager | head -5

# E. workflow path update (normal PR, guard-covered, reviewed):
#    rocm-regression.yml: PY="/opt/lsa-rocm/.venv/bin/python" (replaces $HOME/Desktop path)

# F. validation (read-only)
sudo -u lsa-runner test ! -r /home/amd/.config/gh/hosts.yml && echo "credential isolation OK"
sudo -u lsa-runner /opt/lsa-rocm/.venv/bin/python -c "import torch; assert torch.cuda.is_available() and torch.version.hip; print('GPU OK')"
sudo -u lsa-runner ls /opt/actions-runner-lsa >/dev/null && echo "install OK"
# then a single workflow_dispatch of rocm-regression.yml on main as the end-to-end proof
```

### 2.2 Drafted unit (verified syntax offline; copy lives at `/tmp/github-runner-lsa.service` this session)

```ini
[Unit]
Description=GitHub Actions runner (lsa-rocm gfx1151, dedicated lsa-runner user)
After=network-online.target
Wants=network-online.target

[Service]
User=lsa-runner
Group=lsa-runner
WorkingDirectory=/opt/actions-runner-lsa
ExecStart=/opt/actions-runner-lsa/bin/runsvc.sh
Restart=on-failure
RestartSec=10
KillMode=process
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/actions-runner-lsa /opt/lsa-rocm
DeviceAllow=/dev/kfd rw
DeviceAllow=/dev/dri/renderD128 rw
DeviceAllow=/dev/dri/card1 rw

[Install]
WantedBy=multi-user.target
```

Dry-run results (this session): name free ✅ · disk sufficient (76 GiB ≥ 12 GiB need) ✅ · device nodes present with expected ownership ✅ · unit syntax `systemd-analyze verify` PASS ✅ · no execution performed (no `useradd`/`systemctl`/`sudo`).

### 2.3 Rollback (exact, reversible at every step)

```bash
sudo systemctl disable --now github-runner-lsa.service     # stop new runner
sudo rm /etc/systemd/system/github-runner-lsa.service && sudo systemctl daemon-reload
# revert the one-line workflow path PR (normal git revert)
# old runner restart (original install never modified):
cd ~/actions-runner-lsa && nohup ./run.sh &                 # back to amd-user operation
# optional cleanup AFTER rollback is confirmed stable:
# sudo userdel -r lsa-runner; sudo rm -rf /opt/actions-runner-lsa /opt/lsa-rocm
```

Post-migration owner step (separate approval, policy §4 item 4): rotate the `amd` gh CLI credential (`gh auth logout` / remove `~/.config/gh/hosts.yml`) once nothing legit depends on it.

## 3. Owner action packet (the two settings gaps + migration approval)

| # | Action | Command / UI path | Risk | Rollback |
|---|---|---|---|---|
| 1 | Required reviewers on `gpu-regression` env | Settings → Environments → gpu-regression → Required reviewers (or `gh api -X PUT repos/AIwork4me/latent-sparse-attention-rocm/environments/gpu-regression` with reviewers) | none (adds an approval click per dispatch) | remove reviewers |
| 2 | Enforce SHA pinning | `gh api -X PUT repos/AIwork4me/latent-sparse-attention-rocm/actions/permissions -f sha_pinning_required=true` | none (all current workflows already comply) | same call with `false` |
| 3 | Approve + execute §2.1 migration | sudo on host (owner-run) | medium (host change; reversible per §2.3; siblings untouched) | §2.3 |
| 4 | After 3: rotate gh credential | `gh auth logout` on `amd` | low after 3 | re-auth |
| 5 | (optional) "Require approval for all external contributors" + `allowed_actions: selected` | Settings → Actions | low | revert setting |
| 6 | **Scope the runner group** (adversarial F1 residual): runner settings → group of `lsa-rocm-gfx1151` → "Selected repositories" = this repo only; require approval for external contributors | org/repo runner settings UI | low | revert group scope |

Until items 1–3 land (and given the F1 history, item 6 is strongly advised before the runner ever comes online): **RUNNER HARDENING BLOCKED stands** — no GitHub-dispatched GPU jobs; trusted local execution only, as used throughout R1.
