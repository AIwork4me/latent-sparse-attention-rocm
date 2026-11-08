# A3 — Runner Migration Validation

**Status: `OWNER_ACTION_REQUIRED`** — every nonprivileged preflight, dry-run and empirical validation step below PASSED and is machine-evidenced; the privileged host steps (useradd, /opt copies, systemd) are owner-console actions per the standing authorization boundary. **The migration is NOT complete** — a dry-run never claims completion.

**Date:** 2026-11-08 · **Starting point:** `docs/release-closure/R1_RUNNER_SECURITY.md` · **Artifacts:** `docs/r1.1/ops/{github-runner-lsa.service, migrate-runner.sh, rollback-runner.sh}` (owner-executable, committed on this branch).

## 1. The five required investigations (all executed nonprivileged)

### Q1 — Does copying the Python venv break shebangs / interpreter paths / editable installs / Triton cache / deps?

**Empirically answered: copy breaks 3 specific things; all 3 are mechanically repairable; the repaired copy is functionally validated.**

Experiment (as `amd`, no privileges, full repro in `ops/migrate-runner.sh` §4):
1. `cp -a .venv /tmp/lsa-venv-copy` (9.1 GiB, 1.8 s — same-fs copy).
2. Found **45** files in `bin/` with hardcoded shebang `#!/home/amd/Desktop/lsa-rocm-eval-starter/.venv/bin/python` → sed-repaired to the new prefix. `pip --version` / `python -m pytest --version` work from the copy.
3. `bin/python3.12 → /usr/bin/python3.12` is an absolute symlink to the **system** interpreter — still valid at any venv location. `pyvenv.cfg` (`home=/usr/bin`, `executable=/usr/bin/python3.12`) — no change needed.
4. **Editable install hazard:** `__editable__.lsa_rocm_lab-0.2.0.pth` hardcodes `/home/amd/Desktop/lsa-rocm-eval-starter/src` — an un-repaired copy would import the lab package from amd's interactive checkout (isolation leak). Removed in the copy; the workflow's `PYTHONPATH: ${{ github.workspace }}/src` makes the removal correct.
5. Functional proof from the relocated venv, isolated cwd, `HOME` and `TRITON_CACHE_DIR` pointed at throwaway dirs:
   `torch 2.14.0+rocm10.1.0 | hip 7.16.26385 | avail True`, a real `@triton.jit` kernel **compiled and executed on the gfx1151 device**, cache populated at the explicit dir. Device: `AMD Radeon 8060S Graphics`.

**Fresh-env alternative rejected on evidence:** `triton==3.8.0+git669b31ac.rocm10.1.0` is a custom build not installable from PyPI; a fresh pinned env needs the exact wheel file plus a multi-GiB ROCm torch re-download, while the copy+repair recipe is now proven. The mission's preference rule ("fresh pinned env **if the copy cannot be validated**") resolves to: **copy, repaired per the validated recipe**.

### Q2 — Do `ProtectHome=true`, the new HOME and the systemd device sandbox permit runner startup and GPU access?

**Two defects found in the R1 draft unit; corrected in `ops/github-runner-lsa.service`:**

1. **`ProtectHome=true` hides ALL of `/home` — including the new `/home/lsa-runner`.** With the draft unit, `$HOME` would be unwritable/invisible → Triton kernel cache (default `$HOME/.triton`), torch extension dir and every home-writing library fail exactly at first kernel compile. Fix: `Environment=HOME=/opt/actions-runner-lsa/home` (under `ReadWritePaths`) + explicit `TRITON_CACHE_DIR` + `PYTORCH_EXTENSIONS_DIR` (belt and braces).
2. **`DeviceAllow` without `DevicePolicy=closed` is inert for filtering** (default policy `auto` allows everything). Fix: `DevicePolicy=closed` + the three `DeviceAllow` entries (`/dev/kfd`, `/dev/dri/renderD128`, `/dev/dri/card1` rw) → a true minimal-GPU allowlist. GPU reachability itself is by `video,render` group membership on the device nodes (`0660 root:video|render`, re-verified), the same mechanism the interactive user uses — no other privilege needed.
3. Runner startup: install is relocatable (no `/home/amd` strings in `.settings/` or `env.sh`; `workFolder: _work` relative); `WorkingDirectory=/opt/actions-runner-lsa` + `ReadWritePaths` covers listener state; outbound HTTPS to the Actions service is unrestricted. Syntax of the R1 draft passed `systemd-analyze verify`; the three-line delta preserves syntax (re-verify at install time — script does).

### Q3 — Can copied registration files create duplicate listeners or conflicting identities?

**Yes, if misused — mitigated by design.** `.runner` encodes identity `agentName: lsa-rocm-gfx1151` (pool Default, workFolder `_work`); `.credentials*` authenticate that same identity. Two installs with one identity running concurrently = listener conflict/erratic job leasing. Controls:
- The old `~/actions-runner-lsa` stays **offline permanently** (it is already offline; nothing restarts it).
- After post-migration validation the owner **retires** it (`mv ~/actions-runner-lsa ~/actions-runner-lsa.retired-pre-migration` — in the script's handoff), making accidental double-listen impossible.
- Rollback rule (in `rollback-runner.sh`): legacy runner and migrated copy are **mutually exclusive** — stop one before starting the other.
- Credential file modes tightened in the copy: original `.credentials` was group-readable (`0664`); the migration `chmod 600` all three identity files.
- Same-identity relocation is exactly what makes the **intentionally-kept queued run `42381769814`** (A2) execute on the migrated runner with its label set `[self-hosted, linux, rocm, gfx1151]` intact.

### Q4 — Are runner workspaces, cache, service accounts and ROCm temp files writable?

Mapped to the corrected unit: workspace `_work` (repo checkout) and sandbox `$HOME` under `/opt/actions-runner-lsa` (`ReadWritePaths`, `chown lsa-runner`); Triton cache at the explicit dir under that HOME; venv tree `/opt/lsa-rocm` (`ReadWritePaths` — cache/`.pyc` writes OK); `/tmp` is `PrivateTmp=true` (private writable tmpfs per service — ROCm compilation scratch covered); service account = dedicated system user, no sudo group, `nologin`. Nothing under `/home` or `/etc` needs writing. No pip installs happen at runtime.

### Q5 — Do the two sibling runners remain completely untouched?

**Yes, verified live — and one R1 factual correction.** Siblings `~/actions-runner` (Qwen3-TTS-ROCm) and `~/actions-runner-ovamd` (OpenVINO-Notebooks-on-AMD) are running (PIDs 336554 / 346183) — under **`user@1000.service`, i.e. amd's systemd *user* units, not system units as R1 §1.1 stated**. The migration is system-scoped (`github-runner-lsa.service`, dedicated user, `/opt` trees), reads only from `~/actions-runner-lsa` (a third, distinct directory) and touches no sibling state, unit, or credential. Post-migration validation (script handoff step) re-checks sibling PIDs/listeners. Cross-contamination vector closed: `ProtectHome=true` makes all of amd's home (credentials, sibling installs) invisible to the new service.

## 2. Migration/rollback ledger

- `ops/migrate-runner.sh` — owner-executed `sudo` runbook: preflight aborts (incl. **partial-state guard**: fresh run aborts if user or `/opt` state already exists; `--resume` re-runs the idempotent steps 2–7 from wherever an attempt aborted) → useradd → rsync copies (originals intact) → stale-state strip → **validated venv-repair recipe** → ownership/modes → corrected unit install/verify/enable → **isolation + GPU proof executed in the service security context** (a transient `systemd-run` service transcribing the installed unit's sandbox via `systemctl show`, with a parity guard on the unit file; NOT `sudo -u`, which sees all of `/home` and every device the unit filters) → printed handoff (merge workflow-fix PR, fresh-push-run validation, old-install retirement, credential rotation, runner-group scoping).
- `ops/rollback-runner.sh` — exact reversal at every step; originals were never modified; explicit mutual-exclusion rule against double listeners.
- Contract tests `tests/test_r11_ops_contracts.py` pin the invariants (workflow↔script venv path consistency, partial-state guard, service-context validation, fresh-push-run validation vehicle) on every CI run.
- Disk: needs ~12 GiB in `/opt` (9.1 GiB venv copy + runner); 76 GiB free at freeze (verified; re-checked by script).
- GPU runner stays **offline** until the owner executes the migration — unchanged policy.

## 3. Post-migration validation checklist (owner-gated, then automated where possible)

| Check | Method | Status |
|---|---|---|
| Dedicated service identity | `systemctl status github-runner-lsa` shows `User=lsa-runner` | PENDING OWNER |
| Credential isolation **in the service context** | script §7a: inside the `systemd-run` sandbox (unit's ProtectHome/ProtectSystem/DevicePolicy transcribed from the live unit) `/home` is entirely invisible — `test ! -e /home/amd` | PENDING OWNER |
| Device allowlist **actually filters** | script §7b: `/dev/dri` inside the sandbox ⊆ the unit's `DeviceAllow` list (a non-allowed node appearing = hard abort; also catches host card-renumbering after reboot) | PENDING OWNER |
| ROCm GPU visibility + Triton JIT compile **in the service context** | script §7c: real `@triton.jit` kernel compiled and executed from the relocated venv inside the sandbox (cache written to the sandbox HOME, §7d writability asserted) | PENDING OWNER |
| Runner registration + unique listener identity | script §7e: service active, exactly 1 `Runner.Listener` under `lsa-runner`; gh-side: `gh api repos/AIwork4me/latent-sparse-attention-rocm/actions/runners` → single `lsa-rocm-gfx1151` online | PENDING OWNER |
| Full 17/17 gates + suite via GitHub (the real workflow) | **fresh push run fired by merging PR #6** (workflow `PY="/opt/lsa-rocm/.venv/bin/python"`); approve in `gpu-regression` env. Supersedes cancelled pre-migration run `42381769814` (see ACTIONS_QUEUE_AUDIT.md addendum) | PENDING OWNER |
| Rollback rehearsed | `ops/rollback-runner.sh` reviewed; originals intact by construction | READY |
| Sibling runners unaffected | sibling PIDs still alive; their next jobs succeed | PENDING OWNER |
| Old credential rotation | `gh auth logout` on `amd` after dependency check | PENDING OWNER (separate approval) |

**Verdict table:** nonprivileged validation **PASS** (all five questions answered with evidence); overall migration status **OWNER_ACTION_REQUIRED**.
