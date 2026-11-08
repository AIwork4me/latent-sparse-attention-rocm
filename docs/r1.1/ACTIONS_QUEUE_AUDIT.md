# A2 — Actions Queue Audit (stale GPU runs dispositioned)

**Executed:** 2026-11-08 10:05–10:07 +0800 (02:05–02:07 UTC).

## Runs inspected (pre-disposition state, recorded before any action)

| Run | Workflow / trigger | HEAD | Status before | Jobs | Disposition |
|---|---|---|---|---|---|
| `42381678702` | `rocm-regression.yml` (GPU regression, trusted gfx1151) · `push` on `main` | `9a8b87a` (docs-only commit, superseded) | `waiting` | 1 job `gates-and-kernels` (id 128430432857) `waiting` on offline runner, labels `[self-hosted, linux, rocm, gfx1151]`; created 2026-11-07T14:27:28Z, never started | **CANCELLED** |
| `42381769814` | `rocm-regression.yml` · `push` on `main` | `1ab0e0b` (**current main HEAD** — verified unchanged through 2026-11-08) | `pending` (environment approval), advanced to `waiting` after the other run's cancellation freed the queue | 1 job `gates-and-kernels` (id 128597911989) `waiting` on offline runner, labels `[self-hosted, linux, rocm, gfx1151]` | **INTENTIONALLY KEPT** (see below) |

No other `in_progress`/`waiting`/`queued` runs existed in the repository at disposition time (checked `gh run list --status {waiting,queued,in_progress}`; `pending` is not a list filter — the pending run was confirmed via the runs API). Both runs are repository-owned (`push` to `main` by the owner's release process); no other project's jobs were touched.

## Cancellation rationale — `42381678702`

1. **Non-running:** zero jobs had started; the sole job was waiting for an offline runner since 2026-11-07T14:27Z.
2. **Demonstrably superseded:** same workflow, same branch (`main`), older HEAD (`9a8b87a` — a docs-only commit) than the surviving run at `1ab0e0b`.
3. **Obsolescence risk if left queued:** it would execute the full GPU gate suite on a stale docs-only HEAD the moment the runner returns — wasted GPU-hours and a confusing "validation" of a tree nobody ships.
4. Its cancellation was verified: final state `completed/cancelled` (checked via `gh run view` post-cancel).

## Intentional remaining run — `42381769814` (explicit explanation)

This is **not** obsolete: it targets the current, verified main HEAD `1ab0e0b`, and nothing newer exists on `main`. It is deliberately kept queued as the **post-migration validation vehicle for A3**: when the owner-executed runner migration brings the isolated `lsa-runner` online (keeping the label set `[self-hosted, linux, rocm, gfx1151]`), this run will execute the full GPU gates + kernels on `main` through GitHub Actions — precisely the "successful owner-approved GitHub GPU workflow" check the migration validation matrix requires. It cannot run earlier because no matching runner is online. The environment-approval protection on `gpu-regression` (required reviewers) remains in place as a second gate.

If the owner instead prefers a fresh post-migration run, this one can be cancelled from the Actions UI without side effects; the migration validation (A3) then triggers a new push-based run.

## Final state (verified 2026-11-08T02:07Z)

- `42381678702` → `completed / cancelled` ✓
- `42381769814` → `waiting` (offline runner; job never started) — documented above ✓
- No other queued GPU jobs exist; the persistent GPU runner remains **offline** pending its migration and isolation checks (unchanged, per policy).

**Acceptance: MET** — no obsolete queued GPU job remains capable of unexpectedly executing when the runner comes online; the single intentional remaining run is explicitly explained and non-obsolete.
