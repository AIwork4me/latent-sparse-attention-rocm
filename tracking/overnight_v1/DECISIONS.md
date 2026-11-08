# Overnight V1 decisions (append-only)

## D001 - 2026-11-08 ~21:50 +08:00 - Executor substitution
Owner instructed: run the mission on this host with ZCode instead of `codex exec`.
Consequences: no codex daemon, no tmux (not installed; apt forbidden). ZCode session drives
rounds directly; this ledger replaces inter-round state transfer. Deadline still host-local
07:00 / 10h cap. All mission boundaries unchanged.

## D002 - 2026-11-08 ~21:52 - git transport fallback
`git fetch` HTTPS failed repeatedly (GnuTLS -110 / empty reply). Working fallback:
`git -c http.version=HTTP/1.1 fetch origin main` (succeeded first try). SSH has no key.
Use the HTTP/1.1 variant for any future fetch/push from this host.

## D003 - 2026-11-08 ~22:05 - Evidence manifest path convention
`evidence/v0.4.0/SHA256SUMS` paths are relative to `evidence/v0.4.0/` (manifest header says so).
Verify with `cd evidence/v0.4.0 && sha256sum -c SHA256SUMS`. Initial mis-run from `evidence/`
showed spurious FAILED-open lines; corrected run: 28/28 OK.

## D004 - 2026-11-08 ~22:05 - GPU exclusivity basis
`fuser /dev/kfd` empty at campaign start; GitHub GPU run 42483727525 remains `pending`
(no runner online to pick it up). Local physical GPU tests authorized per mission P0.3.
Recheck kfd pids before every heavy step.
