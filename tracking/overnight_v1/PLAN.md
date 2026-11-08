# Overnight V1 plan (derived from CODEX_GFX1151_OVERNIGHT_V1_MISSION.md)

Executor: ZCode on-host session (owner replaced codex exec 2026-11-08).
Deadline: host-local 2026-11-09 07:00 or 10h cap, whichever first. No new GPU workload within ~15 min of deadline.

## Invariants (absolute)

- Only this worktree `/home/amd/lsa-v1-overnight`, branch `exp/gfx1151-v1-night-20261108-2155`. Never main. No force-push, no tags, no releases, no protected-settings changes, no rulesets, no token rotation, no environment approvals, no runner migration, no sudo/apt, no host ROCm/driver changes.
- PR #8 stays DRAFT. Default `LSA_SELECT_IMPL=reference` unchanged; benchmark `fused` via env var only.
- GPU work only when gfx1151 visible AND idle (check kfd pids before each heavy step). One heavy campaign at a time. Never `pkill -f`; kill by PID.
- OOM => classify + `BLOCKED_RESOURCE`/`OOM`, never a PASS. Reserve >=6 GiB RAM for OS.
- Every experiment: preregistered hypothesis in EXPERIMENT_REGISTRY.jsonl BEFORE the run; append-only receipts after; no post-hoc cherry-picking. Single-run confirmations are labeled as such, never as the 3-seed campaign.

## Sequence

1. **P0** initial proof + ledger (this commit). Gate commands verified runnable.
2. **M1** merged-tree GPU qualification (highest priority)
   - full GPU pytest `tests/` (expect 258 passed on gfx1151; record skips)
   - 17-gate manifest x {default reference, LSA_TREE_IMPL=batched, LSA_SELECT_IMPL=fused}
   - evidence byte verification (done at P0: 28/28) + raw-sample aggregate cross-check of `aggregate-fused.json`
   - bounded matched A/B confirmation 16k+32k bf16, single seed single process, warmup 5 / reps 20, byte-exact pre+post checks - explicitly labeled single-run confirmation vs the 3x3 campaign
   - GO/NO-GO memo for v0.4.0 release (memo only; do not publish)
3. **M2** 64k->256k fail-closed scaling
   - parameterized incremental harness off `scripts/ab_select_fusion.py` pattern; decode + sampled-select at 64k FIRST, then 96/128/192/256k only as measured memory permits; prefill >=64k is a separate optional high-memory lane, never blind O(N^2) preallocation
   - oracle comparisons per shape; registry entries with peak alloc/reserved, MemAvailable, times; BLOCKED_RESOURCE/OOM statuses allowed
4. **M3** true model-level quality (v0.5): inventory cached models first; BLOCKED_MODEL_ASSETS if none; 3-arm contract (dense / LSA reference / LSA fused); pre-registered quality gates before any run.
5. **M4** SGLang read-only compatibility matrix; BLOCKED_WITH_ROOT_CAUSE if no approved container lane; never pull multi-GB images.
6. **M5-M6** only on real M3/M4 results.
7. **M7** v1.0.0 qualification packet - checklist only, NO publication.
8. **Closeout**: OVERNIGHT_REPORT.md (Chinese), RESUME.md with exact next command, STOP.json only if genuinely no safe work remains.

## Time budget (rough, from ~22:00 +08:00)

- M1 gates+pytest+A/B: ~2-2.5h
- M2 harness + ladder: ~2.5-3h
- M3 inventory + (if assets) pilot: ~1-1.5h
- M4 matrix: ~0.5h
- Report + ledger hygiene: ~0.5h
Rebalance dynamically; ledger is authoritative.
