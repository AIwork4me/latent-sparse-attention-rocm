# Open blockers (overnight_v1)

## Owner-action blockers (inherited, unchanged - NOT worked around tonight)

- RUNNER-MIGRATION: sudo runbook `docs/r1.1/ops/migrate-runner.sh`; out of scope here.
- GPU-POST-MERGE-CI: run 42483727525 `pending` = NOT RUN; no approval attempted; no second listener.
- RULESETS-AND-ROTATION (P1-P4 + amd credential rotation): proposal-only, untouched.
- v0.4.0 tag/Release: owner decision after RC review; tonight produces a GO/NO-GO memo only.
- PR #8 promotion: stays DRAFT.

## Campaign-local blockers

(none yet)

## Campaign-local (2026-11-08 overnight_v1)

- M4 SGLang serving: BLOCKED_WITH_ROOT_CAUSE - see EXPERIMENT_REGISTRY m4-sglang-matrix.
  Unblocks only with an owner-approved pre-pulled gfx1151 sgl-dev image (different ROCm
  container lane, ~ROCm 7.2.4) or an explicitly scoped integration effort on the port.
