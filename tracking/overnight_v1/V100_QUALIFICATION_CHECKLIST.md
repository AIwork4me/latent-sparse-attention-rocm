# v1.0.0 qualification checklist (overnight_v1 — advisory packet, NO publication)

Verdict driver: every NO-GO item blocks v1.0.0. Current overall verdict: **NO-GO** (owner
gates + model/serving layer incomplete). This packet is the honest gap map.

| # | Mandatory criterion (mission M7) | Status | Evidence |
|---|---|---|---|
| 1 | Merged-tree + physical gfx1151 correctness | **MET** | suite 314/314 GPU; 17/17 gates × {default,batched,fused} (M1 receipts, commit bf36917) |
| 2 | Precise evidence hashes | **MET** | SHA256SUMS 28/28 verified; aggregate independently recomputed 60/60 cells (m1-agg-recompute) |
| 3 | Long-context envelope honestly measured | **MET** (as bounds) | M2 ladder d0bf6477: 64k–256k decode byte-exact; ≥192k prefill OOM-classified; wide-path fused regression documented |
| 4 | Genuinely implemented model integration | **PARTIAL** | M3: TinyLlama-1.1B forward runs the repo's select+read via official custom-attention interface; select-everything == dense (cos 0.9999973); Qwen3 zero-training path blocked by Ward index_dim ≤ 256 (needs trained map — upstream's design) |
| 5 | Pre-registered quality gates met | **NOT MET** | M3 ppl/NIAH numbers exist but sparse-arm degradation on an untrained full-rope index is expected & measured; the ≤2pp release gate is NOT satisfiable without trained index projections → v1.0 scope must stay stage-level or gate stays open |
| 6 | Honest E2E serving numbers | **NOT MET** | M5 not run; M4 SGLang BLOCKED_WITH_ROOT_CAUSE (no approved container lane) |
| 7 | SGLang status without false implication | **MET** (as BLOCKED) | m4-sglang-matrix registry entry; docs must keep the BLOCKED wording |
| 8 | Source/provenance pins | **MET** | upstream 5ebea732 pinned; all receipts pin code SHA + dirty flag |
| 9 | No security regressions | **MET** | check_workflow_security.py PASS on merged tree |
| 10 | Migration/CI posture | **OWNER-GATED** | run 42483727525 pending ≠ PASS; runner migration OWNER_ACTION_REQUIRED — untouched |
| 11 | Version compatibility & README/runbook | **PARTIAL** | HOW_TO_REPRODUCE_V1.md delivered tonight; release notes RC exist; README quickstart pending owner review |
| 12 | Recovery & rollback | **MET** | PR#8 3-layer rollback doc; overnight ledger RESUME.md verified structure |

## Minimum path from NO-GO to GO (estimates, not promises)

1. Owner: migrate runner, approve post-merge GPU run (turns #10 green).
2. Owner: authorize v0.4.0 tag/Release (scope: stage-level select fusion ≤32k).
3. Engineering: trained index projection (upstream distillation route) for any
   model-level v-next claim; or freeze v1.0 scope to stage-level kernels explicitly.
4. Owner: decide SGLang lane (pre-approved gfx1151 image) before any v0.6 claim.
