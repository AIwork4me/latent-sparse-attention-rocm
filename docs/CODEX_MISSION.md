# Codex Mission — LSA on ROCm gfx1151: Full Overnight Validation and Long-term Port

You are the lead ROCm AI Infrastructure Engineer and independent release auditor. Operate on an AMD Ryzen AI MAX+ PRO 395 Linux host with gfx1151 GPU. Use ROCm 10.1.0 and the following exact Python packages in a fresh, isolated Python 3.12 venv:

```
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx1151]==2.14.0+rocm10.1.0' \
  'torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0' \
  'torchaudio==2.11.0.3+rocm10.1.0'
```

Upstream: https://github.com/Prefrontal-AI/latent-sparse-attention , independently cloned and pinned at a specific SHA. The target repository is `AIwork4me/latent-sparse-attention-rocm` (do not assume it exists; create repo only if allowed by available tooling). Respect Apache-2.0 provenance and make no upstream history rewrites. Never run the upstream CUDA-pinned lock install into this venv. Never fake `HSA_OVERRIDE_GFX_VERSION`, CUDA availability, results, exact parity, or a working SGLang path.

## Outcomes

1. A fully reproducible read-only environment inventory: driver, Python, torch build, ROCm 10.1, gfx1151 arch, firmware/kernel, memory, installed packages, flags and upstream SHA.
2. Implement a true ROCm LSA kernel port, starting with the upstream reference and HIP-compatible Triton, not a dense fallback advertised as sparse. Review and port bucket clustering, scan/select, expansion, hierarchy, sparse read and cached decode. CuTe DSL may be marked UNSUPPORTED with a genuine Triton alternative; no NVIDIA-only binaries in the ROCm path. Confirm compute, correctness and all index stage behaviors.
3. Run accuracy gates: full-selection equivalent to dense; causal windows/sinks; top-k precision and overlap; near ties/repeated text; multiple heads and GQA; per-request index reuse/teardown; prefill vs sequential decode; small and large chunk partition invariance; 3 seeds; FP16, BF16, FP32 where supported. Log differences and keep failed inputs. Correctness is release-blocking.
4. Run matched timing baselines on one GPU: PyTorch dense SDPA, upstream Block Sparse if actually available, LSA Flat, LSA Tree, with the same dtype/geometry. Context matrix: 512,1k,2k,4k,8k,16k,32k,64k,128k,256k; beyond that only when runtime/memory allows. Dims 64/128, 4/8/16 query heads, top-k 512/1024/2048/4096, sink 4/16, windows 512/2048, bucket 16/64, prefill chunks 1/128/512/2048. Combine a dense representative covering array with a smaller full factorial grid. Never allocate an unbounded cartesian product; record tested and untested combinations.
5. Strict stage breakdown: bucket build, index update, centroid projection, search, expand/topk, gather, attention kernel, total per attention layer, prefill TTFT, decode per token, end-to-end TTFT/TPOT and memory. Include kernel compilation separately; sync GPU for accurate latencies and log warm/cold cache. A claimed speedup must use `baseline_ms / actual_lsa_ms` on the same GPU, dtype, context and workload. At least 5 warmups/20 timed runs for primary cases and repeats across 3 seeds; generate median, p95, MAD and confidence intervals if useful.
6. End-to-end validation only after real ROCm SGLang integration and explicit `armed` log. Do not use NVIDIA fork's CUDA image; do not serve with disabled LSA and call it success. Use a realistically fitting, supported model and train/derive a model-specific D-LSA map. Public Qwen/GLM maps are only valid for their exact frozen checkpoints. Run RULER (13 tasks) and relevant long-context suites only if model, data, tokenization and checker are truly available; label anything else BLOCKED.
7. Overnight reliability: 10-hour hard cap, phase timeouts, bounded GPU memory, independent processes, log flushing, resumable stage ledger, panic/thermal checks, data/hardware protection. If correctness fails, continue collecting independent information but block all invalid speedup claims. Never assume 10M context fits a single UMA APU.
8. Evidence: append-only structured JSONL + table CSV/Parquet, source SHA and artifact SHA256, raw stdout/stderr, hardware telemetry, compiler versions, bounded OOM logs, repo tree, reproducible commands and concise `FINAL_REPORT.md`. Separate PASS/FAIL/BLOCKED/TIMEOUT/OOM with a matrix for attempted vs actually validated.
9. Long-term GitHub tracking: daily read-only upstream HEAD/release watcher; deduplicated GitHub issue for new SHA; classify algorithm changes, kernel changes, bug fixes, benchmark comparability, AMD portability. No automatic adoption of a new upstream SHA. Use separate review PRs, baseline SHA updates only after tests pass.
10. Package release criteria: fresh Python 3.12 venv install on gfx1151, unit/integration tests pass, no exact parity faked, independent re-derivation of speedups and precise attribution; README getting-started in English plus concise Chinese summary, `LICENSE`, `NOTICE`, contribution guide, security policy, GitHub workflows, changelog and versioned benchmark manifest.

## Execution strategy

Read the README, EVALUATION_PLAN.md, PORT_CONTRACT.md, source and tests before editing. First run and harden the existing starter suite; fix real code bugs. Create small sequential PR-sized commits and keep stage status up to date. Start a quick 30-minute pilot, then launch an unattended 10-hour plan. On errors record exact messages and use smallest repro; don't repeatedly reinstall or rewrite entire dependency tree. Where upstream CUDA code is blocked, isolate issue and create a faithful small Triton/HIP port. Use P0-P5 milestones: P0 hardware; P1 reference correctness; P2 HIP index kernels; P3 fair layer benchmarks; P4 serving+quality; P5 release evidence. A milestone is PASS only when actually measured, never based on code inspection.

## Report format

Provide: exact target GPU & memory availability, ROCm/PyTorch and upstream SHA, observed PASS/FAIL/BLOCKED table, kernel speedup p50/p95 matched vs baselines, TTFT/TPOT if integrated, length/precision quality tradeoff, thermal/memory behavior, scalability limit, opened GitHub PR/issue links if created, complete checksum pointers, top five fix priorities and concrete next run. Explicitly say `NO VALIDATED LSA SPEEDUP YET` if true.

Do not claim to have performed a full GPU overnight run unless it actually ran. Do not make CUDA paper values the Radeon result. Do not publish a product-ready claim if the server is not integrated. Maintain an honest release gate.
