# Benchmark plan: 10-hour night, single gfx1151

0. Freeze upstream SHA, kernel version, pip freeze, driver, OS, GPU mem; check root free disk and original system environment; no upgrades to host.
1. Reference: PyTorch correctness + upstream unit test subset on CPU and ROCm for all-select parity; FP32/FP16/BF16; cases of masks, selected top-k and causal decode; index unit tests and deterministic fixtures. Run pytest subsets only if imports work without NVIDIA dependencies.
2. HIP/Triton: compiler smoke, port independent kernel stages (Ward clustering, bucket scan, candidate scoring, top-k, expansion, beam, decode), compare to upstream CPU reference; numerical + selection-set gates.
3. Synthetic kernel performance: 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k *as feasible*. Query block 1/128/512/2048. Dtype FP16/BF16; 4/8/16 heads, D64/128 as supported; top-k 512/1024/2048/4096; window 512/2048; sink 4/16; buckets 16/64; flat vs tree vs dense vs block sparse, but only where actually implemented and apples-to-apples. Process highest contexts in separate subprocesses with OOM classification.
4. Genuine LSA prefill/decode: separate index build, select, attention gather/read, total end-to-end layer; include the index maintenance and memory. Warm 5 rounds, measure >=20 repetitions at fixed shape where stable, publish median and p95; include CI/paired bootstrap or spread. Fresh process for each context where needed. Bound time, allocate max ~70% torch-reported free GPU memory per stage, don't treat UMA as equivalent to 128GB dedicated VRAM.
5. Accuracy: pinned small text model if available (4k/8k/16k/32k), then RULER 13 tasks, LongBench variants, Needle, multi-hop, code-agent trace; pinned tokenizer, data license, seed, decoding policy and checksum. Model-specific D-LSA map cannot be reused across checkpoints. No E2E claim unless actual SGLang ROCm plugin loads and logs armed.
6. E2E (conditional): use ROCm-supported SGLang + genuine plugin; 1/2/4 concurrent clients, 128/512 output tokens, 4k-128k prefills where feasible; record TTFT/TPOT/ITL/throughput/memory; quality gates and restart between baseline and candidate.
7. Reliability: 3 repeats of strategic shapes, GPU telemetry and thermal state, 15m steady state; replay previously failed shapes; compare overnight drift.
8. Report: objective table with PASS/FAIL/BLOCKED, matrices coverage (attempted/passed/skipped), formula speedup baseline_ms/candidate_ms, uncertainties, exact JSON logs and file SHA256; never label incomplete suite "all pass".

## Recommended wall-clock allocation

- 0:00-0:30 env/preflight/clone/manifest
- 0:30-2:00 correctness and edge cases
- 2:00-4:30 Triton/HIP kernels (with bounded attempts)
- 4:30-6:30 layer microbenchmarks
- 6:30-8:15 model accuracy and E2E if available (otherwise skip explicitly)
- 8:15-9:15 stress/consistency
- 9:15-10:00 reports/checksums/review

A stage can finish early or block; unused time can be assigned to additional validated shapes, not to fake successes. An initial starter run does not yet implement all stages. Codex should implement milestones, tests and authentic adapters before labeling COMPLETE.
