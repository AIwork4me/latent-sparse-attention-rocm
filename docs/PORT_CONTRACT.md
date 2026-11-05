# ROCm port and evidence contract

A legitimate LSA speedup claim requires a genuine implementation of (A) index-key and index-query projection; (B) bucket build/maintain; (C) flat/tree centroid search and top-p/top-k; (D) sink + window + selected long-range KV read; (E) prefill/decode kernels with synchronized GPU timings. A precomputed-oracle selected-index benchmark can be reported *only* as a gather/read microbenchmark, not as LSA overall. Upstream `topk_attention` is a quadratic reference implementation, never a performance claim.

Adapt upstream components in incremental PRs, preserving provenance and Apache-2.0 notices. DO NOT commit modified upstream code unlicensed. Prefer upstream submodule pinned SHA + small HIP/Triton abstraction.

`LSA_ROCM_KERNEL_CMD` and `LSA_ROCM_E2E_CMD` are future integration hooks. The starter runner treats absent hooks as BLOCKED. An adapter command must exit 0 only on correctness+benchmark PASS and must emit machine-readable raw run tables in `results/`. No dummy smoke wrappers may return PASS for an unimplemented algorithm. Put a versioned `metrics.schema.json` in the production implementation and validate it.

Kernel metrics: p50 p90 p95, sample count, min, max, MAD, HBM/unified-memory usage, compilation time separated from steady state, context length, query chunk, heads, head dim, top-k, sink, window, bucket size, dtype, kernel backend, exact upstream SHA, machine and GPU. Do not advertise CUDA paper speedups as Radeon results.

E2E metrics: TTFT, TPOT, ITL distribution, generated token/s, context tokens, output tokens, concurrency, batch sizes, accuracy score, memory peaks, CPU and GPU utilization, setup cost and prefill chunk length. Same checkpoint and prompts for paired baseline/LSA runs. Full-precision/dense vs sparse across different precision is invalid.

Correctness: full coverage k>=T must reproduce dense reference; causal and mask behavior, top-k near ties, repeated keys, empty buckets, boundary positions, multi-query/GQA, incremental decode vs whole prefill, cache teardown and reuse, CPU/Triton parity and deterministic seed. Distinguish numerical tolerance from exact top-k set equality; publish intersection/recall and task-level accuracy, never suppress mismatches.

On errors, classify OOM / UNSUPPORTED / TIMEOUT / REGRESSION / INTERNAL_ERROR / INFRA_BLOCKED separately. Do not extend a single unsuccessful run into a claimed successful long context. Preserve failed shapes and logs.
