# Independent matched benchmark repetitions

- repetitions: 3 independent adapter processes, same seed/shape/dtype
- upstream SHA: `5ebea732d639576d2a7579b0de40e6252f9001f5` · repo HEAD: `82c891e2ea12700205fcdb56df343b9b3b6cb1c8`
- device: AMD Radeon 8060S Graphics (gfx1151) · torch 2.14.0+rocm10.1.0
- gates: 16/16 PASS in every repetition
- run ids: rep1-20261105T225807Z, rep2-20261105T225807Z, rep3-20261105T225807Z

| shape (ctx/dtype/phase/method) | per-run speedups | median | min-max | candidate p50 median (ms) |
|---|---|---:|---|---:|
| 8192/bf16/decode/block_sparse | 1.10x / 1.07x / 1.01x | **1.07x** | 1.01-1.10 | 0.299 |
| 8192/bf16/decode/lsa_flat | 1.02x / 0.96x / 1.04x | **1.02x** | 0.96-1.04 | 0.311 |
| 8192/bf16/decode/lsa_tree | 0.29x / 0.27x / 0.29x | **0.29x** | 0.27-0.29 | 1.100 |
| 16384/bf16/decode/block_sparse | 1.81x / 1.68x / 1.99x | **1.81x** | 1.68-1.99 | 0.331 |
| 16384/bf16/decode/lsa_flat | 0.89x / 0.84x / 0.89x | **0.89x** | 0.84-0.89 | 0.679 |
| 16384/bf16/decode/lsa_tree | 0.45x / 0.47x / 0.46x | **0.46x** | 0.45-0.47 | 1.317 |
| 32768/bf16/decode/block_sparse | 3.93x / 3.57x / 4.17x | **3.93x** | 3.57-4.17 | 0.300 |
| 32768/bf16/decode/lsa_flat | 2.41x / 1.52x / 2.11x | **2.11x** | 1.52-2.41 | 0.584 |
| 32768/bf16/decode/lsa_tree | 0.89x / 0.74x / 0.92x | **0.89x** | 0.74-0.92 | 1.336 |

Per-repetition receipts: `receipt-1.json`, `receipt-2.json`, `receipt-3.json`
