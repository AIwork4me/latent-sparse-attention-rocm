# ROCm port notes — what was found, what was fixed, what is blocked

Engineering record of porting LSA to gfx1151 / ROCm 10.1 / Triton 3.8.0-rocm.
Every item below is backed by a log under `results/` or a reproducible command.

## 1. Upstream architecture (as audited at SHA 5ebea732)

- `lsa.core` is the pure-torch **oracle**: `index_space.regroup` (GQA-collapsed,
  position-free index space) → `ward.ward_rounds` (bottom-up spherical Ward merge)
  → `reference.build_buckets` / `read` (flat) → `hierarchy.build_hierarchy` /
  `read_hierarchy` (tree) → `attention.topk_attention` (quadratic sparse read).
- `lsa.core.kernels.triton.*` implements **every index step in Triton**
  (Ward merge, node-bucket pairing, top-k listing, scan-select, expansion, beam,
  paged decode). `kernels/cutedsl/*` (NVIDIA CuTe DSL) accelerates them on
  H200/B300 only.
- The sparse *read* has no Triton kernel upstream: serving delegates it to
  FA3 / FlashMLA / TileLang backends (NVIDIA-only).
- The kernel front doors (`kernels/{ward,expansion,scan_select}.py`) and the whole
  `lsa.sglang.index` chain import `cutlass` at module level → **import failure on
  any NVIDIA-free environment**. The port therefore calls the Triton modules
  directly.

## 2. Environment findings

| Finding | Evidence |
|---|---|
| ROCm 10.1 userspace installs cleanly from AMD pip wheels next to a host ROCm 7.2.1 (no host modification) | `results/pilot/preflight.json` |
| torch reports 100 GiB GPU memory on this UMA APU; the dedicated VRAM window is 32 GiB — benchmarks stay well inside it | preflight `total_memory_bytes` vs rocm-smi VRAM |
| `torch.mm(bf16, bf16, out_dtype=fp32)` works on ROCm (production scoring path preserved) | gates B1/B2 |

## 3. Kernel-level findings on gfx1151

1. **`tl.dot` input precisions**: HIP allows only `ieee`, `bf16x3`, `bf16x6`.
   Upstream's default `tf32x3` (and `tf32`) fail Triton compilation on ROCm.
   The port pins `precision="ieee"` everywhere (`lsa_rocm_lab.port.HIP_PRECISION`).
   With `ieee`, upstream's own Ward tests assert **exact label equality** vs the
   torch oracle and pass on gfx1151. Upstream test files exercising tf32 paths
   fail on this platform by construction and are documented as such.
2. **Ward kernel shape floors** (both upstream-enforced): pools per block must be
   a power of two in [16, 4096] and `index_dim` a power of two in [16, 256];
   node pairing needs ≥ 32 buckets per group per halving. Consequence for
   benchmark geometry: `Hkv × Dh ∈ {64, 128, 256}` (e.g. 8q/2kv × 64d).
   The GLM 512-dim MLA index space is beyond the Triton Ward kernel — upstream
   uses CuTe DSL there (NVIDIA-only) — and is out of scope for this port.
3. **Triton-HIP pipelining miscompile (found by this port)**: a masked gather loop
   with the default software pipelining caused nondeterministic illegal memory
   accesses at low occupancy (the 8-program decode launch). `tl.range(...,
   num_stages=1)` is stable (200-iteration stress, both phases). Recorded for
   upstream Triton; the port's read kernel carries the fix and the note.
4. **`libdevice` portability**: upstream's `ffs` / `popc` / `rint` usage maps to
   ROCm libdevice without changes.

## 4. Upstream behaviors a porter must know (not bugs in the port)

- The torch oracle (`lsa.core.reference/ward/hierarchy`) is CPU-bound by
  construction (device-less `torch.full` / `torch.empty` internals) and its
  `topk_attention.keep_set` scatter races when a PAD entry clamps onto a live
  id 0 (every row's window contains id 0). The gates feed it causally-neutral
  padding (the row's own position) and document the workaround.
- Pairing to `node_bucket_size > 2` is **iterated halvings** of the pairing kernel
  (one call = one halving); treating a single call's labels as size-4 groups is
  wrong. Gate A2 catches exactly this (and did).
- The fused scan-select path starts at 32,768 visible buckets — far above this
  campaign's scale; the production short/medium-context composition (dense bf16
  GEMM + Triton top-k + rank-order expansion) is what the port benchmarks.

## 5. Correctness methodology

- **Same-device exactness** where upstream's own tests demand it (Ward pairing,
  labels): oracle runs on the GPU via `ward_rounds` (upstream's tests do the same).
- **Cross-device (CPU oracle / GPU port) set-equality** for composed selection:
  B1 (fp32 scoring) exact per row; B2 (bf16 production scoring) is a measurement
  gate (mean recall ≥ 0.999; measured 0.99991, worst row published).
- **Numeric tolerance** for the read kernel vs the quadratic oracle and dense SDPA
  (fp32 4.5e-8, fp16 1.5e-5, bf16 1.2e-4 max-abs at gate scale).

## 6. SGLang end-to-end on gfx1151: BLOCKED

1. SGLang's AMD support targets Instinct MI300X/MI325X/MI350X/MI355X with
   per-ROCm-release Docker builds; no ROCm wheels on PyPI, and gfx1151 (RDNA
   APU) is not in the support matrix.
2. The upstream plugin arms FA3 / FlashMLA / TileLang backends (NVIDIA-only) for
   the GLM path; its import chain pulls `cutlass` at module level.
3. Target checkpoints (GLM-5.3-Flash fp8 MoE, Qwen3.8-27B) exceed the 32 GiB
   dedicated window of this APU or lack a ROCm sglang build of their attention
   backends.
4. D-LSA distilled projection maps are checkpoint-bound artifacts; deriving them
   for a new checkpoint is a training campaign of its own.

A future E2E path that does not fake arming: SGLang's triton attention backend
(the Qwen plugin arm) on a ROCm-supported checkpoint that fits, with the plugin's
"armed" banner captured in the server log, per `docs/PORT_CONTRACT.md`.

## 7. Excluded upstream tests, with reasons

- `test_scan_select.py`, `test_bucket_scan.py`, `test_dispatch.py`,
  `test_expansion.py` — import CuTe DSL front doors (`cutlass`) at module level.
  The expansion kernels are covered on this GPU by the port's gates B1/B2/C,
  which drive `expand_in_order` against the torch oracle.
- `test_ward_kernel.py` tf32/tf32x3 cases and `test_shape_kernels.py`
  power-iteration tolerances — NVIDIA precisions / RDNA numerical divergence
  (marginal, 1.6% of elements at 8.5e-5 on the shape path, which the port does
  not use: unshaped buckets only).

## 8. is_causal alignment on non-square Q/KV (found in the v0.2.1 audit)

`F.scaled_dot_product_attention(is_causal=True)` builds a **top-left**-aligned
lower-triangular mask: row `i` sees column `j <= i`. For `q_len == 1` against a
full cache that masks the single decode row to **key 0 only** — wrong output
(max abs err 2.44 vs the prefill's last row at 4k bf16) *and* a different (here,
~2.8× slower at 32k) dispatch than the correct full-attention path. Non-square
decode shapes need bottom-right alignment: key `j` is visible to chunk row `i`
iff `j <= (kv_len - q_len) + i`. `pipeline.dense_read` now encodes exactly that
(square → fused causal; 1 row → unmasked; chunk → explicit mask), and gate G +
`tests/test_dense_read.py` pin it. Rule for any future baseline: never pass
`is_causal=True` when `q_len != kv_len`.
