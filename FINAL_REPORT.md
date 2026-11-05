# FINAL REPORT — Latent Sparse Attention on ROCm gfx1151

**Project:** `AIwork4me/latent-sparse-attention-rocm` (working tree: `lsa-rocm-eval-starter`)
**Date:** 2026-11-05/06 (UTC) · **Author:** ROCm AI Infrastructure Engineering (agent-executed session)
**Upstream:** Prefrontal-AI/latent-sparse-attention @ `5ebea732d639576d2a7579b0de40e6252f9001f5` (pinned, read-only)

---

## 1. Executive summary

A genuine ROCm-native LSA path was implemented, validated against upstream's
pure-PyTorch oracle, and benchmarked against matched baselines on the target
Radeon GPU. **Every published number below is a real gfx1151 measurement with a
validated receipt.**

Headline results (bf16, matched geometry 8q/2kv×64d, top_k 4096 / top_p 1024 /
window 512 / sink 4, warm 5 / timed 20, p50):

| Result | Value | Classification |
|---|---|---|
| **D-LSA Flat decode speedup vs dense SDPA** | **5.04× @ 32k ctx** (3.76× @ 16k, 2.81× @ 8k, 1.21× @ 4k) | ✅ VERIFIED |
| **Block-sparse (positional) decode speedup** | **7.86× @ 32k ctx** (3.60× @ 16k) | ✅ VERIFIED |
| D-LSA Tree decode speedup | 2.18× @ 32k (unoptimized beam) | ✅ VERIFIED (with attribution) |
| Correctness gates | **13/13 PASS** (exact set-equality + numeric) | ✅ VERIFIED |
| Upstream Triton kernel tests on ROCm | **203/210 PASS**; 7 fails all NVIDIA-only-precision or unused-shape-path | ✅ VERIFIED (with documented exclusions) |
| Prefill (select+read) vs dense SDPA | **0.19× @ 32k** (slower) — honest deficit, analyzed in §6 | ⚠️ LIMITATION |
| SGLang end-to-end serving on gfx1151 | — | ⛔ BLOCKED (§7) |

**No claim in this report derives from a simulated implementation, a CUDA
fallback, an unarmed plugin, or unmatched benchmarks.** The starter's own
receipt validator (`run.validate_receipt`) accepted the kernel-stage evidence
and set `claimable_lsa_speedup: true`.

## 2. Environment (read-only inventory, nothing on the host modified)

| Item | Value |
|---|---|
| CPU / SoC | AMD Ryzen AI MAX+ PRO 395 (32C), HP ZBook Ultra |
| GPU | Radeon 8060S, **gfx1151**, 32 GiB dedicated VRAM (UMA APU; torch reports ~100 GiB visible — never treated as dedicated) |
| Kernel / OS | 6.17.0-1033-oem, Ubuntu 24.04.4 |
| Host ROCm (untouched) | 7.2.1 at `/opt/rocm` |
| Mission env | fresh `.venv` (python 3.12.3) + AMD pip wheels `rocm[libraries,devel,device-gfx1151]==10.1.0` |
| torch / HIP | `2.14.0+rocm10.1.0` / HIP 7.16.26385 — **exact required versions** |
| torchvision / torchaudio | `0.29.0a0+rocm10.1.0` / `2.11.0.3+rocm10.1.0` — exact |
| Triton | `3.8.0+git669b31ac.rocm10.1.0` (AMD build) |
| Evidence | `results/preflight-manifest` / `evidence/preflight-manifest.json` |

## 3. Stage table (both campaign profiles, this machine)

| Stage | Quick profile | Overnight profile |
|---|---|---|
| preflight | PASS | PASS |
| reference_correctness (upstream oracle == dense SDPA, fp32/16/bf16) | PASS | PASS |
| triton_hip_probe | PASS | PASS |
| dense_prefill_baseline (SDPA 512–32768) | PASS | PASS |
| **lsa_rocm_kernel** (gates + matched matrix + validated receipt) | **PASS** | **PASS** |
| lsa_rocm_end_to_end (SGLang) | BLOCKED (no adapter — correct) | BLOCKED |

Run records: `results/20261105T145658Z/` (quick), `results/20261105T145917Z/`
(overnight), each with `stages.jsonl`, per-stage logs, receipts, `SHA256SUMS`.
**Scope honesty:** both profiles ran to completion tonight on the target machine
(~40 min wall each, bounded by the same 10 h hard cap in `scripts/overnight.sh`).
This was a bounded campaign, **not** a literal 10-hour soak; §9 gives the next
run's extension plan.

## 4. What was built (the port)

- `src/lsa_rocm_lab/port/` — ROCm port layer composing **upstream's own Triton
  kernels**: Ward bucket build (`ieee` Gram precision — tf32x3 is NVIDIA-only),
  node-bucket hierarchy pairing (iterated halvings, as production does), top-k
  listing, packed-key ordering, rank-order expansion.
- `src/lsa_rocm_lab/port/read_kernel.py` — **the Triton sparse gather-attention
  read kernel**: upstream delegates the read to FA3/FlashMLA/TileLang
  (NVIDIA-only); this kernel replaces it with an online-softmax gather read
  honoring the expansion's causal-list contract and GQA. This is the project's
  original kernel contribution.
- `src/lsa_rocm_lab/port/gates.py` — the 13-gate oracle-parity suite.
- `src/lsa_rocm_lab/kernel_adapter.py` — the `LSA_ROCM_KERNEL_CMD` target
  producing receipt-validated comparisons per `schemas/metrics.schema.json`.
- Release scaffolding: README, LICENSE (Apache-2.0), NOTICE (upstream
  provenance), CONTRIBUTING, SECURITY, CHANGELOG, PORT_NOTES, ROADMAP,
  upstream-watch + ROCm-regression GitHub workflows, evidence bundle with
  SHA256SUMS.

## 5. Correctness evidence (release-blocking gates — all green)

Gates (full JSON: `evidence/port-gates.jsonl`):

| Gate | Criterion | Result |
|---|---|---|
| A bucket build | Triton Ward == torch oracle partition, exact; centroids ≤1e-5 | PASS (err 4.5e-8) |
| A2 hierarchy pairing | iterated Triton pairing == torch oracle, same device, exact | PASS |
| B1 flat select (fp32) | exact per-row key-set equality vs `reference.read` | PASS (512/512) |
| B2 flat select (bf16 production scoring) | mean recall/precision ≥ 0.999 | PASS (0.99991; worst row published, min recall 0.959 = one near-tie bucket flip) |
| C tree select | port beam == reference walk over the same tree, exact | PASS (512/512) |
| D1 read vs quadratic oracle | per-dtype allclose (fp32 4.5e-8 / fp16 1.5e-5 / bf16 1.2e-4) | PASS ×3 |
| D2 full select == dense SDPA | same tolerances, both oracles | PASS ×3 |
| E decode step == whole-prefill row | selection exact, output err 0.0 | PASS |
| F GQA (8q/2kv) | D1 on grouped shape | PASS |

Upstream's own kernel tests on this GPU (`evidence/upstream-kernel-tests.log`):
**203 passed, 7 failed** — the 7 are (a) 4 Ward tests exercising
`tf32`/`tf32x3` input precision, which HIP rejects at Triton compilation
(`allowed: ieee, bf16x3, bf16x6`) — NVIDIA-only by construction; (b) 3
shape-kernel tolerance tests with marginal RDNA numerics on the shaped-scoring
path, which this port does not use (unshaped buckets). The exact-parity tests
upstream themselves assert (`torch.equal` Triton vs torch, same device) all
pass on gfx1151.

Two genuine defects were found and fixed during validation (kept in
CHANGELOG/PORT_NOTES): a decode-phase query/position mismatch in the bench that
drove upstream's expansion kernel to out-of-bounds reads, and a Triton-HIP
default-pipelining miscompile of the read kernel's masked gather loop at low
occupancy (pinned `num_stages=1`; 200-iteration stress × both phases).

## 6. Performance — matched measurements (bf16; fp16 matrix also recorded in receipts)

Decode step p50 ms (select+read on a prebuilt index; index build timed separately):

| ctx | dense SDPA | block-sparse | **D-LSA Flat** | D-LSA Tree |
|---:|---:|---:|---:|---:|
| 512 | 0.173 | 0.367 | 0.295 | — |
| 1024 | 0.200 | 0.375 | 0.277 | — |
| 2048 | 0.239 | 0.398 | 0.282 | — |
| 4096 | 0.364 | 0.468 | **0.300 (1.21×)** | 0.972 |
| 8192 | 0.943 | 0.500 (1.89×) | **0.335 (2.81×)** | 1.124 |
| 16384 | 1.714 | 0.476 (3.60×) | **0.456 (3.76×)** | 1.264 (1.36×) |
| 32768 | 3.408 | **0.433 (7.86×)** | **0.676 (5.04×)** | 1.561 (2.18×) |

Prefill (whole-context, select+read) p50 ms: dense **82.3** vs flat 456.6
(0.18×), block-sparse 409.7, tree 533.0 at 32k — LSA prefill is *slower* at
these scales; p95s and MADs in `evidence/stage-rows.jsonl`.

Stage breakdown at 32k decode (p50 ms): flat = select 0.528 + read 0.150;
block-sparse = select 0.521 + read 0.142; tree = select 1.333 + read 0.096.
Index build (amortized, per layer): flat 6.9 ms at 32k (0.98 ms at 512);
hierarchy build 13.0 ms. Prefill's deficit decomposes as select 96.6 ms +
read 362.9 ms at 32k.

**Bottleneck analysis:**
1. The read kernel is correctness-first v1 (1 query-row/head per program,
   elementwise dot, gather loads — no tensor-core tiles). At 32k prefill it
   moves ≈ 3.7 GFLOP-equivalent gathers at ~10–20 GFLOP/s-effective; a tiled
   `tl.dot` design is the single highest-leverage fix (ROADMAP Track 1).
2. At these contexts LSA's density is high: top_k 4096 + window 512 ≈ 14% of
   32k keys — the design's wins compound at 128k+ and in decode-heavy regimes.
3. Prefill select (bf16 GEMM over 4096 buckets + top-k + expansion per query
   row) costs ~3 µs/row — the production path amortizes with page-level
   batched reads; the port composes per-row.
4. Tree select is a torch-composed beam (per-page python loop over 64 pages at
   32k): correct (gate C) but not kernelized — upstream's Triton beam kernels
   exist and are the natural swap-in.

**Quality tradeoff (selection accuracy):** fp32-scored selection is exactly the
oracle's; production bf16 scoring agrees on 99.6% of rows exactly, mean
recall/precision 0.99991 (worst row: one near-tie bucket swap). Read numerics
vs oracle: ≤1.2e-4 (bf16). Task-level accuracy (RULER/LongBench) — not run:
requires the E2E path (BLOCKED) or a model harness; classified LIMITATION.

**Thermal/memory:** GPU edge temperature peaked at 73 °C during the campaign
(telemetry log); no thermal throttling events observed; peak stage allocations
well inside the 32 GiB dedicated window (largest: 32k bf16 q/k/v + index
artifacts ≈ 2.4 GiB). No OOM rows in either profile.

## 7. BLOCKED — SGLang end-to-end on gfx1151 (with causes and the unblocking path)

1. SGLang's AMD platform targets Instinct MI300X/MI325X/MI350X/MI355X via
   per-ROCm-release Docker builds; no ROCm wheels on PyPI; gfx1151 (RDNA APU)
   is outside the support matrix (checked against sglang docs/PyPI, 2026-11-05).
2. Upstream's plugin arms FA3 / FlashMLA / TileLang backends (NVIDIA-only);
   its import chain requires `cutlass` at module level.
3. Target checkpoints (GLM-5.3-Flash fp8 MoE, Qwen3.8-27B) exceed the APU's
   32 GiB dedicated window or have no ROCm backend support.
4. D-LSA distilled projections are checkpoint-bound; deriving new maps is a
   training campaign (ROADMAP Track 3.8), never a reuse of published maps.

No server was started, so no "armed" banner exists and **no E2E claim is made**.

## 8. Upstream tracking & regression testing (long-term wiring)

- `scripts/upstream_watch.py` + `.github/workflows/upstream-watch.yml`: daily
  read-only HEAD/release watch; deduplicated review issue per new SHA;
  baseline SHA moves only after ROCm review.
- `.github/workflows/rocm-regression.yml`: self-hosted gfx1151 pipeline —
  checkout+pin, preflight, upstream kernel tests, 13 gates, repo tests, small
  receipt (never a published speedup).
- Local: `scripts/run_upstream_kernel_tests.sh` (203-test subset runner) and
  `python -m pytest tests/` (4/4 PASS tonight).

## 9. Top five fix priorities + the concrete next run

1. **Tile the read kernel** (query blocks, `tl.dot`, unioned-id gathers) —
   target ≥2× decode p50 and closing the prefill gap.
2. **128k/256k contexts, subprocess-isolated** — where LSA's density advantage
   compounds; expect the decode curve to keep its slope.
3. **Swap the tree beam onto upstream's Triton beam kernels** (level scoring /
   pick / children) — removes the per-page python loop (1.33 ms → expected
   <0.3 ms select at 32k).
4. **Batched decode** (B×1 rows) — real serving shape; the read kernel already
   supports it, the bench does not yet exercise it.
5. **Fused scan-select above 32,768 buckets** — the production long-context
   select; needs the cutlass-free composition (PORT_NOTES §4).

**Next run (bounded, same machine):** `LSA_MAX_CONTEXT=131072 LSA_REPS=20
bash scripts/campaign.sh` with per-context subprocess isolation, batched-decode
stage, and the tiled read kernel — acceptance: gates 13/13, dense baselines at
128k without OOM, decode speedup curve extended monotonically.

## 10. Evidence index

| Artifact | Path |
|---|---|
| Overnight receipt (36 comparisons, 166 stage rows, 13 gates) | `evidence/kernel-receipt-overnight.json` |
| Quick receipt | `evidence/kernel-receipt-quick.json` |
| Stage rows (JSONL) | `evidence/stage-rows.jsonl` |
| Gate verdicts | `evidence/port-gates.jsonl` |
| Upstream kernel test log (203/7) | `evidence/upstream-kernel-tests.log` |
| Preflight manifest | `evidence/preflight-manifest.json` |
| Overnight stage summary | `evidence/overnight-summary.json` |
| Checksums | `evidence/SHA256SUMS` |
| Full run trees (logs, per-stage receipts, SHA256SUMS) | `results/20261105T145658Z/`, `results/20261105T145917Z/` |

Failed attempts preserved: the entire debugging trail (pipelining miscompile,
decode-slicing bug, oracle PAD race, A2 pairing-semantics fix) is recorded in
docs/PORT_NOTES.md §3–4 and CHANGELOG §Fixed, with the faulting signatures.

## 11. Honest-claims statement

- The bounded benchmark campaign above **did execute to completion on the
  target machine tonight** (both profiles, rc=0). A literal 10-hour soak was
  not run and is not claimed.
- Validated, measured LSA speedups exist **for decode-phase attention-layer
  steps** (5.04× flat, 7.86× block-sparse at 32k) on this GPU. Prefill at
  ≤32k is slower and is reported as such.
- No SGLang integration exists; no E2E/TTFT/TPOT numbers exist; no NVIDIA
  paper number is re-labeled as a Radeon result anywhere in this repository.
