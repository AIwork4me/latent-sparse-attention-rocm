# Latent Sparse Attention on ROCm — gfx1151 validation and port

A reproducibility-first, **non-official** ROCm port, validation, and benchmark
harness for [Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (LSA).

Target: **AMD Ryzen AI MAX+ PRO 395 with Radeon 8060S (gfx1151)**, Linux, ROCm 10.1.0
runtime from AMD's pip wheels, PyTorch `2.14.0+rocm10.1.0`, Triton
`3.8.0+git669b31ac.rocm10.1.0`. Upstream is pinned at
`5ebea732d639576d2a7579b0de40e6252f9001f5` and imported read-only.

**What is real here:** the LSA index pipeline (Ward bucket build, node-bucket
hierarchy pairing), the top-k listing and rank-order expansion run **upstream's own
Triton kernels** on the Radeon GPU; the sparse read — which upstream delegates to
NVIDIA-only attention backends — is a **Triton gather-attention kernel written for
this port**; every path is held to upstream's pure-PyTorch oracle by a 16-gate
correctness suite (manifest-enforced: a skipped or short suite can never validate
a receipt). Benchmarks compare dense SDPA, a positional block-sparse arm,
D-LSA Flat and D-LSA Tree on matched shapes with synchronized timings.

**Headline (v0.3.0/T1 multi-seed study — 3 seeds × 3 fresh processes, all
receipts independently audited):** decode select+read at 32k ctx, bf16 —
**block-sparse 4.28×** (cross-seed band 4.14–4.38×, robust), **D-LSA Flat
1.74×** (band 1.69–2.79× — the v0.2.1 "2.1×" figure came from one favorable
seed and does **not** generalize; see the multi-seed section of
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)), D-LSA Tree **0.90× (not faster
than dense)**; prefill remains slower than dense at ≤32k (0.19×). Full
receipt-generated tables: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). **All
v0.2.0 decode speedups are superseded** — the v0.2.0 dense-decode baseline
masked a single-token query to key 0 and was ~2.8× slower than the correct
dispatch, inflating every old speedup ([`docs/RESULTS_v0.2.0.md`](docs/RESULTS_v0.2.0.md),
[`docs/REVIEW_v0.2.1.md`](docs/REVIEW_v0.2.1.md)).

**What is explicitly not claimed:** end-to-end SGLang serving on gfx1151 (BLOCKED,
with documented causes — see the report), any NVIDIA paper number as a Radeon
result, and any speedup that was not measured on this GPU.

## Quick start (fresh machine, nothing on the host is modified)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,devel,device-gfx1151]==10.1.0'
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx1151]==2.14.0+rocm10.1.0' \
  'torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0' \
  'torchaudio==2.11.0.3+rocm10.1.0'
python -m pip install --no-deps -e .          # helper harness only; torch untouched
python -m pip install pytest numpy
bash scripts/checkout_upstream.sh             # pin upstream; never its CUDA lockfile
```

Then, in order:

```bash
python -m lsa_rocm_lab.preflight              # hardware/toolchain manifest (hard checks)
python -m lsa_rocm_lab.correctness            # upstream reference == dense SDPA on GPU
python -m lsa_rocm_lab.triton_probe           # Triton HIP JIT smoke (not LSA)
bash scripts/run_upstream_kernel_tests.sh     # upstream's Triton kernel tests, triaged on ROCm
python -m lsa_rocm_lab.port.gates             # the port's 16-gate oracle parity suite
python -m lsa_rocm_lab.kernel_adapter         # matched benchmarks + validated receipt
python -m lsa_rocm_lab.run --quick            # starter stage chain (BLOCKED-safe)
bash scripts/campaign.sh                      # bounded campaign: pilot + overnight profile
bash scripts/repeat_bench.sh                  # 3 independent repetitions, strategic shapes
python -m pytest tests/ -v                    # the repository's own tests
```

Everything a run produces lands under `results/<timestamp>/` (append-only JSONL,
stage logs, receipts, SHA256SUMS) — see `docs/PORT_CONTRACT.md` for the evidence
contract and `schemas/metrics.schema.json` for the receipt format.

## The port in one picture

```
q, k, v ──regroup (upstream, GQA-collapsed index space)──► index keys / queries
   │                                                        │
   │                     build_buckets_gpu: pools → upstream Triton Ward merge (ieee)
   │                                                        │        (gate A: == torch oracle)
   │                     build_hierarchy_gpu: upstream Triton node-bucket pairing ×⌈log2 size⌉
   │                                                        │        (gate A2: == torch oracle)
   │                     flat_select: bf16 GEMM (fp32 out) → upstream Triton TopK listing
   │                                → packed-key sort → upstream Triton expand_in_order
   │                                                        │        (gates B1/B2: == oracle read)
   │                     tree_select: GPU-torch beam of upstream read_hierarchy semantics
   │                                                        │        (gate C: == reference walk)
   └────────────────────► sparse_read: THIS PORT'S Triton gather-attention kernel
                                                            │        (gates D1/D2/E/F: == oracle/SDPA)
```

Stage attribution is recorded per benchmark row; nothing dense is ever reported as
sparse, and precomputed indices appear only in gather/read microbenchmarks, labeled
as such.

## Status classification

- **PASS** — executed on the GPU with recorded inputs, outputs and receipt.
- **FAIL** — incorrect result or runtime fault; logs preserved.
- **BLOCKED** — no supported ROCm path exists for the stage (e.g. CuTe DSL kernels,
  SGLang E2E on gfx1151). The starter treats absent adapters as BLOCKED, never PASS.
- **TIMEOUT / OOM** — bounded campaign outcomes; recorded, not performance data.

## Layout

```
src/lsa_rocm_lab/
  preflight.py correctness.py dense_baseline.py triton_probe.py run.py   # starter stages
  kernel_adapter.py            # LSA_ROCM_KERNEL_CMD target: gates + matched benchmarks + receipt
  gate_manifest.py             # the required-gate manifest (fail-closed receipts, both ends)
  upstream_triage.py           # upstream-test triage: classified failures visible, new ones fatal
  port/                        # the ROCm port layer (pipeline, read kernel, tree beam, gates, bench)
scripts/                       # checkout, upstream triage, campaign, repetitions, tables, watch
schemas/metrics.schema.json    # versioned receipt schema
tracking/UPSTREAM_BASELINE.json# reviewed upstream SHA (moved only after ROCm review)
docs/BENCHMARKS.md             # GENERATED headline tables (validated receipts only)
docs/RESULTS_v0.2.0.md         # superseded v0.2.0 numbers, kept for the audit trail
docs/REVIEW_v0.2.1.md          # independent review report + GO/NO-GO
tests/                         # gate wrappers + contract tests
```

Upstream: Apache-2.0. This project: Apache-2.0 (see `LICENSE`, `NOTICE`).
 Contributions must keep every gate green and every claim measured — see
`CONTRIBUTING.md`.
