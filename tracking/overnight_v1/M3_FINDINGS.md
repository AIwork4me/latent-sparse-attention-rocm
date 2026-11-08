# M3 findings — true model-level integration (TinyLlama-1.1B, overnight_v1)

Receipts: `m3-model-integration.json` (probe/consistency/ppl), `m3-niah-{dense,lsa_1024,lsa_256}.json`.
Harness: `scripts/m3_model_integration.py`; transformers 5.22.2 custom-attention interface
(registered as `lsa_rocm`), port path `flat_select` + `lsa_read` verbatim, index space =
upstream `regroup` over the model's own RoPE-applied heads (full-dim = RotaryLayout("none")),
**no trained parameters anywhere**.

## 1. Connectability — PROVEN

All 22 attention layers of TinyLlama-1.1B-Chat (Apache-2.0, local cache) run the repo's
select+read inside a real forward — prefill AND token-by-token decode with a real KV cache
(via `model.generate`). Interface math: select-everything vs dense causal attention at
layer 11, random bf16 q/k/v, T=2048: **max_abs 0.00195, cosine 0.9999973**.
Fused-vs-reference select inside the model: layer-7 output **byte-equal (max_abs 0.0)**.

## 2. Quality (PILOT scale, untrained index — honest numbers, no gate PASS claimed)

Perplexity, wikitext-2-raw-v1 test, 12 non-overlapping 2048-token windows (24,576 tokens):

| arm | top_p | ppl | Δ vs dense |
|---|---:|---:|---:|
| dense (eager) | — | 7.7932 | — |
| lsa_full | 4096 (select-all) | 7.7924 | −0.010% |
| lsa_1024 | 1024 (design default) | 7.7924 | −0.010% |
| lsa_256 | 256 (sparse point) | 7.7926 | −0.008% |

NIAH (needle buried ~mid-passage, 1792-token context, 5 cases, **PILOT_ONLY**):
dense 2/5 · lsa_1024 2/5 · lsa_256 2/5 — no additional retrieval loss attributable to
sparsity at this sample size (TinyLlama's own baseline is 40%).

Interpretation (pre-registered as expected-degradation; measured as near-zero instead):
window 512 + sink guarantee recency; wikitext NLL is dominated by local context. **Do not
extrapolate to long-range tasks from this pilot** — that needs the pre-registered RULER/NIAH
campaign (≥100 cases/context) and ideally trained index projections.

## 3. Integration cost (MODEL_ONLY_NOT_SERVING label)

- wall time, 12 windows: dense 9.8 s · lsa_1024 13.9 s · lsa_full 37.4 s — the integrated
  path rebuilds buckets every layer every forward (build p50 11.1 ms ×22×12) and the fused
  read is not the win it is standalone; **no serving-level claim** (mission M5 rules).
- peak alloc: lsa arms 2.78 GiB vs dense 3.89 GiB (no [T,T] score materialization).

## 4. Product defects surfaced (value of running a real model)

1. **`build_buckets_gpu` requires T ≥ cluster_block_size (512)**: transformers' chunked
   prefill produced a 256-token tail chunk → Ward clustering crashed (0 closed blocks).
   Real serving hits this in early decode too. Adapter workaround tonight: dense-causal
   fallback for T<512 (mathematically equal to select-everything), counted, never silent.
   → engineering follow-up: short-sequence support in the port.
2. **Ward kernel index_dim ∈ {16..256 pow2}**: zero-training full-dim regroup works only
   for models with Hkv·head_dim ≤ 256 (TinyLlama 4×64=256 exactly). Qwen3-0.6B (8×128=1024)
   is BLOCKED without upstream's trained `[Hkv·dh → ≤256]` projection route — consistent
   with upstream's own design (Qwen3.5 uses a learned map over position-free dims).
3. Decode-step query rows: `iq` holds exactly the Tq query rows (no tail slicing) — fixed
   in harness after prefill-only paths masked it.

## 5. Status

M3 target (v0.5 "real model quality gates"): **PILOT_DONE / CAMPAIGN_NOT_RUN** — the
integration exists and is instrumented; the ≥100-case pre-registered quality campaign was
not run tonight (time + untrained-index scope). No v0.5 PASS claimed.
