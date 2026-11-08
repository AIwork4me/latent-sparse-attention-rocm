# HOW TO REPRODUCE the overnight_v1 campaign (gfx1151 host)

Hardware/software lane: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151), ROCm pip-wheel
venv (torch 2.14.0+rocm10.1.0, Triton 3.8.0+git669b31ac), upstream pin `5ebea732`,
campaign base main `83e410b`, branch `exp/gfx1151-v1-night-20261108-2155`.

## Environment

```bash
export WT="$HOME/lsa-v1-overnight"
export ROCM_PYTHON="$HOME/Desktop/lsa-rocm-eval-starter/.venv/bin/python"
# third_party/latent-sparse-attention + upstream.sha are untracked: copy them into any
# fresh worktree before running (cp target dir must exist first).
fuser /dev/kfd   # MUST be empty before every GPU step (idle-host discipline)
```

## P0 — start point & evidence integrity

```bash
cd "$WT" && git rev-parse HEAD        # 83e410b + campaign commits
(cd evidence/v0.4.0 && sha256sum -c SHA256SUMS)      # 28/28 OK
python3 scripts/check_workflow_security.py           # PASS
```

## M1 — merged-tree GPU qualification

```bash
cd "$WT"
PYTHONPATH=src "$ROCM_PYTHON" -m pytest tests/ -q                 # 314/314 (GPU)
PYTHONPATH=src "$ROCM_PYTHON" -m lsa_rocm_lab.port.gates          # 17/17 (default)
LSA_TREE_IMPL=batched PYTHONPATH=src "$ROCM_PYTHON" -m lsa_rocm_lab.port.gates
LSA_SELECT_IMPL=fused  PYTHONPATH=src "$ROCM_PYTHON" -m lsa_rocm_lab.port.gates
# single-run A/B confirmation (receipt sha256 9b2026b2):
PYTHONPATH=src "$ROCM_PYTHON" scripts/ab_select_fusion.py --seed 424243 --reps 20 \
  --warmup 5 --dtypes bf16 --contexts 16384,32768 --candidate fused \
  --out /tmp/confirm.json
# independent aggregate recompute: any second implementation; ours compared 60/60 cells
# of evidence/v0.4.0/ab/aggregate-fused.json against the 9 per-seed receipts (tol 5e-4).
```

## M2 — 64k→256k fail-closed ladder

```bash
PYTHONPATH=src "$ROCM_PYTHON" scripts/ctx_scale_ladder.py \
  --ctxs 65536,98304,131072,196608,262144 --lanes decode,prefill \
  --reps 20 --warmup 5 --out /tmp/ladder.json
# smoke first: --ctxs 8192,16384 --reps 5 (cross-checks against campaign bands)
# expectations on this host: decode lane green whole ladder; 64/96/128k prefill measured
# (18/40/70 GiB peaks), 192k+ prefill OOM-classified; wide-path fused <1.0x throughout.
```

## M3 — real-model integration (TinyLlama-1.1B, local HF cache)

```bash
# venv needs transformers + pyarrow (installed from aliyun mirror; NOT global):
HF_HUB_OFFLINE=1 PYTHONPATH=src:third_party/latent-sparse-attention/src \
  "$ROCM_PYTHON" scripts/m3_model_integration.py --model tinyllama \
  --modes probe,consistency,niah,ppl --arms dense,lsa_full,lsa_1024,lsa_256 \
  --out /tmp/m3.json
# probe (interface math): LSA select-everything vs dense causal, cosine 0.9999973.
# Qwen3-0.6B note: zero-training full-dim regroup BLOCKED (Ward index_dim ≤ 256;
# upstream's route is a trained [Hkv*dh → ≤256] map - recorded, not attempted tonight).
```

## Not reproducible tonight (owner-gated or absent)

- GitHub GPU run 42483727525: stays `pending` until runner migration (owner, sudo).
- SGLang serving: no approved gfx1151 container on host (M4 BLOCKED_WITH_ROOT_CAUSE).
- Any v1.0.0 tag/Release/PR#8 merge: explicitly out of bounds.

## Cross-hardware

W7900/gfx1100 and everything non-gfx1151: **NOT_TESTED** — do not cite these numbers
outside this hardware lane.
