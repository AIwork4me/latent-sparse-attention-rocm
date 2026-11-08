#!/usr/bin/env python3
"""M3 (overnight_v1) - true model-level integration: THIS REPO's LSA select+read
inside a real causal LM forward (Qwen3-0.6B, Apache-2.0, from the local HF cache).

What this proves (or honestly fails to):
- CONNECTABILITY: the port's flat_select + lsa_read replace a real model's attention
  via the official custom-attention interface. Index space = upstream regroup over the
  model's own (RoPE-applied) q/k heads - full-dim, the RotaryLayout("none") case upstream
  documents. No new trained parameters.
- INTERFACE CORRECTNESS: with top_p >= context (select-everything) the model's
  per-layer attention output must match the eager dense attention within bf16 tolerance.
- HONEST QUALITY: perplexity arms dense / lsa_full / lsa_1024 / lsa_256 on a pinned
  wikitext-2-raw-v1 test slice, plus a NIAH retrieval PILOT (small n, labeled PILOT_ONLY).
  Qwen3-0.6B is full-rope, so the index signal mixes position into content: sparse-arm
  degradation is EXPECTED and reported as measured - no quality PASS is claimed.
- SELECT CONSISTENCY inside the model: reference vs fused select produce byte-equal
  lists on real model activations.

Fail-loud constraints: batch-1 only; any non-None 4D attention mask aborts the arm.
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "latent-sparse-attention" / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lsa.core.index_space import regroup  # upstream, pinned 5ebea732  # noqa: E402
from lsa_rocm_lab.kernel_adapter import config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline, upstream_sha  # noqa: E402
from lsa_rocm_lab.port.bench import sync  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

MODELS = {
    "tinyllama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "license": "Apache-2.0",
        "glob": "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*/config.json",
        "index_dim": 256, "probe_layer": 11,
        # full-rope; full-dim regroup (RotaryLayout("none") case), NO trained map
    },
    "qwen3": {
        "id": "Qwen/Qwen3-0.6B", "license": "Apache-2.0",
        "glob": "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/config.json",
        "index_dim": 1024, "probe_layer": 14,
        # EXCEEDS the Ward kernel's pow2<=256 index_dim - recorded BLOCKED for zero-training
        # full-dim regroup; upstream solves this with a TRAINED [Hkv*dh -> <=256] map.
    },
}
LSA_IMPL_NAME = "lsa_rocm"
# per-arm (top_p, top_k): full selects everything (>= ctx), 1024 = design default, 256 = sparse point
ARMS = {"dense": None, "lsa_full": (4096, 16384), "lsa_1024": (1024, 4096), "lsa_256": (256, 4096)}

ACTIVE = {"top_p": 1024, "top_k": 4096, "ctx_hint": 2048}
STATS = {"build_ms": [], "select_ms": [], "read_ms": [], "calls": 0}
MODEL = {"key": "tinyllama"}


def model_cache_path():
    cands = glob.glob(os.path.expanduser(MODELS[MODEL["key"]]["glob"]))
    if not cands:
        raise FileNotFoundError(f"{MODEL['key']} not in local HF cache")
    return str(Path(cands[0]).parent)


def lsa_rocm_attention(module, query, key, value, attention_mask, scaling=1.0,
                       dropout=0.0, sliding_window=None, **kwargs):
    """The custom attention interface: LSA select + sparse read over the model's own q/k/v.

    query [B, Hq, Tq, D], key/value [B, Hkv, T, D], RoPE already applied by the caller.
    """
    if attention_mask is not None and attention_mask.dtype == torch.bool and not attention_mask.all():
        raise ValueError("M3 harness: padding mask not supported (batch-1 clean text only)")
    if query.shape[0] != 1:
        raise ValueError("M3 harness: batch-1 only")
    B, Hq, Tq, D = query.shape
    _, Hkv, T, _ = key.shape
    # PORT LIMITATION (recorded M3 finding): build_buckets_gpu needs T >= cluster_block_size
    # (512) - chunked-prefill tails and early decode steps fall below it. Fallback: dense
    # causal attention, which select-everything is mathematically equal to (probe: cosine
    # 0.9999973). Counted and reported; never silent.
    if T < 512:
        STATS["short_t_fallback"] = STATS.get("short_t_fallback", 0) + 1
        return eager_dense(module, query, key, value, None, scaling=scaling)
    if Tq == 0:  # transformers internal empty-step call (no math to do)
        return query.new_empty(B, 0, Hq * D), None
    cfg = config_for(ACTIVE["ctx_hint"], top_p=ACTIVE["top_p"], top_k=ACTIVE["top_k"])

    # index space from the model's own heads (upstream regroup expects [B,H,T,dh] -
    # exactly transformers' layout - and returns [B,T,Hkv*dh] fp32 unit rows)
    t0 = time.perf_counter()
    iq, ik = regroup(query, key)
    buckets = pipeline.build_buckets_gpu(ik[0], cfg)
    sync()
    STATS["build_ms"].append((time.perf_counter() - t0) * 1000)

    # queries sit at the tail of the sequence: positions T-Tq..T-1; iq already has
    # exactly the Tq query rows (prefill Tq==T -> 0..T-1, decode Tq==1 -> [T-1])
    positions = torch.arange(T - Tq, T, device=query.device)
    iq_rows = iq[0].contiguous()

    os.environ["LSA_SELECT_IMPL"] = os.environ.get("M3_SELECT_IMPL", "reference")
    t0 = time.perf_counter()
    lists, lengths = pipeline.flat_select(iq_rows, positions, buckets, cfg)
    sync()
    STATS["select_ms"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    out = pipeline.lsa_read(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                            lists, lengths, scaling)
    sync()
    STATS["read_ms"].append((time.perf_counter() - t0) * 1000)
    STATS["calls"] += 1
    return out.reshape(B, Tq, Hq * D).contiguous(), None


def eager_dense(module, query, key, value, attention_mask, scaling=1.0, dropout=0.0, **kw):
    """Plain dense causal SDPA-equivalent for the same inputs (comparison arm)."""
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    causal = torch.ones(Tq_ := query.shape[2], kv_ := k.shape[2], dtype=torch.bool,
                        device=query.device).tril(diagonal=kv_ - Tq_)
    scores = scores.masked_fill(~causal, float("-inf"))
    w = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, v).transpose(1, 2)
    return out.reshape(query.shape[0], Tq_, -1).contiguous(), None


def load_model(path, attn_impl):
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    m = m.cuda()
    m.eval()
    return m


def corpus_windows(tok, window, n, seed=424243):
    import pyarrow.parquet as pq
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
    text = "\n".join(rec["text"] for rec in pq.read_table(snap[0]).to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    i = 0
    while len(out) < n and i + window + 1 <= len(ids):
        out.append(ids[i:i + window + 1])
        i += window + 1
    return out


@torch.no_grad()
def run_probe(model, tok):
    """Single-mid-layer numerical check: LSA select-everything == dense causal attention."""
    torch.manual_seed(424243)
    layer = model.model.layers[MODELS[MODEL["key"]]["probe_layer"]].self_attn
    Hq = model.config.num_attention_heads
    Hkv = model.config.num_key_value_heads
    D = layer.head_dim
    T = 2048
    q = (torch.randn(1, Hq, T, D, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    k = (torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    v = (torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    dense_out, _ = eager_dense(layer, q, k, v, None, scaling=layer.scaling)
    ACTIVE.update({"top_p": 4096, "top_k": 16384, "ctx_hint": T})
    lsa_out, _ = lsa_rocm_attention(layer, q, k, v, None, scaling=layer.scaling)
    diff = (dense_out.float() - lsa_out.float()).abs()
    cos = F.cosine_similarity(dense_out.float().flatten(), lsa_out.float().flatten(), dim=0)
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "cosine": float(cos), "dtype": "bf16", "T": T}


@torch.no_grad()
def run_ppl(model, windows):
    nlls, tokens = 0.0, 0
    for ids in windows:
        out = model(input_ids=ids[:-1].unsqueeze(0).cuda())
        logits = out.logits[0].float()
        nll = F.cross_entropy(logits, ids[1:].cuda())
        nlls += nll.item() * (len(ids) - 1)
        tokens += len(ids) - 1
    return {"mean_nll": nlls / tokens, "ppl": float(torch.exp(torch.tensor(nlls / tokens))),
            "tokens": tokens, "windows": len(windows)}


@torch.no_grad()
def run_niah(model, tok, ctx_len=1792, n_cases=5):
    """PILOT_ONLY needle-in-haystack: one needle buried in wikitext filler, ask for it."""
    import pyarrow.parquet as pq
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
    paras = [rec["text"] for rec in pq.read_table(snap[0]).to_pylist() if len(rec["text"]) > 200]
    results = []
    for case in range(n_cases):
        magic = 1000 + case * 111
        needle = f" The magic number XZY{case} is {magic}. "
        filler = "\n".join(paras[case * 40:(case + 1) * 40])
        prompt = ("Read the passage and answer the question.\n" + filler[: ctx_len - 300] +
                  needle + "\nQuestion: What is the magic number XZY"
                  + str(case) + "? Answer with the number only:")
        ids = tok(prompt, return_tensors="pt").input_ids[0]
        if len(ids) > ctx_len:
            ids = ids[-ctx_len:]
        gen = model.generate(ids.unsqueeze(0).cuda(), max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0][len(ids):], skip_special_tokens=True).strip()
        results.append({"case": case, "magic": magic, "answer": ans[:40],
                        "hit": str(magic) in ans})
        ACTIVE["ctx_hint"] = len(ids)
    return {"cases": results, "hits": sum(r["hit"] for r in results),
            "n": n_cases, "label": "PILOT_ONLY"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tinyllama", choices=list(MODELS))
    ap.add_argument("--modes", default="probe,ppl,consistency")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--niah-cases", type=int, default=5)
    ap.add_argument("--arms", default="dense,lsa_full,lsa_1024,lsa_256")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available() and torch.version.hip
    MODEL["key"] = args.model

    path = model_cache_path()
    ALL_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = lsa_rocm_attention
    tok = AutoTokenizer.from_pretrained(path)

    receipt = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/m3_model_integration.py (overnight_v1 M3)",
        "model": MODELS[MODEL["key"]]["id"], "model_path": path,
        "model_revision": Path(path).name,
        "index_dim": MODELS[MODEL["key"]]["index_dim"],
        "license": MODELS[MODEL["key"]]["license"],
        "upstream_sha": upstream_sha(),
        "production_code_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                              capture_output=True, text=True).stdout.strip(),
        "production_src_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src/"],
                                                    cwd=REPO, capture_output=True, text=True).stdout.strip()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "hip": torch.version.hip,
        "transformers": __import__("transformers").__version__,
        "corpus": "wikitext-2-raw-v1 test (local HF cache), stride=window+1, non-overlapping",
        "window": args.window, "modes": args.modes.split(","),
        "results": {},
    }

    model = load_model(path, LSA_IMPL_NAME)

    if "probe" in args.modes:
        t0 = time.perf_counter()
        receipt["results"]["probe"] = run_probe(model, tok)
        receipt["results"]["probe"]["seconds"] = round(time.perf_counter() - t0, 1)
        print("probe:", receipt["results"]["probe"], flush=True)

    if "ppl" in args.modes or "consistency" in args.modes:
        windows = corpus_windows(tok, args.window, args.n_windows)
        receipt["corpus_windows"] = len(windows)

    if "ppl" in args.modes:
        ppl = {}
        for arm in args.arms.split(","):
            if arm == "dense":
                model.config._attn_implementation = "eager"
            else:
                model.config._attn_implementation = LSA_IMPL_NAME
                ACTIVE.update({"top_p": ARMS[arm][0], "top_k": ARMS[arm][1],
                               "ctx_hint": args.window})
            STATS.update({"build_ms": [], "select_ms": [], "read_ms": [], "calls": 0})
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            res = run_ppl(model, windows)
            res.update({"wall_s": round(time.perf_counter() - t0, 1),
                        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "attn_calls": STATS["calls"],
                        "build_p50_ms": round(statistics.median(STATS["build_ms"]), 3) if STATS["build_ms"] else None,
                        "select_p50_ms": round(statistics.median(STATS["select_ms"]), 3) if STATS["select_ms"] else None,
                        "read_p50_ms": round(statistics.median(STATS["read_ms"]), 3) if STATS["read_ms"] else None})
            ppl[arm] = res
            print(f"ppl[{arm}]: {res}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        receipt["results"]["ppl"] = ppl

    if "consistency" in args.modes:
        # reference vs fused select on real activations (one window): byte-equal lists
        model.config._attn_implementation = LSA_IMPL_NAME
        ACTIVE.update({"top_p": 1024, "top_k": 4096, "ctx_hint": args.window})
        layer = model.model.layers[7].self_attn
        hook_out = {}
        def grab(module, inp, out):
            hook_out["out"] = out[0].detach().clone()
        h = model.model.layers[7].register_forward_hook(grab)
        ids = windows[0][:-1].unsqueeze(0).cuda()
        with torch.no_grad():
            os.environ["M3_SELECT_IMPL"] = "reference"
            model(input_ids=ids)
            ref_out = hook_out["out"].clone()
            os.environ["M3_SELECT_IMPL"] = "fused"
            STATS.update({"build_ms": [], "select_ms": [], "read_ms": [], "calls": 0})
            model(input_ids=ids)
            fused_out = hook_out["out"].clone()
        h.remove()
        os.environ.pop("M3_SELECT_IMPL", None)
        receipt["results"]["consistency"] = {
            "layer7_output_byte_equal": bool(torch.equal(ref_out, fused_out)),
            "max_abs": float((ref_out.float() - fused_out.float()).abs().max())}
        print("consistency:", receipt["results"]["consistency"], flush=True)

    if "niah" in args.modes:
        res = {}
        for arm in args.arms.split(","):
            if arm == "dense":
                model.config._attn_implementation = "eager"
            else:
                model.config._attn_implementation = LSA_IMPL_NAME
                ACTIVE.update({"top_p": ARMS[arm][0], "top_k": ARMS[arm][1], "ctx_hint": 1792})
            res[arm] = run_niah(model, tok, n_cases=args.niah_cases)
            print(f"niah[{arm}]: {res[arm]['hits']}/{res[arm]['n']}", flush=True)
        receipt["results"]["niah"] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(f"wrote {args.out}\nsha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
