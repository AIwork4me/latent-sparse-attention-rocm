#!/usr/bin/env python3
"""v0.4.0/B4 - interleaved A/B of the fused select against the v0.3.1 reference.

Both arms time the PRODUCTION entry point (``pipeline.flat_select`` under the
``LSA_SELECT_IMPL`` switch) on IDENTICAL inputs; the correctness gate (byte-exact
lists + lengths vs the reference) runs BEFORE any timing and again after the last
sample - a candidate that diverges is reported FAILED and gets no speedup. The
two arms interleave sample-by-sample (ref, cand, ref, cand, ...) so thermal or
temporal drift hits both equally. Receipt methodology throughout: warmup 5,
reps >= 20, synchronized wall time, raw samples retained.

Per (dtype, context, phase) row the receipt records both arms' p50/p95/min/max,
launches per iteration (device rows of a profiler pass), peak allocation, first
(fused-compile) call, the dense baseline for the same shape, and the speedups
computed from p50s. Rows the run did not measure are simply absent - an
unmeasured configuration is NOT_RUN, never PASS.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline, upstream_sha  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402

ARMS = ["reference", "fused", "fused_unpack"]


def timed_pair(fn_ref, fn_cand, warmup, reps):
    """Interleaved synchronized samples for both arms; returns raw sample lists."""
    for _ in range(warmup):
        fn_ref()
        fn_cand()
    sync()
    ref_samples, cand_samples = [], []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn_ref()
        sync()
        ref_samples.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        fn_cand()
        sync()
        cand_samples.append((time.perf_counter() - t0) * 1000)
    return ref_samples, cand_samples


def launches_per_iter(fn, iters=8):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        sync()
    rows = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    return sum(e.count for e in rows) / iters


def summarize(samples):
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": [round(s, 6) for s in samples],
    }


def select_fn(impl, idx_rows, positions, buckets, config):
    def fn():
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            return pipeline.flat_select(idx_rows, positions, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
    return fn


def select_read_fn(impl, q_read, k, v, idx_rows, positions, buckets, config, scale):
    def fn():
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            lists, lengths = pipeline.flat_select(idx_rows, positions, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
        return pipeline.lsa_read(q_read, k, v, lists, lengths, scale)
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--contexts", default="512,2048,8192,16384,32768")
    parser.add_argument("--candidate", default="fused", choices=["fused", "fused_unpack"])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ragged-rows", type=int, default=32)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1151"):
        print(json.dumps({"status": "BLOCKED", "reason": f"not gfx1151: {arch}"}))
        return 78

    repo = Path(__file__).resolve().parents[1]
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src/"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_arch": arch,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "production_code_sha": code_sha,
        "production_src_dirty": dirty,
        "upstream_sha": upstream_sha(),
        "seed": args.seed,
        "candidate": args.candidate,
        "baseline": "reference",
        "warmup": args.warmup,
        "reps": args.reps,
        "interleaved": True,
        "gpu_state": subprocess.run(["rocm-smi", "--showclocks", "--showtemp", "--showmeminfo", "vram"],
                                    capture_output=True, text=True).stdout[-800:],
        "rows": [],
    }
    cand = args.candidate

    for dtype_name in args.dtypes.split(","):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        for context in [int(c) for c in args.contexts.split(",")]:
            q, k, v = build_geometry(context, dtype, seed=args.seed + context, **GEOMETRY)
            config = config_for(context)
            idx_q, idx_k = make_index_space(q, k)
            buckets = pipeline.build_buckets_gpu(idx_k, config)
            scale = q.shape[-1] ** -0.5
            positions_all = torch.arange(context, device=q.device)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + context + 7)
            ragged = torch.sort(torch.randint(1, context, (args.ragged_rows,), generator=gen)).values.to(q.device)
            shapes = [
                ("prefill", positions_all, idx_q, q),
                ("decode_b1", positions_all[-1:], idx_q[-1:], q[:, -1:]),
                # ragged rows as batch-1 x 32 query rows (the read kernel addresses
                # k/v by each row's batch: batch must match k/v's batch 1)
                ("decode_ragged", ragged, idx_q[ragged.long()], q[0, ragged.long()].unsqueeze(0)),
            ]
            # dense baselines for the same shape (same conditions)
            dense = {}
            dense["prefill"] = _dense_ms(lambda: pipeline.dense_read(q, k, v, scale), args.warmup, args.reps)
            q_tail = q[:, -1:]
            dense["decode"] = _dense_ms(lambda: pipeline.dense_read(q_tail, k, v, scale), args.warmup, args.reps)

            for phase, positions, idx_rows, q_read in shapes:
                # ---- correctness gate first, both candidate arms, byte-exact
                verdicts = {}
                for impl in [cand, "fused_unpack"]:
                    os.environ["LSA_SELECT_IMPL"] = impl
                    try:
                        got = pipeline.flat_select(idx_rows, positions, buckets, config)
                    finally:
                        os.environ.pop("LSA_SELECT_IMPL", None)
                    want = pipeline.flat_select_reference(idx_rows, positions, buckets, config)
                    verdicts[impl] = bool(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))
                    del got, want
                gate_ok = verdicts.get(cand, False)

                row = {"dtype": dtype_name, "context": context, "phase": phase,
                       "rows": int(positions.shape[0]), "correctness_exact": verdicts,
                       "dense_ms": dense}
                if not gate_ok:
                    row["status"] = "FAILED"
                    receipt["rows"].append(row)
                    print(f"[{dtype_name} {context} {phase}] CORRECTNESS FAILED {verdicts}", flush=True)
                    continue

                for stage_kind in ["select", "select_read"]:
                    if stage_kind == "select":
                        fn_ref = select_fn("reference", idx_rows, positions, buckets, config)
                        fn_cand = select_fn(cand, idx_rows, positions, buckets, config)
                    else:
                        fn_ref = select_read_fn("reference", q_read, k, v, idx_rows, positions, buckets, config, scale)
                        fn_cand = select_read_fn(cand, q_read, k, v, idx_rows, positions, buckets, config, scale)
                    # first-call (compile) probe for the candidate, after a reference warm call
                    fn_ref()
                    sync(); t0 = time.perf_counter()
                    fn_cand(); sync()
                    first_call_ms = (time.perf_counter() - t0) * 1000
                    ref_s, cand_s = timed_pair(fn_ref, fn_cand, args.warmup, args.reps)
                    entry = {
                        "stage": stage_kind,
                        "reference": summarize(ref_s),
                        "candidate": summarize(cand_s),
                        "launches_reference": launches_per_iter(fn_ref),
                        "launches_candidate": launches_per_iter(fn_cand),
                        "candidate_first_call_ms": first_call_ms,
                        "speedup_p50": statistics.median(ref_s) / statistics.median(cand_s),
                    }
                    if stage_kind == "select_read" and phase in ("prefill", "decode_b1"):
                        dense_ms = dense["prefill" if phase == "prefill" else "decode"]
                        entry["vs_dense_reference"] = dense_ms / statistics.median(ref_s)
                        entry["vs_dense_candidate"] = dense_ms / statistics.median(cand_s)
                    row.setdefault("stages", {})[stage_kind] = entry
                    print(f"[{dtype_name} {context} {phase} {stage_kind}] ref p50 "
                          f"{entry['reference']['p50_ms']:.3f} cand p50 {entry['candidate']['p50_ms']:.3f} "
                          f"({entry['speedup_p50']:.3f}x, L {entry['launches_reference']:.0f}->"
                          f"{entry['launches_candidate']:.0f})", flush=True)
                # post-timing re-verification (drift check)
                os.environ["LSA_SELECT_IMPL"] = cand
                try:
                    got = pipeline.flat_select(idx_rows, positions, buckets, config)
                finally:
                    os.environ.pop("LSA_SELECT_IMPL", None)
                want = pipeline.flat_select_reference(idx_rows, positions, buckets, config)
                row["correctness_exact_post"] = bool(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))
                row["status"] = "PASS" if row["correctness_exact_post"] else "FAILED"
                del got, want
                receipt["rows"].append(row)
            del q, k, v, idx_q, idx_k, buckets
            torch.cuda.empty_cache()

    receipt["summary"] = {
        "rows_total": len(receipt["rows"]),
        "rows_pass": sum(1 for r in receipt["rows"] if r.get("status") == "PASS"),
        "rows_failed": sum(1 for r in receipt["rows"] if r.get("status") == "FAILED"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}\nsha256 {digest}", flush=True)
    return 0


def _dense_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


if __name__ == "__main__":
    raise SystemExit(main())
