#!/usr/bin/env python3
"""v0.4.0/B1 - substage breakdown of the production flat select on gfx1151.

Times the REAL production stages of ``pipeline.flat_select`` (score GEMM, the
upstream Triton top-k listing/pack, the second sorted top-k, the id unpack,
the rank-order expansion) plus the complete select and select+read, on
identical inputs, with the receipt methodology (warmup 5 / reps 20,
synchronized). Also records launch counts (device rows of a profiler pass)
and peak allocations per stage. Pure instrumentation: every timed callable is
the production code at the pinned tree (no reimplementation here).

Rows are written for prefill (all positions), single-token decode (batch 1)
and a ragged multi-batch decode (B=32 random positions).
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline, triton_top_k, upstream_sha  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402
from lsa.core.kernels.triton.keys import unpack_ids  # noqa: E402
from lsa.core.kernels.triton.expansion import expand_in_order  # noqa: E402

CONTEXTS = [512, 2048, 8192, 16384, 32768]


def timed(fn, warmup=5, reps=20) -> dict:
    """Receipt-methodology synchronized p50/p95/min/max + samples."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": [round(s, 6) for s in samples],
    }


def launches_per_iter(fn, iters=8) -> tuple[float, list]:
    """Device-side kernel rows of a profiler pass: launch count + attribution."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        sync()
    rows = [
        {"name": e.key, "count": e.count, "total_us": e.self_device_time_total}
        for e in prof.key_averages()
        if e.self_device_time_total > 0
    ]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return sum(r["count"] for r in rows) / iters, rows[:10]


def peak_alloc(fn) -> tuple[int, object]:
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    sync()
    return torch.cuda.max_memory_allocated() - before, out


def substage_stages(idx_q, positions, buckets, config, dtype_name, q, k, v, scale, lists, lengths):
    """(name, fn) production stages on identical inputs; later fns rebuild their own inputs."""
    counts = pipeline.visible_counts(positions, buckets, config)
    listing = triton_top_k.TopK(config.top_p, exact=False)
    top_k = config.top_k

    def score_stage():
        queries = idx_q.to(torch.bfloat16)
        return torch.mm(queries, buckets.centroids.t(), out_dtype=torch.float32)

    def pack_stage():
        return listing.listed(score_stage(), counts)

    keys0, _ = listing.listed(score_stage(), counts)
    ordered0 = pipeline.best_first_keys(keys0, config.top_p)
    picked0 = unpack_ids(ordered0)

    def sort_stage():
        return pipeline.best_first_keys(keys0, config.top_p)

    def unpack_stage():
        return unpack_ids(ordered0)

    def expand_stage():
        return expand_in_order(picked0, buckets.members, positions.int(),
                               config.window_size, config.sink_size, top_k)

    def select_stage():
        return pipeline.flat_select(idx_q, positions, buckets, config)

    def select_read_stage():
        lists_i, lengths_i = pipeline.flat_select(idx_q, positions, buckets, config)
        return pipeline.lsa_read(q, k, v, lists_i, lengths_i, scale)

    return [
        ("score_gemm", score_stage),
        ("topk_pack_listed", pack_stage),
        ("topk_sort_best_first", sort_stage),
        ("unpack_ids", unpack_stage),
        ("expand_in_order", expand_stage),
        ("select_complete", select_stage),
        ("select_plus_read", select_read_stage),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--batch-ragged", type=int, default=32)
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
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/"], cwd=repo,
                           capture_output=True, text=True).stdout.strip()
    env = {
        "gpu_arch": arch,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "production_code_sha": code_sha,
        "production_src_dirty": bool(dirty),
        "upstream_sha": upstream_sha(),
        "seed": args.seed,
        "clock_smoking_gun": subprocess.run(
            ["rocm-smi", "--showclocks", "--showtemp", "--showmeminfo", "vram"],
            capture_output=True, text=True).stdout[-1200:],
    }
    report = {"environment": env, "rows": []}

    for dtype_name in args.dtypes.split(","):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        for context in [int(c) for c in args.contexts.split(",")]:
            q, k, v = build_geometry(context, dtype, seed=args.seed + context, **GEOMETRY)
            config = config_for(context)
            idx_q, idx_k = make_index_space(q, k)
            buckets = pipeline.build_buckets_gpu(idx_k, config)
            scale = q.shape[-1] ** -0.5
            positions_all = torch.arange(context, device=q.device)
            gen = torch.Generator(device="cpu").manual_seed(args.seed + context)
            ragged = torch.sort(torch.tensor(
                [int(x) for x in torch.randint(1, context, (args.batch_ragged,),
                                               generator=gen)])).values.to(q.device)
            shapes = [
                ("prefill", positions_all, idx_q, q),
                ("decode_b1", positions_all[-1:], idx_q[-1:], q[:, -1:]),
                ("decode_ragged_b32", ragged, idx_q[ragged.long()], q[:, -1:]),
            ]
            for phase, positions, queries, q_read in shapes:
                lists, lengths = pipeline.flat_select(queries, positions, buckets, config)
                stages = substage_stages(queries, positions, buckets, config, dtype_name,
                                         q_read, k, v, scale, lists, lengths)
                row = {"dtype": dtype_name, "context": context, "phase": phase,
                       "rows": int(positions.shape[0]), "stages": {}}
                for name, fn in stages:
                    entry, _ = peak_alloc(lambda: fn())  # alloc probe (also warms)
                    t = timed(fn, reps=args.reps)
                    launches, kernels = launches_per_iter(fn)
                    row["stages"][name] = {
                        **t, "launches_per_iter": launches,
                        "peak_alloc_bytes": entry, "top_kernels": kernels[:6],
                    }
                    print(f"[{dtype_name} {context} {phase}] {name}: p50 {t['p50_ms']:.3f} ms, "
                          f"{launches:.0f} launches, peak {entry / 2**20:.1f} MiB", flush=True)
                report["rows"].append(row)
                del lists, lengths, stages
            del q, k, v, idx_q, idx_k, buckets
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"sha256 {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
