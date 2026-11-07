#!/usr/bin/env python3
"""T2 stage profiler: cold-first-call vs warm, with device kernel attribution.

For every stage of the LSA step (and the dense baselines) this reports:
- ``--mode cold``: the FIRST invocation wall time in a fresh process with an
  isolated empty Triton cache (compile + first launch), measured inside the
  dependency-ordered build sequence itself so no stage is silently warmed by
  the setup that produces its inputs;
- ``--mode warm``: synchronized p50/p95 over reps plus a torch.profiler pass
  whose kernel table keeps only device-side rows (HIP kernel names are the
  IDs), shares of the GPU total - never python/launch time mislabeled as
  kernel time.

The output feeds docs/OPTIMIZATION_v0.3.md's bottleneck table; it makes no
claims by itself. Stages later in the sequence share already-compiled kernels
with earlier ones (e.g. the decode selects reuse the prefill selects' Triton
binaries): their first-call rows are launch-cost, not compile-cost - the
interpretation notes in the doc say which.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402
from lsa_rocm_lab.port.read_kernel import sparse_read  # noqa: E402


def kernel_rows(events: list) -> list[dict]:
    """Device-side attribution rows: name, count, total us, share, cumulative.

    Rows with zero device time (pure python/aten bookkeeping) are dropped so
    the shares are shares of GPU time only.
    """
    kept = [{"name": e.key, "count": e.count, "total_us": e.self_device_time_total,
             "cpu_us": e.self_cpu_time_total}
            for e in events if e.self_device_time_total > 0]
    kept.sort(key=lambda r: r["total_us"], reverse=True)
    total = sum(r["total_us"] for r in kept)
    cumulative = 0.0
    for row in kept:
        row["share"] = row["total_us"] / total if total else 0.0
        cumulative += row["share"]
        row["cumulative"] = cumulative
    return kept


def format_kernel_table(rows: list[dict], top: int = 10) -> str:
    lines = ["| kernel | count | total us | share |", "|---|---:|---:|---:|"]
    for row in rows[:top]:
        lines.append(f"| {row['name']} | {row['count']} | {row['total_us']:.0f} "
                     f"| {row['share'] * 100:.1f}% |")
    return "\n".join(lines)


def build_workspace(context: int, dtype_name: str, seed: int, first_call_log: list | None = None):
    """Build the matched tensors and return (name, fn) stages in dependency order.

    With ``first_call_log``, every setup call IS the stage's first-call
    measurement (appended to the log) - nothing runs twice, nothing is
    pre-warmed. Without it, setup runs uninstrumented and the caller does the
    warm/profiler pass over the returned stages.
    """
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]

    def run(name, fn):
        if first_call_log is None:
            return fn()
        sync()
        start = time.perf_counter()
        out = fn()
        sync()
        first_call_log.append({"stage": name, "first_call_ms": (time.perf_counter() - start) * 1000})
        return out

    q, k, v = build_geometry(context, dtype, seed=seed + context, **GEOMETRY)
    config = config_for(context)
    idx_q, idx_k = run("index_regroup", lambda: make_index_space(q, k))
    buckets = run("ward_cluster", lambda: pipeline.build_buckets_gpu(idx_k, config))
    hierarchy = run("hierarchy_build", lambda: pipeline.build_hierarchy_gpu(idx_k, buckets, config))
    positions = torch.arange(q.shape[1], device=q.device)
    dec_pos, dec_idx_q = positions[-1:], idx_q[-1:]
    scale = q.shape[-1] ** -0.5
    flat_lists, flat_lengths = run("flat_select_prefill",
                                   lambda: pipeline.flat_select(idx_q, positions, buckets, config))
    flat_dec_lists, flat_dec_lengths = run("flat_select_decode",
                                           lambda: pipeline.flat_select(dec_idx_q, dec_pos, buckets, config))
    tree_lists, tree_lengths = run("tree_select_prefill",
                                    lambda: pipeline.tree_select(idx_q, positions, buckets, hierarchy, config))
    tree_dec_lists, tree_dec_lengths = run("tree_select_decode",
                                            lambda: pipeline.tree_select(dec_idx_q, dec_pos, buckets, hierarchy, config))
    q_tail = q[:, -1:]

    stages = [
        ("dense_prefill", lambda: pipeline.dense_read(q, k, v, scale)),
        ("dense_decode", lambda: pipeline.dense_read(q_tail, k, v, scale)),
        ("index_regroup", lambda: make_index_space(q, k)),
        ("ward_cluster", lambda: pipeline.build_buckets_gpu(idx_k, config)),
        ("hierarchy_build", lambda: pipeline.build_hierarchy_gpu(idx_k, buckets, config)),
        ("flat_select_prefill", lambda: pipeline.flat_select(idx_q, positions, buckets, config)),
        ("flat_select_decode", lambda: pipeline.flat_select(dec_idx_q, dec_pos, buckets, config)),
        ("tree_select_prefill", lambda: pipeline.tree_select(idx_q, positions, buckets, hierarchy, config)),
        ("tree_select_decode", lambda: pipeline.tree_select(dec_idx_q, dec_pos, buckets, hierarchy, config)),
        ("sparse_read_prefill_flat", lambda: sparse_read(q, k, v, flat_lists, flat_lengths, scale)),
        ("sparse_read_decode_flat", lambda: sparse_read(q_tail, k, v, flat_dec_lists, flat_dec_lengths, scale)),
        ("sparse_read_prefill_tree", lambda: sparse_read(q, k, v, tree_lists, tree_lengths, scale)),
        ("sparse_read_decode_tree", lambda: sparse_read(q_tail, k, v, tree_dec_lists, tree_dec_lengths, scale)),
    ]
    return stages


def _guard_gfx1151() -> str:
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        sys.exit(78)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1151"):
        print(json.dumps({"status": "BLOCKED", "reason": f"not gfx1151: {arch}"}))
        sys.exit(78)
    return arch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cold", "warm"], required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    arch = _guard_gfx1151()

    if args.mode == "cold":
        # A driver sets TRITON_CACHE_DIR to an empty dir for a true compile-cold
        # pass; every setup call is the timed first call of its stage.
        log: list = []
        stages = build_workspace(args.context, args.dtype, args.seed, first_call_log=log)
        seen = {entry["stage"] for entry in log}
        for name, fn in stages:  # stages not exercised by setup: dense + sparse reads
            sync()
            start = time.perf_counter()
            fn()
            sync()
            elapsed = (time.perf_counter() - start) * 1000
            # setup already exercised the pipeline stages: this second call is
            # launch-cost reference, labeled as such - never a compile measurement
            key = "second_call_ms" if name in seen else "first_call_ms"
            log.append({"stage": name, key: elapsed})
        payload = {"mode": "cold", "context": args.context, "dtype": args.dtype,
                   "seed": args.seed, "gpu_arch": arch, "stages": log,
                   "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "default")}
    else:
        stages = build_workspace(args.context, args.dtype, args.seed)
        rows = []
        for name, fn in stages:
            for _ in range(5):  # warmup: compile and cache effects leave the rows
                fn()
            sync()
            samples = []
            for _ in range(args.reps):
                sync()
                start = time.perf_counter()
                fn()
                sync()
                samples.append((time.perf_counter() - start) * 1000)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(args.profile_iters):
                    fn()
                sync()
            kernels = kernel_rows(prof.key_averages())
            gpu_total_us = sum(r["total_us"] for r in kernels) / args.profile_iters
            rows.append({"stage": name,
                         "p50_ms": statistics.median(samples),
                         "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
                         "gpu_time_us_per_iter": gpu_total_us,
                         "launches_per_iter": sum(r["count"] for r in kernels) / args.profile_iters,
                         "kernels": kernels[:12],
                         "kernel_table": format_kernel_table(kernels)})
            print(f"## {name}: p50 {rows[-1]['p50_ms']:.3f} ms, gpu {gpu_total_us:.0f} us/iter, "
                  f"{rows[-1]['launches_per_iter']:.0f} launches/iter", flush=True)
            print(rows[-1]["kernel_table"], flush=True)
        payload = {"mode": "warm", "context": args.context, "dtype": args.dtype,
                   "seed": args.seed, "gpu_arch": arch, "profile_iters": args.profile_iters,
                   "stages": rows}

    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
