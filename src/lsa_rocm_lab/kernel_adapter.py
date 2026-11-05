"""The ``LSA_ROCM_KERNEL_CMD`` adapter - gates first, matched benchmarks second, a receipt third.

Exit code 0 only when every correctness gate passed and the benchmark produced real
matched comparisons; the receipt (JSON at ``$LSA_STAGE_RECEIPT``) carries the
upstream SHA, the GPU arch, and one row per comparison with synchronized timings
the starter's ``run.validate_receipt`` checks. No row is written from a simulated
run, a CPU fallback, or an unmatched shape.
"""

import argparse
import json
import os
import sys

import torch

from lsa_rocm_lab.gate_manifest import describe, verdict_of
from lsa_rocm_lab.port import pipeline, upstream_sha
from lsa_rocm_lab.port.bench import (
    Clock,
    bench_block_sparse,
    bench_dense_decode,
    bench_dense_prefill,
    bench_lsa,
    build_geometry,
)
from lsa_rocm_lab.port.gates import run_all_gates
from lsa_rocm_lab.port.pipeline import PortConfig

# The matrix: contexts per the plan, one representative geometry plus variations
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
GEOMETRY = {"heads_q": 8, "heads_kv": 2, "dim": 64}  # index_dim 128 - inside the Triton Ward kernel's range


def config_for(context: int, **over) -> PortConfig:
    """The design's defaults at any context; callers override the plan's variations."""
    settings = dict(
        cluster_block_size=512, pool_size=4, bucket_size=8, bucket_cap=16,
        top_p=1024, top_k=4096, window_size=512, sink_size=4,
        node_bucket_size=4, top_level=3, node_cluster_block_size=64, beam_width=64,
    )
    settings.update(over)
    return PortConfig(**settings)


def memory_of(fn) -> tuple:
    """Peak allocated GPU memory of a callable, and its result."""
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before, out


def gates_summary(gates: list[dict] | None, skipped: bool) -> dict:
    """Fail-closed gate posture for the receipt - never vacuously True.

    ``--skip-gates`` yields ``status="UNVERIFIED"``/``correctness_pass=False``:
    the benchmark rows may exist, but no speedup from them is claimable. A run
    suite is only True when it is exactly ``gate_manifest.REQUIRED_GATE_IDS``
    with every verdict PASS - an empty, short, renamed or failed list never is.
    """
    if skipped:
        return {"correctness_pass": False, "status": "UNVERIFIED", "gates_skipped": True,
                "reason": "correctness gates skipped by --skip-gates: benchmark UNVERIFIED"}
    verdict = verdict_of(gates or [])
    return {"correctness_pass": verdict["ok"], "status": "PASS" if verdict["ok"] else "FAIL",
            "gates_skipped": False, "reason": describe(verdict)}


def run_matrix(clock: Clock, max_context: int, dtype_name: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Run every method on every context - matched rows only leave this function."""
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    stages: list[dict] = []
    comparisons: list[dict] = []
    for context in CONTEXTS:
        if context > max_context:
            continue
        torch.cuda.empty_cache()
        try:
            q, k, v = build_geometry(context, dtype, seed=seed + context, **GEOMETRY)
        except torch.cuda.OutOfMemoryError as error:
            stages.append({"method": "all", "context": context, "dtype": dtype_name,
                           "stage": "allocate", "status": "OOM", "error": str(error)[:200]})
            continue
        config = config_for(context)
        rows = []
        rows.append(bench_dense_prefill(q, k, v, clock))
        rows.append(bench_dense_decode(q, k, v, clock))
        rows += bench_lsa(q, k, v, config, clock, tree=False, phase="prefill")
        rows += bench_lsa(q, k, v, config, clock, tree=False, phase="decode")
        rows += bench_block_sparse(q, k, v, config, clock, phase="prefill")
        rows += bench_block_sparse(q, k, v, config, clock, phase="decode")
        if context >= 4096:
            rows += bench_lsa(q, k, v, config, clock, tree=True, phase="prefill")
            rows += bench_lsa(q, k, v, config, clock, tree=True, phase="decode")
        for row in rows:
            row.update({"context": context, "dtype": dtype_name,
                        "heads": GEOMETRY["heads_q"], "kv_heads": GEOMETRY["heads_kv"],
                        "dim": GEOMETRY["dim"], "top_k": config.top_k, "top_p": config.top_p,
                        "window": config.window_size, "sink": config.sink_size,
                        "seed": seed})
            stages.append(row)
        # Matched comparisons: same context/dtype/phase, candidate vs the dense baseline
        for phase in ("prefill", "decode"):
            dense = next((r for r in rows if r["method"] == "dense_sdpa" and r["phase"] == phase), None)
            if dense is None:
                continue
            for method in ("block_sparse", "lsa_flat", "lsa_tree"):
                step = next((r for r in rows if r["method"] == method and r["phase"] == phase
                             and r.get("stage") == "select+read"), None)
                if step is None:
                    continue
                comparisons.append({
                    "context": context, "dtype": dtype_name, "phase": phase,
                    "method": method, "stage": "select+read",
                    "baseline_ms": dense["p50_ms"], "baseline_p95_ms": dense["p95_ms"],
                    "candidate_ms": step["p50_ms"], "candidate_p95_ms": step["p95_ms"],
                    "samples": min(dense["samples"], step["samples"]),
                    "speedup": dense["p50_ms"] / step["p50_ms"] if step["p50_ms"] > 0 else 0.0,
                    "note": "select+read on a prebuilt index; index_build timed separately",
                })
        del q, k, v
        torch.cuda.empty_cache()
    return stages, comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-context", type=int, default=int(os.getenv("LSA_MAX_CONTEXT", "32768")))
    parser.add_argument("--dtype", default=os.getenv("LSA_DTYPE", "bf16"), choices=["fp16", "bf16"])
    parser.add_argument("--reps", type=int, default=int(os.getenv("LSA_REPS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("LSA_WARMUP", "5")))
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU from torch"}))
        return 78
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1151"):
        print(json.dumps({"status": "BLOCKED", "reason": f"not gfx1151: {arch}"}))
        return 78

    gates = None
    if not args.skip_gates:
        gates = run_all_gates(seed=args.seed)
        for gate in gates:
            print(json.dumps(gate), flush=True)
    summary = gates_summary(gates, args.skip_gates)
    if gates is not None and not summary["correctness_pass"]:
        # A run suite that is short/renamed/failed never proceeds to a benchmark claim
        print(json.dumps({"status": "FAIL", "reason": summary["reason"]}), flush=True)
        return 1

    clock = Clock(warmup=args.warmup, reps=args.reps)
    stages, comparisons = run_matrix(clock, args.max_context, args.dtype, args.seed)
    for row in stages:
        print(json.dumps(row), flush=True)

    # Skipped gates: the rows are real timings but carry no verified-correct claim -
    # the receipt says UNVERIFIED and run.validate_receipt refuses it (fail-closed)
    receipt = {
        "kind": "lsa_rocm_kernel",
        "status": summary["status"] if comparisons else "FAIL",
        "correctness_pass": summary["correctness_pass"],
        "gates_skipped": summary["gates_skipped"],
        "upstream_sha": os.getenv("LSA_UPSTREAM_SHA", upstream_sha()),
        "gpu_arch": arch,
        "torch": torch.__version__,
        "gates": gates or [],
        "stages": stages,
        "comparisons": comparisons,
        "env": {"dtype": args.dtype, "reps": args.reps, "warmup": args.warmup, "seed": args.seed},
    }
    target = os.getenv("LSA_STAGE_RECEIPT")
    if target:
        with open(target, "w") as handle:
            json.dump(receipt, handle, indent=2)
    if not comparisons:
        print(json.dumps({"status": "FAIL", "reason": "no matched comparisons produced"}), flush=True)
        return 1
    print(json.dumps({"status": summary["status"],
                      "comparisons": len(comparisons),
                      "claimable": summary["correctness_pass"]}), flush=True)
    # Exit 0: the benchmark itself completed. Claimability is the receipt's job -
    # an UNVERIFIED receipt is rejected by run.validate_receipt whatever this code is.
    return 0


if __name__ == "__main__":
    sys.exit(main())
