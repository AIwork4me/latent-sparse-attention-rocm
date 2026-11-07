#!/usr/bin/env python3
"""A/B the tiled sparse read (v2) against v1 on identical inputs (v0.3.0/T3).

Matched, in-process, correctness-checked: for every context, the SAME q/k/v and
the SAME production read lists (flat_select on the built index) feed both
kernels; v2 outputs are verified against v1 (dtype tolerance) before any
timing counts, for every (tile_rows, chunk) config. A config that fails the
equivalence check is reported FAILED and excluded from the win table - a
wrong-output speedup is not a speedup.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402
from lsa_rocm_lab.port.read_kernel import sparse_read  # noqa: E402
from lsa_rocm_lab.port.read_kernel_tiled import sparse_read_tiled  # noqa: E402

TOLERANCE = {"fp16": 5e-3, "bf16": 2e-2}
CONFIGS = [(1, 64), (2, 64), (2, 128), (4, 32), (4, 64), (4, 128), (8, 16), (8, 32), (8, 64), (16, 16), (16, 32)]


def timed(fn, warmup=5, reps=20) -> dict:
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
    return {"p50_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tol = TOLERANCE[args.dtype]
    report = {"dtype": args.dtype, "seed": args.seed, "rows": []}
    for context in [int(c) for c in args.contexts.split(",")]:
        q, k, v = build_geometry(context, dtype, seed=args.seed + context, **GEOMETRY)
        config = config_for(context)
        idx_q, idx_k = make_index_space(q, k)
        buckets = pipeline.build_buckets_gpu(idx_k, config)
        positions = torch.arange(context, device=q.device)
        lists, lengths = pipeline.flat_select(idx_q, positions, buckets, config)
        scale = q.shape[-1] ** -0.5
        for phase, q_arg, l_arg, n_arg in [
            ("prefill", q, lists, lengths),
            ("decode", q[:, -1:], lists[-1:], lengths[-1:]),
        ]:
            l_arg, n_arg = l_arg.unsqueeze(0), n_arg.unsqueeze(0)
            base = sparse_read(q_arg, k, v, l_arg, n_arg, scale)
            base_t = timed(lambda: sparse_read(q_arg, k, v, l_arg, n_arg, scale))
            row = {"context": context, "phase": phase, "v1_p50_ms": base_t["p50_ms"],
                   "v1_p95_ms": base_t["p95_ms"], "configs": []}
            print(f"== {context} {phase}: v1 p50 {base_t['p50_ms']:.3f} ms", flush=True)
            for tile, chunk in CONFIGS:
                try:
                    cand = sparse_read_tiled(q_arg, k, v, l_arg, n_arg, scale,
                                             tile_rows=tile, chunk=chunk)
                    err = float((cand.float() - base.float()).abs().max())
                    ok = bool(torch.allclose(cand.float(), base.float(), atol=tol, rtol=tol))
                    if not ok:
                        row["configs"].append({"tile": tile, "chunk": chunk, "status": "WRONG_OUTPUT",
                                               "max_abs_err": err})
                        print(f"   TM{tile} CN{chunk}: WRONG_OUTPUT max_err {err:.4f}", flush=True)
                        continue
                    t = timed(lambda: sparse_read_tiled(q_arg, k, v, l_arg, n_arg, scale,
                                                         tile_rows=tile, chunk=chunk))
                    row["configs"].append({"tile": tile, "chunk": chunk, "status": "PASS",
                                           "max_abs_err": err, "p50_ms": t["p50_ms"],
                                           "p95_ms": t["p95_ms"],
                                           "speedup_vs_v1": base_t["p50_ms"] / t["p50_ms"]})
                    print(f"   TM{tile} CN{chunk}: p50 {t['p50_ms']:.3f} ms "
                          f"({base_t['p50_ms'] / t['p50_ms']:.2f}x v1, err {err:.4f})", flush=True)
                except Exception as exc:  # a config that will not compile/run is a finding, not a crash
                    row["configs"].append({"tile": tile, "chunk": chunk, "status": "ERROR",
                                           "error": str(exc)[:200]})
                    print(f"   TM{tile} CN{chunk}: ERROR {str(exc)[:120]}", flush=True)
            report["rows"].append(row)
            del base, l_arg, n_arg
            torch.cuda.empty_cache()
        del lists, lengths
        del q, k, v, idx_q, idx_k, buckets
        torch.cuda.empty_cache()

    best = {}
    for row in report["rows"]:
        passing = [c for c in row["configs"] if c.get("status") == "PASS"]
        if passing:
            top = max(passing, key=lambda c: c["speedup_vs_v1"])
            best[f"{row['context']}/{row['phase']}"] = {
                "config": [top["tile"], top["chunk"]], "p50_ms": top["p50_ms"],
                "speedup_vs_v1": top["speedup_vs_v1"]}
    report["best"] = best
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
