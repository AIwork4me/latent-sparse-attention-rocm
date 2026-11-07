#!/usr/bin/env python3
"""A/B the experimental page-batched tree walk against the per-page walk (v0.3.0/T4).

Matched, in-process, on identical production inputs: the same index, buckets and
hierarchy feed both implementations; set-equality of the picks is asserted
before any timing counts (gate C's bar), so a semantics drift can never pose
as a performance number. Timing uses the receipts' methodology (synchronized
warmup then reps) so the rows are directly comparable with receipt stage rows.
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
from lsa_rocm_lab.port._tree import tree_select_per_page  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402


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
    parser.add_argument("--contexts", default="8192,32768")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78

    report = {"seed": args.seed, "rows": []}
    import os
    saved_impl = os.environ.get("LSA_TREE_IMPL")
    for context in [int(c) for c in args.contexts.split(",")]:
        q, k, v = build_geometry(context, torch.bfloat16, seed=args.seed + context, **GEOMETRY)
        config = config_for(context)
        idx_q, idx_k = make_index_space(q, k)
        buckets = pipeline.build_buckets_gpu(idx_k, config)
        hierarchy = pipeline.build_hierarchy_gpu(idx_k, buckets, config)
        positions = torch.arange(context, device=q.device)
        for phase, iq, pos in [
            ("prefill", idx_q, positions),
            ("decode", idx_q[-1:], positions[-1:]),
        ]:
            per = lambda: tree_select_per_page(iq, pos, buckets, hierarchy, config)

            def bat():
                os.environ["LSA_TREE_IMPL"] = "batched"
                try:
                    return pipeline.tree_select(iq, pos, buckets, hierarchy, config)
                finally:
                    if saved_impl is None:
                        os.environ.pop("LSA_TREE_IMPL", None)
                    else:
                        os.environ["LSA_TREE_IMPL"] = saved_impl

            a_lists, _ = per()
            b_lists, _ = bat()
            sets_a = [{int(x) for x in row.tolist() if x >= 0} for row in a_lists]
            sets_b = [{int(x) for x in row.tolist() if x >= 0} for row in b_lists]
            set_equal = sets_a == sets_b
            per_t = timed(per)
            bat_t = timed(bat)
            row = {"context": context, "phase": phase, "set_equal": set_equal,
                   "per_page_p50_ms": per_t["p50_ms"], "per_page_p95_ms": per_t["p95_ms"],
                   "batched_p50_ms": bat_t["p50_ms"], "batched_p95_ms": bat_t["p95_ms"],
                   "ratio_batched_over_per_page": bat_t["p50_ms"] / per_t["p50_ms"]}
            report["rows"].append(row)
            print(f"== {context} {phase}: per-page {per_t['p50_ms']:.2f} ms, "
                  f"batched {bat_t['p50_ms']:.2f} ms ({row['ratio_batched_over_per_page']:.2f}x), "
                  f"set_equal={set_equal}", flush=True)
        del q, k, v, idx_q, idx_k, buckets, hierarchy
        torch.cuda.empty_cache()

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
