#!/usr/bin/env python3
"""Aggregate the multi-seed study receipts (v0.3.0/T1) into per-seed variance bands.

Every receipt is validated first (run.validate_receipt + pinned upstream SHA) -
an invalid receipt stops aggregation rather than feeding numbers into a band.
Grouping is by (context, dtype, phase, method) then by ``env.seed``: per-seed
medians stay separate from the pooled median, and the cross-seed band is built
from per-seed medians (never from the pooled runs, which would let a lucky seed
hide an unlucky one). Coverage below --min-seeds distinct seeds or
--min-runs-per-seed fresh processes per seed for ANY key makes the whole study
INSUFFICIENT (exit 1): T2-T4 gain claims are gated on this band, so a thin
study must fail loudly, not print a narrower band.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa_rocm_lab.run import validate_receipt  # noqa: E402

UPSTREAM_SHA = (Path(__file__).resolve().parents[1] / "third_party" / "upstream.sha").read_text().strip()


def load_validated(out_dir: Path) -> list[dict]:
    receipts = []
    for path in sorted(out_dir.glob("receipt-*.json")):
        validate_receipt(path, "lsa_rocm_kernel", UPSTREAM_SHA)
        receipts.append(json.loads(path.read_text()))
    if not receipts:
        raise SystemExit(f"no receipt-*.json found in {out_dir}")
    return receipts


def _key_of(case: dict) -> tuple:
    return (case["context"], case["dtype"], case["phase"], case["method"])


def coverage_of(receipts: list[dict], key: tuple) -> tuple[list, dict]:
    """Distinct seeds and fresh-process runs per seed for one comparison key.

    ``key`` matches the trailing fields of the full comparison key, so
    ("lsa_flat",) matches the method at any context/phase.
    """
    runs: dict[int, int] = {}
    for receipt in receipts:
        seed = int(receipt.get("env", {}).get("seed", -1))
        hit = any(_key_of(case)[-len(key):] == key for case in receipt.get("comparisons", []))
        if hit:
            runs[seed] = runs.get(seed, 0) + 1
    return sorted(runs), runs


def aggregate(receipts: list[dict], min_seeds: int, min_runs_per_seed: int) -> dict:
    """Per (context, dtype, phase, method): per-seed rows, cross-seed band, pooled median."""
    runs: dict[tuple, dict[int, list]] = {}
    for receipt in receipts:
        seed = int(receipt.get("env", {}).get("seed", -1))
        for case in receipt.get("comparisons", []):
            runs.setdefault(_key_of(case), {}).setdefault(seed, []).append(
                {"run": receipt.get("run_id", "?"),
                 "baseline_ms": case["baseline_ms"], "candidate_ms": case["candidate_ms"],
                 "speedup": case["speedup"]}
            )
    keys: dict[str, dict] = {}
    all_covered = True
    for key, by_seed in sorted(runs.items()):
        per_seed = {}
        for seed, rows in sorted(by_seed.items()):
            speedups = [r["speedup"] for r in rows]
            per_seed[str(seed)] = {
                "runs": len(rows), "run_ids": [r["run"] for r in rows],
                "median_speedup": statistics.median(speedups),
                "spread": max(speedups) - min(speedups),
                "median_candidate_ms": statistics.median(r["candidate_ms"] for r in rows),
                "rows": rows,
            }
        seed_medians = [v["median_speedup"] for v in per_seed.values()]
        coverage_ok = (len(per_seed) >= min_seeds
                       and all(v["runs"] >= min_runs_per_seed for v in per_seed.values()))
        all_covered = all_covered and coverage_ok
        pooled = [r["speedup"] for rows in by_seed.values() for r in rows]
        keys["/".join(str(k) for k in key)] = {
            "coverage_ok": coverage_ok,
            "seeds": {s: v["runs"] for s, v in per_seed.items()},
            "per_seed": per_seed,
            "cross_seed": {
                "min_seed_median": min(seed_medians),
                "max_seed_median": max(seed_medians),
                "median_of_seed_medians": statistics.median(seed_medians),
                "band_ratio": (max(seed_medians) / min(seed_medians)) if min(seed_medians) > 0 else None,
            },
            "pooled_median_speedup": statistics.median(pooled),
        }
    return {"status": "OK" if all_covered else "INSUFFICIENT",
            "min_seeds": min_seeds, "min_runs_per_seed": min_runs_per_seed,
            "receipts": len(receipts), "keys": keys}


def markdown(report: dict) -> str:
    lines = ["| context/dtype/phase/method | seeds | per-seed medians | cross-seed band | pooled |",
             "|---|---|---|---|---|"]
    for name, key in report["keys"].items():
        medians = ", ".join(f"s{s}={v['median_speedup']:.2f}" for s, v in key["per_seed"].items())
        band = key["cross_seed"]
        lines.append(
            f"| {name} | {len(key['seeds'])}{'' if key['coverage_ok'] else ' (UNDER-COVERED)'} "
            f"| {medians} | {band['min_seed_median']:.2f}–{band['max_seed_median']:.2f} "
            f"(med {band['median_of_seed_medians']:.2f}) | {key['pooled_median_speedup']:.2f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-runs-per-seed", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = aggregate(load_validated(args.out_dir), args.min_seeds, args.min_runs_per_seed)
    target = args.output or (args.out_dir / "seed-study.json")
    target.write_text(json.dumps(report, indent=2))
    print(f"seed study: {report['status']} ({report['receipts']} receipts) -> {target}")
    print(markdown(report))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
