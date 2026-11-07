#!/usr/bin/env python3
"""Aggregate the independent repetition receipts into one variance report.

Every receipt is validated first (run.validate_receipt + the gate manifest) - a
repetition whose receipt does not validate stops the aggregation rather than
silently feeding numbers into a headline table.
"""

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
    if len(receipts) < 2:
        raise SystemExit(f"need at least 2 validated repetition receipts in {out_dir}")
    return receipts


def aggregate(receipts: list[dict]) -> dict:
    """Per (context, dtype, phase, method): per-run p50s, speedups, median, spread."""
    runs = {}
    for index, receipt in enumerate(receipts, start=1):
        for case in receipt["comparisons"]:
            key = (case["context"], case["dtype"], case["phase"], case["method"])
            runs.setdefault(key, []).append(
                {
                    "run": receipt.get("run_id", f"run{index}"),
                    "baseline_ms": case["baseline_ms"],
                    "candidate_ms": case["candidate_ms"],
                    "speedup": case["speedup"],
                }
            )
    table = {}
    for key, rows in sorted(runs.items()):
        speedups = [r["speedup"] for r in rows]
        candidates = [r["candidate_ms"] for r in rows]
        table["/".join(str(k) for k in key)] = {
            "repetitions": rows,
            "speedup_median": statistics.median(speedups),
            "speedup_min": min(speedups),
            "speedup_max": max(speedups),
            "speedup_spread": max(speedups) - min(speedups),
            "candidate_p50_median_ms": statistics.median(candidates),
        }
    return table


def render(out_dir: Path, receipts: list[dict], table: dict) -> str:
    first = receipts[0]
    lines = [
        "# Independent matched benchmark repetitions",
        "",
        f"- repetitions: {len(receipts)} independent adapter processes, same seed/shape/dtype",
        f"- upstream SHA: `{first['upstream_sha']}` · repo HEAD: `{first.get('repo_head_sha', 'n/a')}`",
        f"- device: {first.get('gpu_name', 'n/a')} ({first['gpu_arch']}) · torch {first['torch']}",
        f"- gates: {'16/16 PASS in every repetition' if all(r.get('correctness_pass') for r in receipts) else 'NOT ALL PASS'}",
        f"- run ids: {', '.join(r.get('run_id', '?') for r in receipts)}",
        "",
        "| shape (ctx/dtype/phase/method) | per-run speedups | median | min-max | candidate p50 median (ms) |",
        "|---|---|---:|---|---:|",
    ]
    for shape, row in table.items():
        per_run = " / ".join(f"{r['speedup']:.2f}x" for r in row["repetitions"])
        lines.append(
            f"| {shape} | {per_run} | **{row['speedup_median']:.2f}x** "
            f"| {row['speedup_min']:.2f}-{row['speedup_max']:.2f} "
            f"| {row['candidate_p50_median_ms']:.3f} |"
        )
    lines += [
        "",
        "Per-repetition receipts: " + ", ".join(f"`{p.name}`" for p in sorted(out_dir.glob('receipt-*.json'))),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    receipts = load_validated(out_dir)
    table = aggregate(receipts)
    report = render(out_dir, receipts, table)
    (out_dir / "REPETITION_REPORT.md").write_text(report)
    (out_dir / "aggregated.json").write_text(json.dumps(table, indent=2))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
