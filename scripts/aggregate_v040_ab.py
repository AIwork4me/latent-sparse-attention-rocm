#!/usr/bin/env python3
"""Aggregate the v0.4.0 A/B receipts into the mission-B4 statistics.

Per (dtype, context, phase, stage): per-seed medians (median of the three
process-run p50s of that seed), pooled median over every process run, min/max
of the per-seed medians, and within-seed run-to-run spread (max/min of the rep
p50s). All labeled OBSERVED RANGES over 3 seeds x 3 process runs - never a
confidence interval. Speedups recompute from the arms' p50s in the same run
(paired by construction: interleaved samples); nothing is aggregated across
unmatched runs. CPU-only: reads the receipts, writes JSON + a markdown table.
"""

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    # rows[(dtype, ctx, phase, stage)][seed] = [rep p50s]
    rows = {}
    candidates = set()
    for path in args.receipts:
        receipt = json.loads(Path(path).read_text())
        cand = receipt["candidate"]
        candidates.add(cand)
        seed = receipt["seed"]
        for row in receipt["rows"]:
            if row.get("status") != "PASS":
                continue
            for stage, entry in row.get("stages", {}).items():
                key = (row["dtype"], row["context"], row["phase"], stage, cand)
                rows.setdefault(key, {}).setdefault(seed, []).append(
                    {"ref_p50": entry["reference"]["p50_ms"],
                     "cand_p50": entry["candidate"]["p50_ms"],
                     "speedup": entry["speedup_p50"],
                     "l_ref": entry["launches_reference"],
                     "l_cand": entry["launches_candidate"],
                     "run_id": receipt["run_id"]}
                )

    summary = []
    for key in sorted(rows):
        dtype, ctx, phase, stage, cand = key
        per_seed = rows[key]
        seed_medians_speedup = {}
        all_speedups, all_ref, all_cand = [], [], []
        for seed, reps in sorted(per_seed.items()):
            sp = [r["speedup"] for r in reps]
            seed_medians_speedup[seed] = statistics.median(sp)
            all_speedups.extend(sp)
            all_ref.extend(r["ref_p50"] for r in reps)
            all_cand.extend(r["cand_p50"] for r in reps)
        med = statistics.median
        summary.append({
            "dtype": dtype, "context": ctx, "phase": phase, "stage": stage, "candidate": cand,
            "seeds": len(per_seed),
            "process_runs": sum(len(v) for v in per_seed.values()),
            "speedup_per_seed_median": {str(s): round(v, 4) for s, v in seed_medians_speedup.items()},
            "speedup_pooled_median": round(med(all_speedups), 4),
            "speedup_range_of_seed_medians": [
                round(min(seed_medians_speedup.values()), 4), round(max(seed_medians_speedup.values()), 4)],
            "speedup_all_runs_min_max": [round(min(all_speedups), 4), round(max(all_speedups), 4)],
            "ref_p50_ms_pooled_median": round(med(all_ref), 4),
            "cand_p50_ms_pooled_median": round(med(all_cand), 4),
            "launches_ref": per_seed[min(per_seed)][0]["l_ref"],
            "launches_cand": per_seed[min(per_seed)][0]["l_cand"],
            "_label": "observed range (3 seeds x 3 process runs), not a confidence interval",
        })

    args.out.write_text(json.dumps(summary, indent=2))
    lines = [
        f"# v0.4.0 A/B aggregation ({len(args.receipts)} receipts, candidates: {sorted(candidates)})",
        "",
        "Speedup = interleaved reference p50 / candidate p50, same run. OBSERVED RANGES, not CIs.",
        "",
        "| dtype | ctx | phase | stage | cand | pooled med | seed medians (min-max) | all runs (min-max) | ref→cand p50 ms | launches |",
        "|---|---:|---|---|---|---:|---|---|---|---|",
    ]
    for s in summary:
        lo, hi = s["speedup_range_of_seed_medians"]
        alo, ahi = s["speedup_all_runs_min_max"]
        lines.append(
            f"| {s['dtype']} | {s['context']} | {s['phase']} | {s['stage']} | {s['candidate']} "
            f"| {s['speedup_pooled_median']:.3f}x | {lo:.3f}-{hi:.3f}x | {alo:.3f}-{ahi:.3f}x "
            f"| {s['ref_p50_ms_pooled_median']:.3f} → {s['cand_p50_ms_pooled_median']:.3f} "
            f"| {s['launches_ref']:.0f}→{s['launches_cand']:.0f} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
