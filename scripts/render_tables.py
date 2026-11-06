#!/usr/bin/env python3
"""Generate the headline benchmark tables from AUDITED receipts - nothing else.

Every table in README/FINAL_REPORT citing performance is emitted by this script
from receipt files that pass TWO independent gates: ``run.validate_receipt``
(kind, PASS, gate manifest, upstream SHA, device) and the independent
``evidence.auditor`` (F12), which recomputes every speedup from raw stage rows,
re-verifies the gate manifest and provenance, and rejects tampered, replayed or
scope-mislabeled evidence. An UNVERIFIED or tampered receipt aborts the render,
so no superseded or unverified number can reach a headline table.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa_rocm_lab.evidence.auditor import audit_files  # noqa: E402
from lsa_rocm_lab.run import validate_receipt  # noqa: E402

UPSTREAM_SHA = (ROOT / "third_party" / "upstream.sha").read_text().strip()

METHOD_LABEL = {"dense_sdpa": "Dense SDPA", "block_sparse": "Block-sparse",
                "lsa_flat": "D-LSA Flat", "lsa_tree": "D-LSA Tree"}


def load_receipts(paths: list[Path]) -> list[dict]:
    # Gate 1: the campaign runner's own validator (fail-closed receipt contract)
    for path in paths:
        validate_receipt(path, "lsa_rocm_kernel", UPSTREAM_SHA)  # raises on anything off
    # Gate 2: the independent auditor - recompute, never trust the adapter
    reports = audit_files(paths, expected_upstream_sha=UPSTREAM_SHA)
    for report in reports:
        print(report.summary())
        if report.verdict != "APPROVED":
            raise SystemExit(f"render aborted: {report.source} did not survive the independent audit")
    receipts = [json.loads(path.read_text()) for path in paths]
    for path, receipt in zip(paths, receipts):
        print(f"audited receipt: {path} (run {receipt.get('run_id')})")
    return receipts


def _ms(receipt: dict, method: str, phase: str, stage: str | None, context: int) -> float | None:
    """p50 of a stage row; ``stage=None`` matches the dense rows (no stage key)."""
    for row in receipt["stages"]:
        if (row.get("method") == method and row.get("phase") == phase
                and row.get("stage", None) == stage and row.get("context") == context):
            return row["p50_ms"]
    return None


def phase_table(receipt: dict, phase: str) -> str:
    """Per-context p50 ms of the step (select+read) with speedups vs the dense baseline."""
    header = f"### {phase.capitalize()} — select+read p50 ms (speedup vs dense)\n\n"
    contexts = sorted({c["context"] for c in receipt["comparisons"] if c["phase"] == phase})
    if not contexts:
        return header + "_no rows_\n"
    lines = ["| ctx | dense SDPA | block-sparse | D-LSA Flat | D-LSA Tree |",
             "|---:|---:|---:|---:|---:|"]
    for context in contexts:
        dense = _ms(receipt, "dense_sdpa", phase, None, context)
        cells = []
        for method in ("block_sparse", "lsa_flat", "lsa_tree"):
            case = next((c for c in receipt["comparisons"]
                         if c["phase"] == phase and c["method"] == method and c["context"] == context), None)
            if case is None:
                cells.append("—")
            else:
                cells.append(f"{case['candidate_ms']:.3f} ({case['speedup']:.2f}×)")
        dense_cell = f"{dense:.3f}" if dense is not None else "—"
        lines.append(f"| {context} | {dense_cell} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return header + "\n".join(lines) + "\n"


def stage_breakdown(receipt: dict, context: int) -> str:
    lines = [f"### Stage breakdown @ {context} (p50 ms)\n",
             "| method | phase | index_build | hierarchy_build | select | sparse_read | select+read |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method in ("dense_sdpa", "block_sparse", "lsa_flat", "lsa_tree"):
        for phase in ("prefill", "decode"):
            row = [METHOD_LABEL[method], phase]
            empty = True
            for stage in ("index_build", "hierarchy_build", "select", "sparse_read", "select+read"):
                value = _ms(receipt, method, phase, stage, context)
                row.append(f"{value:.3f}" if value is not None else "—")
                empty = empty and value is None
            if not empty:
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def provenance(receipt: dict, path: Path) -> str:
    env = receipt.get("env", {})
    return (
        f"- receipt: `{path}` · run id `{receipt.get('run_id', 'n/a')}` · status {receipt['status']}\n"
        f"- gates: {len(receipt.get('gates', []))}/{len(receipt.get('gates', []))} PASS "
        f"(manifest-validated) · correctness_pass: {receipt['correctness_pass']}\n"
        f"- repo HEAD: `{receipt.get('repo_head_sha', 'n/a')}` · upstream: `{receipt['upstream_sha'][:12]}`\n"
        f"- device: {receipt.get('gpu_name', 'n/a')} ({receipt['gpu_arch']}) · torch {receipt['torch']}\n"
        f"- dtype {env.get('dtype')} · seed {env.get('seed')} · warmup {env.get('warmup')} / reps {env.get('reps')}\n"
        f"- command: `{' '.join(receipt.get('command', []))}`\n"
    )


def render(receipts: list[dict], paths: list[Path], repetitions: Path | None,
           seed_study: tuple[list[dict], dict] | None = None) -> str:
    out = ["<!-- GENERATED by scripts/render_tables.py from independently AUDITED receipts - do not edit by hand. -->",
           "# Benchmark tables (independently audited receipts only)", ""]
    for receipt, path in zip(receipts, paths):
        env = receipt.get("env", {})
        contexts = sorted({c["context"] for c in receipt["comparisons"]})
        out += [f"## {env.get('dtype')} matrix — {receipt.get('gpu_name', '')}", "",
                provenance(receipt, path), "",
                phase_table(receipt, "decode"), "",
                phase_table(receipt, "prefill"), "",
                stage_breakdown(receipt, contexts[-1] if contexts else 32768), ""]
    if repetitions and (repetitions / "REPETITION_REPORT.md").is_file():
        out += ["## Independent repetitions (strategic shapes)", "",
                (repetitions / "REPETITION_REPORT.md").read_text(), ""]
    if seed_study is not None:
        study_receipts, study_report = seed_study
        out += [seed_study_section(study_receipts, study_report), ""]
    return "\n".join(out)


def seed_study_section(receipts: list[dict], report: dict) -> str:
    """Multi-seed variance bands (v0.3.0/T1) - the honest-publication section.

    Per-seed medians stay separate from the pooled median; the headline number
    for any claim is the cross-seed band, never one seed's figure (the v0.2.1
    2.1x flat-decode figure came from a favorable seed; see evidence/
    seed-study-20261106T151843Z). An INSUFFICIENT study is published as such.
    """
    seeds = sorted({r.get("env", {}).get("seed") for r in receipts})
    heads = [f"## Multi-seed variance bands (v0.3.0/T1) — {report.get('receipts')} receipts", "",
             f"- seeds: {', '.join(str(s) for s in seeds)} · min seeds {report['min_seeds']} · "
             f"min runs/seed {report['min_runs_per_seed']}",
             f"- runs: {', '.join(sorted(r.get('run_id', '?') for r in receipts))}",
             f"- coverage status: **{report['status']}**", "",
             "A single-seed speedup is NOT a headline claim: cite the cross-seed band "
             "(per-seed median speedups, min–max) plus the pooled median.", "",
             "| context/dtype/phase/method | per-seed medians | cross-seed band | pooled |",
             "|---|---|---|---|"]
    for name, key in report["keys"].items():
        medians = ", ".join(f"s{s}={v['median_speedup']:.2f}" for s, v in key["per_seed"].items())
        band = key["cross_seed"]
        flag = "" if key["coverage_ok"] else " ⚠️ under-covered"
        heads.append(f"| {name}{flag} | {medians} | {band['min_seed_median']:.2f}–"
                     f"{band['max_seed_median']:.2f} (med {band['median_of_seed_medians']:.2f}) "
                     f"| {key['pooled_median_speedup']:.2f} |")
    return "\n".join(heads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True, type=Path)
    parser.add_argument("--repetitions", type=Path, default=None)
    parser.add_argument("--seed-study-dir", type=Path, default=None,
                        help="directory of seed-study receipts (receipt-*.json) to "
                             "validate, audit and publish as variance bands")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "BENCHMARKS.md")
    args = parser.parse_args()
    receipts = load_receipts(args.receipt)
    seed_study = None
    if args.seed_study_dir:
        sys.path.insert(0, str(ROOT / "scripts"))
        from aggregate_seed_study import aggregate, load_validated
        study_receipts = load_validated(args.seed_study_dir)
        reports = audit_files(list(args.seed_study_dir.glob("receipt-*.json")),
                              expected_upstream_sha=UPSTREAM_SHA)
        for study_report in reports:
            print(study_report.summary())
            if study_report.verdict != "APPROVED":
                raise SystemExit(f"render aborted: {study_report.source} failed the seed-study audit")
        seed_study = (study_receipts,
                      aggregate(study_receipts, min_seeds=3, min_runs_per_seed=3))
    args.out.write_text(render(receipts, args.receipt, args.repetitions, seed_study))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
