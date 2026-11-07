#!/usr/bin/env python3
"""Independent receipt auditor CLI (F12): recompute, cross-check, fail closed.

Usage:
  python scripts/audit_receipts.py [--upstream-sha-file F] [--expect-repo-sha SHA]
                                   [--report OUT.json] RECEIPT [RECEIPT ...]

Exit 0 only when EVERY receipt is APPROVED with zero findings. Run IDs are
deduplicated across the argument set, so a replayed receipt fails the audit.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsa_rocm_lab.evidence.auditor import audit_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--upstream-sha-file", type=Path, default=ROOT / "third_party" / "upstream.sha",
                        help="pinned upstream SHA the receipts must carry")
    parser.add_argument("--expect-repo-sha", default=None,
                        help="repo HEAD SHA the receipts must be tied to (default: no repo pin)")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the audit report JSON here (always written on failure)")
    args = parser.parse_args()

    upstream_sha = None
    if args.upstream_sha_file and args.upstream_sha_file.is_file():
        upstream_sha = args.upstream_sha_file.read_text().strip()

    reports = audit_files(args.receipts, expected_upstream_sha=upstream_sha,
                          expected_repo_sha=args.expect_repo_sha)
    for report in reports:
        print(report.summary())

    payload = {"schema": "kernel-receipt-audit/1",
               "verdict": "APPROVED" if all(r.verdict == "APPROVED" for r in reports) else "REJECTED",
               "receipts": [r.to_dict() for r in reports]}
    if args.report:
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"audit report written: {args.report}")

    if payload["verdict"] != "APPROVED":
        print("receipt audit FAILED: no claim may be built on these receipts")
        return 1
    total = sum(len(r.recomputed.get("comparisons") or []) for r in reports)
    print(f"receipt audit passed: {len(reports)} receipt(s), {total} recomputed comparison(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
