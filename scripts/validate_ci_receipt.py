#!/usr/bin/env python3
"""Validate a kernel receipt exactly as the campaign runner would (CI gate).

Usage: python scripts/validate_ci_receipt.py <receipt.json> [upstream.sha file]
Exits 0 only for a PASS receipt with the full gate manifest satisfied.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsa_rocm_lab.run import validate_receipt  # noqa: E402


def main() -> int:
    receipt = Path(sys.argv[1])
    sha_file = Path(sys.argv[2] if len(sys.argv) > 2 else "third_party/upstream.sha")
    sha = sha_file.read_text().strip()
    validate_receipt(receipt, "lsa_rocm_kernel", sha)
    print(f"receipt validated: {receipt} PASS with the full gate manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
