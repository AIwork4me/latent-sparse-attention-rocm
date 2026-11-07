"""Upstream kernel-test triage - classify known failures, fail on anything new.

Upstream's suite has tests that cannot pass on gfx1151 by construction (their
Ward/pairing kernels default to the tf32x3 dot precision, which HIP rejects at
Triton compilation) and tests whose NVIDIA-calibrated exactness/tolerance
expectations miss marginally on RDNA (the shape-scoring path this port does not
use). Those are *classified*: each carries a reproducible reason in
``evidence/upstream-test-classification.json`` and stays fully visible in the
log. The triage comparator fails (exit 1) on any UNCLASSIFIED failure - a real
regression can never hide behind the known list.
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "evidence" / "upstream-test-classification.json"
SUITE = [
    "tests/unit/core/test_ward_kernel.py",
    "tests/unit/core/test_ward.py",
    "tests/unit/core/test_keys.py",
    "tests/unit/core/test_top_k.py",
    "tests/unit/core/test_beam.py",
    "tests/unit/core/test_shape_kernels.py",
    "tests/unit/core/test_paged.py",
    "tests/unit/core/test_reference.py",
    "tests/unit/core/test_attention.py",
    "tests/unit/core/test_index_space.py",
    "tests/unit/core/test_hierarchy.py",
]
# Excluded at the runner level (import the NVIDIA CuTe DSL front doors); see
# scripts/run_upstream_kernel_tests.sh for the reasons.


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """The committed classification manifest: node id -> class, reason, evidence."""
    data = json.loads(Path(path).read_text())
    return data["tests"]


def triage(failed_ids: list[str], manifest: dict) -> dict:
    """Split failures into classified (with reason) and unclassified (regressions)."""
    classified = {i: manifest[i] for i in failed_ids if i in manifest}
    unclassified = [i for i in failed_ids if i not in manifest]
    return {"classified": classified, "unclassified": sorted(unclassified)}


def render(failed_ids: list[str], passed: int, result: dict) -> str:
    """The visible classification table - known failures stay in the open."""
    lines = [
        f"upstream kernel tests: {passed} passed, {len(failed_ids)} failed",
        "",
    ]
    for node_id, entry in sorted(result["classified"].items()):
        reason = entry["reason"]
        lines.append(f"  [classified: {entry['classification']}] {node_id}")
        lines.append(f"      {reason}")
    for node_id in result["unclassified"]:
        lines.append(f"  [UNCLASSIFIED - REGRESSION?] {node_id}")
    lines.append("")
    if result["unclassified"]:
        lines.append("TRIAGE: FAIL - unclassified upstream test failures above")
    else:
        lines.append(
            "TRIAGE: PASS - every failure is classified with a reproducible reason; "
            "any new failure will fail this step"
        )
    return "\n".join(lines)


def node_ids_from_junit(path: Path) -> tuple[list[str], int]:
    """(failed node ids, passed count) from a pytest --junitxml report."""
    root = ElementTree.parse(path).getroot()
    failed, passed = [], 0
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        file_path = classname.replace(".", "/") + ".py"
        node_id = f"{file_path}::{name}"
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(node_id)
        else:
            passed += 1
    return failed, passed


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default=str(ROOT / "third_party" / "latent-sparse-attention"))
    parser.add_argument("--junit", default="/tmp/lsa-upstream-junit.xml")
    args = parser.parse_args()
    upstream = Path(args.upstream)

    cmd = [sys.executable, "-m", "pytest", *SUITE, f"--junitxml={args.junit}", "-q", "--tb=short"]
    completed = subprocess.run(cmd, cwd=upstream, env={
        **os.environ, "PYTHONPATH": f"{upstream / 'src'}",
    })
    failed, passed = node_ids_from_junit(Path(args.junit))
    manifest = load_manifest()
    result = triage(failed, manifest)
    print(render(failed, passed, result))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "checked_at_utc": stamp,
        "upstream_sha": (upstream / ".git").exists() and subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "UNAVAILABLE",
        "passed": passed,
        "failed": len(failed),
        "failed_ids": sorted(failed),
        "classified": sorted(result["classified"]),
        "unclassified": result["unclassified"],
        "exit": 1 if result["unclassified"] else 0,
    }
    out = ROOT / "results" / f"upstream-triage-{stamp}.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"record: {out}")
    # The pytest exit code is subsumed: only an UNCLASSIFIED failure fails here.
    return record["exit"] if completed.returncode in (0, 1) else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
