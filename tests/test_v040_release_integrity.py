"""Release-time v0.4.0 receipt integrity checks (hosted CPU CI; no ROCm required).

Prevents a published SHA256 manifest from silently diverging from the exact
committed receipt bytes, including archived dirty-tree evidence. This is a
file-integrity gate, not a substitute for running physical gfx1151 kernels.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v0.4.0"
MANIFEST = EVIDENCE / "SHA256SUMS"


def manifest_records():
    assert MANIFEST.is_file(), "v0.4.0 release SHA256 manifest missing"
    rows = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", raw)
        assert match is not None, f"malformed SHA256SUMS entry: {raw!r}"
        digest, rel = match.groups()
        assert not rel.startswith("/") and ".." not in Path(rel).parts, (
            f"unsafe evidence path: {rel}"
        )
        rows.append((digest, rel))
    assert rows, "empty v0.4.0 manifest"
    assert len({rel for _, rel in rows}) == len(rows), "duplicate evidence path"
    return rows


def test_v040_manifest_exact_coverage():
    records = manifest_records()
    actual = {
        str(p.relative_to(EVIDENCE))
        for p in EVIDENCE.rglob("*")
        if p.is_file() and p != MANIFEST
    }
    expected = {rel for _, rel in records}
    assert expected == actual, (
        f"missing in manifest: {sorted(actual - expected)}; "
        f"missing on disk: {sorted(expected - actual)}"
    )
    assert len(records) == 25, "release candidate audit expected 25 evidence artifacts"


def test_v040_sha256_manifest_matches_every_committed_byte():
    for expected, rel in manifest_records():
        actual = hashlib.sha256((EVIDENCE / rel).read_bytes()).hexdigest()
        assert actual == expected, f"SHA256 mismatch: {rel}"


def test_v040_clean_tree_memory_receipt_is_canonical():
    receipt = json.loads((EVIDENCE / "ab" / "memory-probe.json").read_text())
    assert receipt["production_code_sha"] == (
        "45a0ed12201b3a301712815dab32c0dac9c8fee4"
    )
    assert receipt["production_src_dirty"] is False
    assert receipt["production_tree_dirty"] is False
    for arm in ("reference", "fused", "fused_unpack"):
        assert len(receipt["samples"][arm]) == 5
        values = sorted(receipt["samples"][arm])
        assert receipt["medians_bytes"][arm] == values[2]
    assert receipt["medians_bytes"]["reference"] > receipt["medians_bytes"]["fused"]


def test_v040_final_sha_confirmation_stays_explicit():
    receipt = json.loads((EVIDENCE / "ab" / "fused-final-s424243.json").read_text())
    assert receipt["production_code_sha"] == (
        "45a0ed12201b3a301712815dab32c0dac9c8fee4"
    )
    assert receipt["production_src_dirty"] is False
    assert receipt["summary"]["rows_failed"] == 0
    assert receipt["summary"]["rows_pass"] == receipt["summary"]["rows_total"]
    assert receipt["summary"]["rows_total"] >= 3
