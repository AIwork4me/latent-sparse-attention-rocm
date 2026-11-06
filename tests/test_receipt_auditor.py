"""Adversarial fixtures for the independent receipt auditor (F12).

Every tampering an author (or an automated table generator) could apply to
make weak evidence look strong must be caught: altered speedup, swapped
baseline, wrong SHAs, mixed seed/dtype, duplicate/missing rows, fake PASS
gates, NaN/Inf, a precomputed index mislabeled as full LSA, corrupted bundle
checksums, changed repo HEAD and replayed run ids. The auditor recomputes;
it never trusts the receipt's own claims.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lsa_rocm_lab.evidence.auditor import audit_file, audit_receipt
from lsa_rocm_lab.gate_manifest import REQUIRED_GATE_IDS

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "5ebea732d639576d2a7579b0de40e6252f9001f5"
REPO = "d6bb5e32e1a9be10cc63ce0851b81fd1e53e1d41"


def _gates():
    return [{"gate_id": i, "gate": f"gate {i}", "status": "PASS"} for i in REQUIRED_GATE_IDS]


def _dense_row(phase, p50, query_len=None):
    row = {"method": "dense_sdpa", "phase": phase, "p50_ms": p50, "p95_ms": p50 * 1.1,
           "min_ms": p50 * 0.9, "max_ms": p50 * 1.2, "mad_ms": p50 * 0.01, "samples": 20,
           "context": 32768, "dtype": "bf16", "heads": 8, "kv_heads": 2, "dim": 64, "seed": 20261105}
    if phase == "decode":
        row["query_len"] = query_len if query_len is not None else 1
    return row


def _sparse_row(p50, top_k=4096):
    return {"method": "lsa_flat", "phase": "decode", "stage": "select+read",
            "p50_ms": p50, "p95_ms": p50 * 1.1, "min_ms": p50 * 0.9, "max_ms": p50 * 1.2,
            "mad_ms": p50 * 0.01, "samples": 20, "context": 32768, "dtype": "bf16",
            "heads": 8, "kv_heads": 2, "dim": 64, "seed": 20261105,
            "top_k": top_k, "top_p": 1024, "window": 512, "sink": 4}


def _comparison(baseline_p50, candidate_p50, **over):
    comp = {"context": 32768, "dtype": "bf16", "phase": "decode", "method": "lsa_flat",
            "stage": "select+read",
            "baseline_ms": baseline_p50, "baseline_p95_ms": baseline_p50 * 1.1,
            "candidate_ms": candidate_p50, "candidate_p95_ms": candidate_p50 * 1.1,
            "samples": 20, "speedup": baseline_p50 / candidate_p50,
            "note": "select+read on a prebuilt index; index_build timed separately"}
    comp.update(over)
    return comp


def _receipt(**over):
    dense = _dense_row("decode", 1.0)
    cand = _sparse_row(0.5)
    data = {
        "kind": "lsa_rocm_kernel", "status": "PASS", "correctness_pass": True,
        "gates_skipped": False, "run_id": "20261106T000000Z-adapter",
        "repo_head_sha": REPO, "upstream_sha": UPSTREAM,
        "gpu_arch": "gfx1151", "gpu_name": "AMD Radeon 8060S", "torch": "2.14.0+rocm10.1.0",
        "command": ["python", "-m", "lsa_rocm_lab.kernel_adapter", "--dtype", "bf16"],
        "gates": _gates(),
        "stages": [dense, cand, _dense_row("prefill", 5.0)],
        "comparisons": [_comparison(1.0, 0.5)],
        "env": {"dtype": "bf16", "reps": 20, "warmup": 5, "seed": 20261105,
                "contexts": "all", "phases": ["prefill", "decode"]},
    }
    data.update(over)
    return data


def _audit(data, **kwargs):
    return audit_receipt(data, source="fixture", **kwargs)


def _codes(report):
    return [f.code for f in report.findings]


class TestHonestReceipt:
    def test_valid_receipt_approved_and_recomputed(self):
        report = _audit(_receipt(), expected_upstream_sha=UPSTREAM, expected_repo_sha=REPO)
        assert report.verdict == "APPROVED", report.summary()
        row = report.recomputed["comparisons"][0]
        assert row["speedup"] == pytest.approx(2.0)
        assert row["top_k"] == 4096  # sparse budget carried into the audit record


class TestTamperedNumbers:
    def test_altered_speedup_rejected(self):
        receipt = _receipt()
        receipt["comparisons"][0]["speedup"] = 3.9  # flattered
        assert "SPEEDUP_MISMATCH" in _codes(_audit(receipt))

    def test_swapped_baseline_rejected(self):
        receipt = _receipt()
        receipt["comparisons"][0]["baseline_ms"] = 0.5  # candidate's own timing
        assert "BASELINE_MISMATCH" in _codes(_audit(receipt))

    def test_edited_candidate_ms_rejected(self):
        receipt = _receipt()
        receipt["comparisons"][0]["candidate_ms"] = 0.25
        assert "CANDIDATE_MISMATCH" in _codes(_audit(receipt))

    def test_nan_timing_rejected(self):
        receipt = _receipt()
        receipt["stages"][1]["p50_ms"] = float("nan")
        assert "NAN_INF" in _codes(_audit(receipt))

    def test_inf_speedup_rejected(self):
        receipt = _receipt()
        receipt["comparisons"][0]["speedup"] = float("inf")
        assert "NAN_INF" in _codes(_audit(receipt))

    def test_samples_inflation_rejected(self):
        receipt = _receipt()
        receipt["comparisons"][0]["samples"] = 100
        assert "SAMPLES" in _codes(_audit(receipt))

    def test_sub_evidence_bar_samples_rejected(self):
        receipt = _receipt()
        for row in receipt["stages"]:
            row["samples"] = 5
        receipt["comparisons"][0]["samples"] = 5
        assert "SAMPLES" in _codes(_audit(receipt))

    def test_p50_outside_min_max_rejected(self):
        receipt = _receipt()
        receipt["stages"][0]["min_ms"] = 2.0  # p50 below min: impossible
        assert "TIMING" in _codes(_audit(receipt))


class TestTamperedEvidence:
    def test_wrong_upstream_sha_rejected(self):
        receipt = _receipt(upstream_sha="f" * 40)
        assert "UPSTREAM_SHA_MISMATCH" in _codes(
            _audit(receipt, expected_upstream_sha=UPSTREAM))

    def test_changed_repo_head_rejected(self):
        receipt = _receipt(repo_head_sha="a" * 40)
        assert "REPO_SHA_MISMATCH" in _codes(
            _audit(receipt, expected_repo_sha=REPO))

    def test_malformed_sha_rejected_even_without_pin(self):
        receipt = _receipt(upstream_sha="not-a-sha")
        assert "PROVENANCE" in _codes(_audit(receipt))

    def test_mixed_seed_rejected(self):
        receipt = _receipt()
        receipt["stages"][1]["seed"] = 999
        assert "MIXED_SEED" in _codes(_audit(receipt))

    def test_mixed_dtype_rejected(self):
        receipt = _receipt()
        receipt["stages"][1]["dtype"] = "fp16"
        assert "MIXED_DTYPE" in _codes(_audit(receipt))

    def test_duplicate_stage_row_rejected(self):
        receipt = _receipt()
        receipt["stages"].append(dict(receipt["stages"][1]))
        assert "DUPLICATE_ROW" in _codes(_audit(receipt))

    def test_missing_dense_baseline_rejected(self):
        receipt = _receipt()
        receipt["stages"] = [row for row in receipt["stages"] if not (
            row["method"] == "dense_sdpa" and row["phase"] == "decode")]
        assert "MISSING_BASELINE" in _codes(_audit(receipt))

    def test_missing_stage_row_rejected(self):
        receipt = _receipt()
        receipt["stages"] = [receipt["stages"][0]]  # candidate row deleted
        assert "MISSING_CANDIDATE" in _codes(_audit(receipt))

    def test_fake_extra_pass_gate_rejected(self):
        receipt = _receipt()
        receipt["gates"].append({"gate_id": "H:fp8", "gate": "fake", "status": "PASS"})
        assert "GATE_MANIFEST" in _codes(_audit(receipt))

    def test_failed_gate_with_pass_claim_rejected(self):
        receipt = _receipt()
        receipt["gates"][5]["status"] = "FAIL"
        assert "GATE_MANIFEST" in _codes(_audit(receipt))

    def test_skipped_gates_never_verified(self):
        receipt = _receipt(status="UNVERIFIED", correctness_pass=False, gates=[])
        receipt["gates_skipped"] = True
        codes = _codes(_audit(receipt))
        assert "UNVERIFIED_GATES" in codes and "GATE_MANIFEST" in codes

    def test_decode_baseline_query_len_must_be_one(self):
        """The v0.2.0 bug shape: dense decode attending only key 0 (wrong mask)."""
        receipt = _receipt()
        receipt["stages"][0]["query_len"] = 0
        assert "DECODE_BASELINE" in _codes(_audit(receipt))

    def test_missing_decode_parity_gate_rejected(self):
        receipt = _receipt()
        receipt["gates"] = [g for g in receipt["gates"] if g["gate_id"] != "G:bf16"]
        assert "GATE_MANIFEST" in _codes(_audit(receipt))

    def test_precomputed_index_mislabeled_as_full_lsa(self):
        receipt = _receipt()
        receipt["comparisons"][0].pop("note")
        receipt["comparisons"][0]["stage"] = "full_e2e_attention"  # scope inflation
        receipt["stages"][1]["stage"] = "full_e2e_attention"
        assert "MISLABELED_SCOPE" in _codes(_audit(receipt))

    def test_replayed_run_id_rejected_across_set(self):
        seen = set()
        first = _audit(_receipt(), seen_run_ids=seen)
        second = _audit(_receipt(), seen_run_ids=seen)
        assert first.verdict == "APPROVED"
        assert "REPLAYED_RUN_ID" in _codes(second)

    def test_unreadable_json_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{truncated")
        assert audit_file(bad).verdict == "REJECTED"


class TestBundleChecksum:
    def _bundle(self, tmp_path):
        receipt_path = tmp_path / "receipt.json"
        receipt_path.write_text(json.dumps(_receipt()))
        return receipt_path

    def test_matching_checksums_pass(self, tmp_path):
        receipt_path = self._bundle(tmp_path)
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        (tmp_path / "SHA256SUMS").write_text(f"{digest}  receipt.json\n")
        report = audit_file(receipt_path)
        assert report.verdict == "APPROVED"

    def test_corrupted_checksum_rejected(self, tmp_path):
        receipt_path = self._bundle(tmp_path)
        (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  receipt.json\n")
        report = audit_file(receipt_path)
        assert "CHECKSUM_MISMATCH" in _codes(report)

    def test_tampered_file_after_checksum_rejected(self, tmp_path):
        receipt_path = self._bundle(tmp_path)
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        (tmp_path / "SHA256SUMS").write_text(f"{digest}  receipt.json\n")
        data = _receipt()
        data["comparisons"][0]["speedup"] = 99.0  # edit after checksumming
        receipt_path.write_text(json.dumps(data))
        report = audit_file(receipt_path)
        assert "CHECKSUM_MISMATCH" in _codes(report)
        assert "SPEEDUP_MISMATCH" in _codes(report)


class TestCli:
    def test_cli_approves_and_writes_report(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps(_receipt()))
        report_path = tmp_path / "audit.json"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "audit_receipts.py"),
             "--upstream-sha-file", str(tmp_path / "absent.sha"),
             "--report", str(report_path), str(good)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(report_path.read_text())["verdict"] == "APPROVED"

    def test_cli_rejects_tampered_receipt_nonzero(self, tmp_path):
        receipt = _receipt()
        receipt["comparisons"][0]["speedup"] = 8.8
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(receipt))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "audit_receipts.py"), str(bad)],
            capture_output=True, text=True)
        assert result.returncode == 1
        assert "SPEEDUP_MISMATCH" in result.stdout
