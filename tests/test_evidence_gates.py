"""Fail-closed evidence: skipped or short gate suites can never be claimable.

The v0.2.0 adapter computed ``correctness_pass = all(gates)`` - vacuously True
for the empty list ``--skip-gates`` produced, so an unverified benchmark wrote a
PASS receipt and ``claimable_lsa_speedup: true``. These tests pin the closed
posture: the manifest decides, the receipt says UNVERIFIED, and the runner's
validator rejects anything whose gates are not exactly the required set, all PASS.
"""

import json

import pytest

from lsa_rocm_lab.gate_manifest import REQUIRED_GATE_IDS, describe, verdict_of


def _full_suite(status="PASS"):
    return [{"gate_id": i, "gate": f"gate {i}", "status": status} for i in REQUIRED_GATE_IDS]


class TestVerdictOf:
    def test_empty_suite_is_not_ok(self):
        verdict = verdict_of([])
        assert verdict["ok"] is False
        assert set(verdict["missing"]) == set(REQUIRED_GATE_IDS)

    def test_exact_suite_all_pass_is_ok(self):
        assert verdict_of(_full_suite())["ok"] is True

    def test_one_failed_gate_is_not_ok(self):
        suite = _full_suite()
        suite[3]["status"] = "FAIL"
        verdict = verdict_of(suite)
        assert verdict["ok"] is False
        assert verdict["failed"] == [REQUIRED_GATE_IDS[3]]

    def test_missing_gate_is_not_ok(self):
        verdict = verdict_of(_full_suite()[:-1])
        assert verdict["ok"] is False
        assert verdict["missing"] == [REQUIRED_GATE_IDS[-1]]

    def test_renamed_or_extra_gate_is_not_ok(self):
        suite = _full_suite()
        suite[-1]["gate_id"] = "G:fp8"
        verdict = verdict_of(suite)
        assert verdict["ok"] is False
        assert "G:bf16" in verdict["missing"] and "G:fp8" in verdict["unexpected"]

    def test_duplicate_gate_is_not_ok(self):
        suite = _full_suite()
        suite.append(dict(suite[0]))
        assert verdict_of(suite)["ok"] is False

    def test_describe_names_the_reason(self):
        verdict = verdict_of([{"gate_id": "A", "gate": "A", "status": "PASS"}])
        text = describe(verdict)
        assert "missing" in text and "A2" in text


class TestAdapterGatesSummary:
    def test_skipped_gates_are_unverified(self):
        from lsa_rocm_lab.kernel_adapter import gates_summary

        summary = gates_summary(gates=None, skipped=True)
        assert summary["correctness_pass"] is False
        assert summary["status"] == "UNVERIFIED"
        assert summary["gates_skipped"] is True

    def test_run_gates_ok_only_on_exact_manifest(self):
        from lsa_rocm_lab.kernel_adapter import gates_summary

        assert gates_summary(gates=_full_suite(), skipped=False)["correctness_pass"] is True
        assert gates_summary(gates=_full_suite()[:-1], skipped=False)["correctness_pass"] is False
        assert gates_summary(gates=[], skipped=False)["correctness_pass"] is False


def _receipt(tmp_path, stage="lsa_rocm_kernel", **over):
    """A minimal valid kernel receipt; perturbations via ``over``."""
    data = {
        "kind": stage,
        "status": "PASS",
        "correctness_pass": True,
        "upstream_sha": "5ebea732d639576d2a7579b0de40e6252f9001f5",
        "gpu_arch": "gfx1151",
        "gates": _full_suite(),
        "comparisons": [
            {"context": 8192, "dtype": "bf16", "phase": "decode", "method": "lsa_flat",
             "baseline_ms": 1.0, "candidate_ms": 0.5, "samples": 20, "speedup": 2.0}
        ],
    }
    data.update(over)
    path = tmp_path / f"{stage}.json"
    path.write_text(json.dumps(data))
    return path


class TestValidateReceipt:
    SHA = "5ebea732d639576d2a7579b0de40e6252f9001f5"

    def test_valid_receipt_accepted(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        validate_receipt(_receipt(tmp_path), "lsa_rocm_kernel", self.SHA)  # no raise

    def test_empty_gate_list_rejected(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        with pytest.raises(ValueError, match="[Gg]ate manifest"):
            validate_receipt(_receipt(tmp_path, gates=[]), "lsa_rocm_kernel", self.SHA)

    def test_unverified_receipt_rejected_loudly(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        with pytest.raises(ValueError, match="UNVERIFIED"):
            validate_receipt(
                _receipt(tmp_path, status="UNVERIFIED", correctness_pass=False, gates=[]),
                "lsa_rocm_kernel", self.SHA,
            )

    def test_skipped_but_pass_claiming_receipt_rejected(self, tmp_path):
        """A forged receipt that claims PASS with no gates behind it must not validate."""
        from lsa_rocm_lab.run import validate_receipt

        with pytest.raises(ValueError, match="[Gg]ate"):
            validate_receipt(_receipt(tmp_path, gates=[]), "lsa_rocm_kernel", self.SHA)

    def test_short_suite_rejected(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        with pytest.raises(ValueError, match="G:bf16"):
            validate_receipt(_receipt(tmp_path, gates=_full_suite()[:-1]), "lsa_rocm_kernel", self.SHA)

    def test_failed_verdict_rejected(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        suite = _full_suite()
        suite[5]["status"] = "FAIL"
        with pytest.raises(ValueError, match=REQUIRED_GATE_IDS[5]):
            validate_receipt(_receipt(tmp_path, gates=suite), "lsa_rocm_kernel", self.SHA)

    def test_missing_receipt_file_rejected(self, tmp_path):
        from lsa_rocm_lab.run import validate_receipt

        with pytest.raises(ValueError, match="no validation receipt|zero"):
            validate_receipt(tmp_path / "absent.json", "lsa_rocm_kernel", self.SHA)
