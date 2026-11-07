"""The upstream-failure triage comparator - known failures visible, new ones fatal."""

from lsa_rocm_lab.upstream_triage import node_ids_from_junit, render, triage

MANIFEST = {
    "tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere": {
        "classification": "UNSUPPORTED_PRECISION", "reason": "tf32x3 rejected by HIP",
    },
    "tests/unit/core/test_shape_kernels.py::test_power_iteration_finds_the_leading_directions": {
        "classification": "NUMERIC_DIVERGENCE_UNUSED_PATH", "reason": "marginal RDNA tolerance",
    },
}


class TestTriage:
    def test_classified_failure_is_ok(self):
        result = triage(
            ["tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere"],
            MANIFEST,
        )
        assert result["unclassified"] == []
        assert "UNSUPPORTED_PRECISION" in result["classified"][
            "tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere"
        ]["classification"]

    def test_unclassified_failure_is_a_regression(self):
        result = triage(["tests/unit/core/test_ward_kernel.py::test_brand_new_regression"], MANIFEST)
        assert result["unclassified"] == ["tests/unit/core/test_ward_kernel.py::test_brand_new_regression"]

    def test_render_keeps_classified_visible_and_names_regressions(self):
        ok = render(
            ["tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere"],
            203,
            triage(["tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere"], MANIFEST),
        )
        assert "TRIAGE: PASS" in ok and "UNSUPPORTED_PRECISION" in ok
        bad = render(
            ["tests/unit/core/test_new_one"],
            202,
            triage(["tests/unit/core/test_new_one"], MANIFEST),
        )
        assert "TRIAGE: FAIL" in bad and "UNCLASSIFIED" in bad


JUNIT_OK = """<testsuites><testsuite>
<testcase classname="tests.unit.core.test_ward" name="test_pairing_is_exact"/>
<testcase classname="tests.unit.core.test_ward_kernel" name="test_tf32x3_merges_the_same_pools_almost_everywhere"><failure message="AssertionError"/></testcase>
</testsuite></testsuites>"""


class TestJunitParsing:
    def test_failures_and_passes_from_junit(self, tmp_path):
        path = tmp_path / "junit.xml"
        path.write_text(JUNIT_OK)
        failed, passed = node_ids_from_junit(path)
        assert failed == [
            "tests/unit/core/test_ward_kernel.py::test_tf32x3_merges_the_same_pools_almost_everywhere"
        ]
        assert passed == 1
