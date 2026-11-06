"""Seed-study section of the benchmark renderer (v0.3.0/T1 publication).

The published multi-seed table must carry the honest band (per-seed medians,
cross-seed min/median/max, pooled median) and must state explicitly that a
single-seed speedup is not a headline claim - the v0.2.1 2.1x flat figure came
from one favorable seed, and the renderer must never let that shape return.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_seed_study import aggregate  # noqa: E402
from render_tables import seed_study_section  # noqa: E402


def _receipt(seed: int, run_id: str, flat: float) -> dict:
    dense_p50 = 1.0
    cand_p50 = dense_p50 / flat
    dense = {"method": "dense_sdpa", "phase": "decode", "p50_ms": dense_p50,
             "p95_ms": dense_p50 * 1.1, "min_ms": dense_p50 * 0.9, "max_ms": dense_p50 * 1.2,
             "mad_ms": 0.01, "samples": 20, "context": 32768, "dtype": "bf16",
             "heads": 8, "kv_heads": 2, "dim": 64, "seed": seed, "query_len": 1}
    cand = {"method": "lsa_flat", "phase": "decode", "stage": "select+read",
            "p50_ms": cand_p50, "p95_ms": cand_p50 * 1.1, "min_ms": cand_p50 * 0.9,
            "max_ms": cand_p50 * 1.2, "mad_ms": 0.01, "samples": 20, "context": 32768,
            "dtype": "bf16", "heads": 8, "kv_heads": 2, "dim": 64, "seed": seed,
            "top_k": 4096, "top_p": 1024, "window": 512, "sink": 4}
    comp = {"context": 32768, "dtype": "bf16", "phase": "decode", "method": "lsa_flat",
            "stage": "select+read", "baseline_ms": dense_p50, "baseline_p95_ms": dense_p50 * 1.1,
            "candidate_ms": cand_p50, "candidate_p95_ms": cand_p50 * 1.1, "samples": 20,
            "speedup": flat,
            "note": "select+read on a prebuilt index; index_build timed separately"}
    return {"kind": "lsa_rocm_kernel", "status": "PASS", "correctness_pass": True,
            "gates_skipped": False, "run_id": run_id, "repo_head_sha": "0" * 40,
            "upstream_sha": "1" * 40, "gpu_arch": "gfx1151", "gpu_name": "AMD Radeon 8060S",
            "torch": "2.14.0+rocm10.1.0", "command": ["kernel_adapter"],
            "gates": [], "stages": [dense, cand], "comparisons": [comp],
            "env": {"dtype": "bf16", "reps": 20, "warmup": 5, "seed": seed,
                    "contexts": [32768], "phases": ["decode"]}}


def _nine_receipts():
    # seed 11 lucky (2.6x), seeds 22/33 honest band around 1.7x
    speeds = {11: [2.5, 2.6, 2.7], 22: [1.6, 1.7, 1.8], 33: [1.65, 1.75, 1.85]}
    receipts, n = [], 0
    for seed in (11, 22, 33):
        for flat in speeds[seed]:
            n += 1
            receipts.append(_receipt(seed, f"rep{n}-20261106T000000Z", flat))
    return receipts


def test_section_carries_band_and_disclaimer():
    receipts = _nine_receipts()
    report = aggregate(receipts, min_seeds=3, min_runs_per_seed=3)
    section = seed_study_section(receipts, report)
    # per-seed medians visible
    assert "s11=2.60" in section and "s22=1.70" in section and "s33=1.75" in section
    # cross-seed band from per-seed medians (1.70, 1.75, 2.60) - min/median/max
    assert "1.70–2.60" in section and "(med 1.75)" in section
    # pooled median of the 9 runs (sorted -> 1.80) is reported separately
    assert "1.80" in section
    # the disclaimer that single-seed numbers are not headline claims
    assert "single-seed" in section.lower()
    # provenance: every run id appears
    for n in range(1, 10):
        assert f"rep{n}-20261106T000000Z" in section


def test_section_flags_insufficient_coverage():
    receipts = _nine_receipts()[:3]  # only seed 11
    report = aggregate(receipts, min_seeds=3, min_runs_per_seed=3)
    section = seed_study_section(receipts, report)
    assert "INSUFFICIENT" in section
