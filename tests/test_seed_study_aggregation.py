"""Seed-study aggregation (v0.3.0/T1): per-seed bands, fail-closed coverage.

The aggregator must never let a same-seed process spread masquerade as
cross-seed robustness, must separate per-seed medians from the pooled median,
and must flag (not silently accept) under-covered matrices: fewer than
``min_seeds`` distinct seeds or fewer than ``min_runs_per_seed`` fresh
processes for a key makes the whole study INSUFFICIENT, because every later
T2-T4 gain claim is gated on this variance band.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_seed_study import aggregate, coverage_of  # noqa: E402


def _receipt(seed: int, run_id: str, speedups: dict) -> dict:
    """Minimal receipt: three comparisons (one per method) at 32k bf16 decode."""
    comparisons = [
        {"context": 32768, "dtype": "bf16", "phase": "decode", "method": method,
         "stage": "select+read", "baseline_ms": 1.0, "candidate_ms": 1.0 / speedup,
         "speedup": speedup, "samples": 20}
        for method, speedup in speedups.items()
    ]
    return {"run_id": run_id, "env": {"seed": seed, "dtype": "bf16"}, "comparisons": comparisons}


def test_per_seed_medians_separate_from_pooled_band():
    receipts = [
        _receipt(11, "rep1-t", {"lsa_flat": 2.0, "block_sparse": 4.0, "lsa_tree": 0.9}),
        _receipt(11, "rep2-t", {"lsa_flat": 2.4, "block_sparse": 3.8, "lsa_tree": 0.9}),
        _receipt(11, "rep3-t", {"lsa_flat": 2.2, "block_sparse": 4.2, "lsa_tree": 0.9}),
        _receipt(22, "rep4-t", {"lsa_flat": 1.5, "block_sparse": 3.0, "lsa_tree": 0.9}),
        _receipt(22, "rep5-t", {"lsa_flat": 1.7, "block_sparse": 3.4, "lsa_tree": 0.9}),
        _receipt(22, "rep6-t", {"lsa_flat": 1.6, "block_sparse": 3.2, "lsa_tree": 0.9}),
        _receipt(33, "rep7-t", {"lsa_flat": 2.1, "block_sparse": 4.4, "lsa_tree": 0.9}),
        _receipt(33, "rep8-t", {"lsa_flat": 2.3, "block_sparse": 4.0, "lsa_tree": 0.9}),
        _receipt(33, "rep9-t", {"lsa_flat": 2.2, "block_sparse": 4.2, "lsa_tree": 0.9}),
    ]
    report = aggregate(receipts, min_seeds=3, min_runs_per_seed=3)
    assert report["status"] == "OK"
    key = report["keys"]["32768/bf16/decode/lsa_flat"]
    # per-seed medians: seed11 -> 2.2, seed22 -> 1.6, seed33 -> 2.2
    assert key["per_seed"]["11"]["median_speedup"] == 2.2
    assert key["per_seed"]["22"]["median_speedup"] == 1.6
    assert key["per_seed"]["33"]["median_speedup"] == 2.2
    # cross-seed band from per-seed medians [1.6, 2.2, 2.2]: min 1.6, max 2.2, median 2.2
    assert key["cross_seed"]["min_seed_median"] == 1.6
    assert key["cross_seed"]["max_seed_median"] == 2.2
    assert key["cross_seed"]["median_of_seed_medians"] == 2.2
    # within-seed process spread is reported per seed (seed 11 spread = 0.4)
    assert key["per_seed"]["11"]["spread"] == pytest.approx(0.4)
    # the pooled median across all 9 runs (1.5..2.4 -> median 2.1) is also kept
    assert key["pooled_median_speedup"] == 2.1


def test_undercovered_matrix_is_insufficient_not_ok():
    # only 2 distinct seeds -> every key is under-covered
    receipts = [_receipt(11, "rep1-t", {"lsa_flat": 2.0}) for _ in range(3)]
    receipts += [_receipt(22, "rep4-t", {"lsa_flat": 2.0}) for _ in range(3)]
    report = aggregate(receipts, min_seeds=3, min_runs_per_seed=3)
    assert report["status"] == "INSUFFICIENT"
    assert report["keys"]["32768/bf16/decode/lsa_flat"]["coverage_ok"] is False


def test_missing_seed_run_flags_coverage_on_that_seed_only():
    # seed 33 has only 2 runs; other keys fine
    receipts = [
        _receipt(11, "r1", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(11, "r2", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(11, "r3", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(22, "r4", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(22, "r5", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(22, "r6", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(33, "r7", {"lsa_flat": 2.0, "block_sparse": 4.0}),
        _receipt(33, "r8", {"lsa_flat": 2.0, "block_sparse": 4.0}),
    ]
    report = aggregate(receipts, min_seeds=3, min_runs_per_seed=3)
    assert report["status"] == "INSUFFICIENT"
    flat = report["keys"]["32768/bf16/decode/lsa_flat"]
    assert flat["per_seed"]["33"]["runs"] == 2
    assert flat["coverage_ok"] is False


def test_coverage_of_counts_seeds_and_runs():
    receipts = [
        _receipt(11, "r1", {"lsa_flat": 2.0}),
        _receipt(11, "r2", {"lsa_flat": 2.0}),
        _receipt(22, "r3", {"lsa_flat": 2.0}),
    ]
    seeds, runs_per_seed = coverage_of(receipts, key=("lsa_flat",))
    assert seeds == [11, 22]
    assert runs_per_seed == {11: 2, 22: 1}
