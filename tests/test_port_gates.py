"""The port's correctness gates as pytest - GPU-required tests skip loudly, never pass silently.

Run on the target machine with the ROCm venv:
    .venv/bin/python -m pytest tests/ -v
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("LSA_SKIP_GPU_TESTS") == "1", reason="LSA_SKIP_GPU_TESTS=1"
)


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no ROCm GPU visible to torch")
def test_port_gates_all_pass(tmp_path):
    """Every gate in lsa_rocm_lab.port.gates must PASS on the GPU."""
    from lsa_rocm_lab.port.gates import run_all_gates

    verdicts = run_all_gates()
    failures = [v for v in verdicts if v["status"] != "PASS"]
    assert not failures, json.dumps(failures, indent=2)


@pytest.mark.skipif(not _gpu_available(), reason="no ROCm GPU visible to torch")
def test_sparse_read_full_selection_matches_sdpa():
    """The design's anchor: k >= T reproduces dense causal attention.

    The read kernel's contract is causal lists (what the expansion emits), so the
    full-selection ids here are causal prefixes - row t lists keys 0..t, -1 padded.
    """
    import torch

    from lsa_rocm_lab.port import pipeline
    from lsa_rocm_lab.port.read_kernel import sparse_read

    torch.manual_seed(7)
    T, H, D = 256, 4, 64
    q = torch.randn(1, T, H, D, device="cuda", dtype=torch.float32) * 0.2
    k = torch.randn(1, T, 1, D, device="cuda", dtype=torch.float32) * 0.2
    v = torch.randn_like(k)
    row = torch.arange(T, device="cuda", dtype=torch.int32)
    causal = row[None, :] <= row[:, None]
    ids = torch.where(causal, row[None, :].expand(T, T), torch.tensor(-1, device="cuda", dtype=torch.int32))
    lengths = (row + 1).to(torch.int32)
    out = sparse_read(q, k, v, ids.view(1, T, T).contiguous(), lengths.view(1, T), D**-0.5)
    dense = pipeline.dense_read(q, k, v, D**-0.5)
    assert torch.allclose(out, dense, atol=1e-4, rtol=1e-4), float((out - dense).abs().max())


def test_missing_port_still_blocks():
    """The starter's contract: no adapter command means BLOCKED, never a fake PASS."""
    from lsa_rocm_lab.run import stage_command

    saved = os.environ.pop("LSA_ROCM_KERNEL_CMD", None)
    try:
        assert stage_command("lsa_rocm_kernel", True) is None
    finally:
        if saved is not None:
            os.environ["LSA_ROCM_KERNEL_CMD"] = saved
