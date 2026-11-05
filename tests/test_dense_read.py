"""Dense baseline semantics - decode and multi-token tails must be the prefill's tail.

``pipeline.dense_read`` is the baseline every speedup divides by. Its query rows
are the *tail* ``q_len`` positions of the ``kv_len``-long cache: a single decode
token attends the whole cache, a multi-token chunk attends with bottom-right
aligned causality. This is the release-blocking companion of gate G - the v0.2.0
harness passed ``is_causal=True`` for a 1-row query, which masks to key 0 only.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("LSA_SKIP_GPU_TESTS") == "1", reason="LSA_SKIP_GPU_TESTS=1"
)


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and bool(torch.version.hip)
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_available(), reason="no ROCm GPU visible to torch")

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def geometry(dtype, T=1024, Hq=8, Hkv=2, D=64, seed=20261106):
    torch.manual_seed(seed)
    q = (torch.randn(1, T, Hq, D, device="cuda") * 0.2).to(dtype)
    k = (torch.randn(1, T, Hkv, D, device="cuda") * 0.2).to(dtype)
    v = (torch.randn(1, T, Hkv, D, device="cuda") * 0.2).to(dtype)
    return q, k, v


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
)
@gpu
def test_dense_decode_equals_prefill_last_row(dtype):
    """A single-token query over the full cache is the prefill's last row."""
    from lsa_rocm_lab.port import pipeline

    q, k, v = geometry(dtype)
    scale = q.shape[-1] ** -0.5
    prefill = pipeline.dense_read(q, k, v, scale)
    decode = pipeline.dense_read(q[:, -1:], k, v, scale)
    atol, rtol = TOLERANCES[dtype]
    assert torch.allclose(decode.float(), prefill[:, -1:].float(), atol=atol, rtol=rtol), float(
        (decode.float() - prefill[:, -1:].float()).abs().max()
    )


@pytest.mark.parametrize("query_len", [2, 3, 16, 127])
@gpu
def test_dense_multi_token_tail_equals_prefill_tail(query_len):
    """A non-square chunk of the last positions keeps absolute-position causality."""
    from lsa_rocm_lab.port import pipeline

    q, k, v = geometry(torch.bfloat16)
    scale = q.shape[-1] ** -0.5
    prefill = pipeline.dense_read(q, k, v, scale)
    tail = pipeline.dense_read(q[:, -query_len:], k, v, scale)
    assert torch.allclose(tail.float(), prefill[:, -query_len:].float(), atol=2e-2, rtol=2e-2), float(
        (tail.float() - prefill[:, -query_len:].float()).abs().max()
    )


@gpu
def test_dense_square_prefill_unchanged():
    """The square case must remain the plain causal SDPA the D2 anchor pins."""
    import torch.nn.functional as F

    from lsa_rocm_lab.port import pipeline

    q, k, v = geometry(torch.float32, T=256)
    scale = q.shape[-1] ** -0.5
    got = pipeline.dense_read(q, k, v, scale)
    want = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        is_causal=True, scale=scale, enable_gqa=True,
    ).permute(0, 2, 1, 3)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


@gpu
def test_dense_query_longer_than_cache_rejected():
    """Queries longer than the cache have no causal meaning - refuse them."""
    from lsa_rocm_lab.port import pipeline

    q, k, v = geometry(torch.float32, T=64)
    with pytest.raises(ValueError):
        pipeline.dense_read(q, k[:, :32], v[:, :32], q.shape[-1] ** -0.5)
