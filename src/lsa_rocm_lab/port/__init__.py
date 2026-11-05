"""ROCm-native LSA port layer over the pinned upstream checkout.

Upstream (Prefrontal-AI/latent-sparse-attention, Apache-2.0) is imported read-only
from ``third_party/latent-sparse-attention`` at its pinned SHA. This package never
modifies upstream files and never installs upstream CUDA-pinned dependencies; it
composes the upstream *Triton* kernels (which are backend-portable) directly,
bypassing the CuTe DSL front doors that hard-import ``cutlass`` (NVIDIA-only).

Composition mirrors what upstream's ``lsa.sglang.index.selector.Selector`` does at
short/medium context (visible buckets under ``SCAN_SELECT_BUCKETS``): a bf16
centroid GEMM with fp32 out, a Triton top-k listing, and the Triton expansion.
Every composed path is gated against the pure-torch oracle in ``lsa.core.reference``
/ ``lsa.core.ward`` / ``lsa.core.hierarchy`` by ``lsa_rocm_lab.port.gates``.

The one algorithm piece upstream does not ship in Triton is the sparse *read*
(serving delegates it to FA3/FlashMLA/TileLang backends). This port provides it as
``lsa_rocm_lab.port.read_kernel`` - a Triton gather-attention kernel - which is the
genuine ROCm-native contribution of this project.
"""

import os
import pathlib
import sys

# Provenance: the pinned upstream SHA this port was validated against is recorded in
# third_party/upstream.sha by scripts/checkout_upstream.sh and re-read here at import.
ROOT = pathlib.Path(__file__).resolve().parents[3]
UPSTREAM = pathlib.Path(os.getenv("LSA_UPSTREAM", ROOT / "third_party" / "latent-sparse-attention"))


def upstream_sha() -> str:
    """Return the pinned upstream commit SHA, or UNAVAILABLE when the checkout is missing."""
    marker = UPSTREAM.parent / "upstream.sha"
    if marker.is_file():
        return marker.read_text().strip()
    return "UNAVAILABLE"


def _bootstrap() -> None:
    src = UPSTREAM / "src"
    if not (src / "lsa" / "core" / "reference.py").is_file():
        raise FileNotFoundError(
            f"Upstream checkout missing at {UPSTREAM}; run scripts/checkout_upstream.sh"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap()

# The Triton kernels are the port surface. Importing them here fails fast (and loudly)
# if the pinned upstream moves them, instead of deep inside a benchmark loop.
from lsa.core.kernels.triton import ward as triton_ward  # noqa: E402
from lsa.core.kernels.triton import expansion as triton_expansion  # noqa: E402
from lsa.core.kernels.triton import top_k as triton_top_k  # noqa: E402
from lsa.core.kernels.triton import scan_select as triton_scan_select  # noqa: E402

# ROCm note: upstream's Triton ward merge defaults to the tf32x3 Gram dot, an NVIDIA
# input precision. gfx1151 has no tf32 tensor cores, so the port pins "ieee" (full
# fp32), which is also the closest to the fp32 torch oracle.
HIP_PRECISION = "ieee"

__all__ = [
    "triton_ward",
    "triton_expansion",
    "triton_top_k",
    "triton_scan_select",
    "HIP_PRECISION",
    "upstream_sha",
    "UPSTREAM",
    "ROOT",
]
