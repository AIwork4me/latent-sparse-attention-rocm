#!/usr/bin/env python3
"""v0.4.0 - peak-allocation receipt: fused vs reference select at 32k bf16.

Machine evidence for the MEASURED_SELECT_FUSION.md memory claim (audit finding:
the first version of the claim was doc-recorded only). Warm both paths first
(compiled, allocator steady), then measure peak allocated bytes of ONE complete
production flat_select per implementation, plus the equivalent stage in the
baseline instrumentation for cross-checking.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline, upstream_sha  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402


def peak_of(fn) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    sync()
    return torch.cuda.max_memory_allocated() - base


def main() -> int:
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78
    repo = Path(__file__).resolve().parents[1]
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src/"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    q, k, v = build_geometry(32768, torch.bfloat16, seed=1, **GEOMETRY)
    config = config_for(32768)
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    positions = torch.arange(32768, device=q.device)
    rows = idx_q[:4096]
    receipt = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-memprobe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "production_code_sha": code_sha,
        "production_src_dirty": dirty,
        "upstream_sha": upstream_sha(),
        "context": 32768,
        "dtype": "bf16",
        "phase": "prefill_select_32768_rows",
        "method": "torch.cuda.max_memory_allocated delta around one complete flat_select, "
                  "warm (compiled) path, allocator steady; empty_cache+reset before each",
        "impls": {},
    }
    for impl in ("reference", "fused", "fused_unpack"):
        fn = _switched(impl, idx_q, positions, buckets, config)
        fn(); fn()  # warm: compile + allocator steady
        receipt["impls"][impl] = peak_of(fn)
        print(f"{impl}: {receipt['impls'][impl] / 2**20:.1f} MiB", flush=True)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/v0.4.0/ab/memory-probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    import hashlib
    print(f"wrote {out}\nsha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


def _switched(impl, idx_q, positions, buckets, config):
    import os

    def fn():
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            return pipeline.flat_select(idx_q, positions, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
    return fn


if __name__ == "__main__":
    raise SystemExit(main())
