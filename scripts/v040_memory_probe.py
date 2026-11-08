#!/usr/bin/env python3
"""v0.4.0 - peak-allocation receipt: fused vs reference select at 32k bf16.

Machine evidence for the MEASURED_SELECT_FUSION.md memory claim (audit finding:
the first version of the claim was doc-recorded only). Warm both paths first
(compiled, allocator steady), then measure peak allocated bytes of ONE complete
production flat_select per implementation, several raw samples each (the receipt
keeps every sample; medians are derived, reductions are recomputed independently
downstream - never inside this script).

Provenance (R1.1-era lessons baked in): the receipt records the production tree
SHA, per-src and whole-tree dirty flags, the pinned upstream SHA, the exact
device and runtime versions, and the sha256 of this script and of the raw
samples - so a receipt can never again silently measure a dirty checkout or an
unidentified interpreter. --repo pins WHICH tree is measured (defaults to this
script's own checkout); imports always come from that tree's src/.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()


def peak_of(fn) -> int:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="production tree to measure and to import src/ from (default: this checkout)")
    parser.add_argument("--samples", type=int, default=5, help="raw samples per implementation")
    parser.add_argument("--out", type=Path, default=Path("evidence/v0.4.0/ab/memory-probe.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    import torch  # noqa: E402
    import triton  # noqa: E402

    from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
    from lsa_rocm_lab.port import pipeline, upstream_sha  # noqa: E402
    from lsa_rocm_lab.port.bench import build_geometry  # noqa: E402
    from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402

    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78
    src_dirty = bool(_git(repo, "status", "--porcelain", "--", "src/"))
    tree_dirty = bool(_git(repo, "status", "--porcelain"))
    receipt = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-memprobe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "production_repo": str(repo),
        "production_code_sha": _git(repo, "rev-parse", "HEAD"),
        "production_src_dirty": src_dirty,
        "production_tree_dirty": tree_dirty,
        "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_sha": upstream_sha(),
        "context": 32768,
        "dtype": "bf16",
        "phase": "prefill_select_32768_rows",
        "method": "torch.cuda.max_memory_allocated delta around one complete flat_select, "
                  "warm (compiled) path, allocator steady; empty_cache+reset before each sample; "
                  f"{args.samples} raw samples per impl (medians derived, not sampled)",
        "samples": {},
        "medians_bytes": {},
    }
    if src_dirty or tree_dirty:
        print(json.dumps({"status": "BLOCKED",
                          "reason": f"production tree dirty (src={src_dirty}, tree={tree_dirty}) - "
                                    "receipts measure clean checkouts only"},
                         ))
        return 78

    q, k, v = build_geometry(32768, torch.bfloat16, seed=1, **GEOMETRY)
    config = config_for(32768)
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    positions = torch.arange(32768, device=q.device)
    rows = idx_q[:4096]
    import os

    def _switched(impl):
        def fn():
            os.environ["LSA_SELECT_IMPL"] = impl
            try:
                return pipeline.flat_select(rows, positions, buckets, config)
            finally:
                os.environ.pop("LSA_SELECT_IMPL", None)
        return fn

    for impl in ("reference", "fused", "fused_unpack"):
        fn = _switched(impl)
        fn(); fn()  # warm: compile + allocator steady
        samples = [peak_of(fn) for _ in range(args.samples)]
        receipt["samples"][impl] = samples
        ordered = sorted(samples)
        receipt["medians_bytes"][impl] = ordered[len(ordered) // 2]
        print(f"{impl}: median {ordered[len(ordered) // 2] / 2**20:.1f} MiB "
              f"(raw {[f'{s / 2**20:.1f}' for s in samples]})", flush=True)
    receipt["samples_sha256"] = hashlib.sha256(
        json.dumps(receipt["samples"], sort_keys=True).encode()).hexdigest()
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(f"wrote {out}\nsha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
