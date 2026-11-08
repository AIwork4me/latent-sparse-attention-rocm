#!/usr/bin/env python3
"""M2 (overnight_v1) - fail-closed long-context scaling ladder, 64k -> 256k.

Measures the production select / select+read path per context size with an explicit
memory envelope. Before every (ctx, lane) step the script estimates projected
dominant allocations per arm:

- if the FUSED arm's projection exceeds --budget-frac of currently-free VRAM the
  step is refused: BLOCKED_RESOURCE (and the prefill lane closes for larger ctx);
- if the REFERENCE arm's projection exceeds --ref-arm-frac of free VRAM the
  reference arm is skipped proactively (REFERENCE_ARM_SKIPPED_PROJECTION) so a
  doomed O(ctx^2/8) int64 allocation can never destabilize the UMA host; the fused
  arm then runs single-arm with a CHUNKED reference oracle;
- runtime OOM anywhere is caught, classified, recorded (OOM / single-arm fallback),
  never reported as PASS or as zero time.

Lanes: decode (b1 + ragged) is light across the whole ladder; prefill is the
high-memory lane (reference materializes [ctx, ~ctx/8] int64 keys, fused does not).

Every phase row records: oracle mode + exact verdict, determinism (3 repeats),
timing mode, both arms' p50/p95/min/max + raw samples, launches, first-call
compile, peak allocated/reserved MiB, VRAM/MemAvailable before+after.
Statuses: PASS | FAILED | OOM | BLOCKED_RESOURCE (+ arm-level markers).
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lsa_rocm_lab.kernel_adapter import GEOMETRY, config_for  # noqa: E402
from lsa_rocm_lab.port import pipeline, upstream_sha  # noqa: E402
from lsa_rocm_lab.port.bench import build_geometry, sync  # noqa: E402
from lsa_rocm_lab.port.gates import make_index_space  # noqa: E402


def mem_state():
    free_b, total_b = torch.cuda.mem_get_info()
    mi = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        mi[k] = int(v.strip().split()[0]) * 1024
    return {
        "vram_free_gib": round(free_b / 2**30, 2),
        "vram_total_gib": round(total_b / 2**30, 2),
        "mem_available_gib": round(mi.get("MemAvailable", 0) / 2**30, 2),
        "torch_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }


def columns_for(ctx: int) -> int:
    """Dominant score-matrix width (matches observed widths: 32k -> 4096,
    65536 -> ~8128); estimate-only, slightly conservative."""
    return ctx // 8 + 640


def estimate_bytes(ctx: int, lane: str, impl: str, dtype_name: str) -> int:
    esz = 2 if dtype_name in ("fp16", "bf16") else 4
    cols = columns_for(ctx)
    kv = ctx * 2 * 64 * esz * 2          # k + v [1, T, 2, 64]
    q = ctx * 8 * 64 * esz               # q [1, T, 8, 64]
    idx = ctx * 128 * 4 * 2              # index spaces fp32
    base = kv + q + idx + (1 << 28)      # +256 MiB slack (buckets, fragmentation)
    lists_out = ctx * 1024 * 8 * 2       # output lists both arms (+lengths small)
    if lane == "decode":
        return base + cols * 4 + 8 * 1024 * 8
    scores = ctx * cols * 4              # fp32 score matrix (both arms)
    keys_ref = ctx * cols * 8 if impl == "reference" else 0
    packed_topk_ref = ctx * 1024 * 8 * 2 if impl == "reference" else 0
    return base + scores + keys_ref + packed_topk_ref + lists_out


def summarize(samples):
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": [round(s, 6) for s in samples],
    }


def timed_pair(fn_ref, fn_cand, warmup, reps):
    for _ in range(warmup):
        fn_ref()
        fn_cand()
    sync()
    ref_s, cand_s = [], []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn_ref()
        sync()
        ref_s.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        fn_cand()
        sync()
        cand_s.append((time.perf_counter() - t0) * 1000)
    return ref_s, cand_s


def timed_single(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    sync()
    s = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        s.append((time.perf_counter() - t0) * 1000)
    return s


def launches_per_iter(fn, iters=8):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        sync()
    rows = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    return sum(e.count for e in rows) / iters


def select_fn(impl, idx_rows, positions, buckets, config):
    def fn():
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            return pipeline.flat_select(idx_rows, positions, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
    return fn


def select_read_fn(impl, q_read, k, v, idx_rows, positions, buckets, config, scale):
    def fn():
        os.environ["LSA_SELECT_IMPL"] = impl
        try:
            lists, lengths = pipeline.flat_select(idx_rows, positions, buckets, config)
        finally:
            os.environ.pop("LSA_SELECT_IMPL", None)
        return pipeline.lsa_read(q_read, k, v, lists, lengths, scale)
    return fn


def chunked_oracle(idx_rows, positions, buckets, config, block):
    """Reference semantics in row-blocks to bound memory where the monolithic
    reference cannot fit. flat_select is row-independent (per-row scores + top-k);
    equivalence to the monolithic oracle is asserted at ladder start."""
    outs = []
    for i in range(0, positions.shape[0], block):
        outs.append(pipeline.flat_select_reference(
            idx_rows[i:i + block], positions[i:i + block], buckets, config))
    return torch.cat([o[0] for o in outs], 0), torch.cat([o[1] for o in outs], 0)


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def recover():
    gc.collect()
    torch.cuda.empty_cache()
    sync()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctxs", default="65536,98304,131072,196608,262144")
    ap.add_argument("--dtypes", default="bf16")
    ap.add_argument("--lanes", default="decode,prefill")
    ap.add_argument("--seed", type=int, default=424243)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ragged-rows", type=int, default=32)
    ap.add_argument("--budget-frac", type=float, default=0.55,
                    help="fused-arm projection must stay below this fraction of free VRAM")
    ap.add_argument("--ref-arm-frac", type=float, default=0.85,
                    help="reference arm skipped by projection above this fraction")
    ap.add_argument("--chunk-rows", type=int, default=8192)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "BLOCKED", "reason": "no ROCm GPU"}))
        return 78
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1151"):
        print(json.dumps({"status": "BLOCKED", "reason": f"not gfx1151: {arch}"}))
        return 78

    repo = Path(__file__).resolve().parents[1]
    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src/"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    receipt = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/ctx_scale_ladder.py (overnight_v1 M2)",
        "gpu_arch": arch, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "production_code_sha": code_sha, "production_src_dirty": dirty,
        "upstream_sha": upstream_sha(),
        "seed": args.seed, "warmup": args.warmup, "reps": args.reps,
        "budget_frac": args.budget_frac, "ref_arm_frac": args.ref_arm_frac,
        "gpu_state_start": subprocess.run(
            ["rocm-smi", "--showclocks", "--showtemp", "--showmeminfo", "vram"],
            capture_output=True, text=True).stdout[-600:],
        "rows": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        args.out.write_text(json.dumps(receipt, indent=2))

    # sanity: chunked oracle == monolithic reference at a small context
    q, k, v = build_geometry(8192, torch.bfloat16, seed=args.seed, **GEOMETRY)
    config = config_for(8192)
    idx_q, idx_k = make_index_space(q, k)
    buckets = pipeline.build_buckets_gpu(idx_k, config)
    pos = torch.arange(8192, device=q.device)
    mono = pipeline.flat_select_reference(idx_q, pos, buckets, config)
    chunk = chunked_oracle(idx_q, pos, buckets, config, 2048)
    assert torch.equal(mono[0], chunk[0]) and torch.equal(mono[1], chunk[1]), \
        "chunked oracle != monolithic reference at 8k - do not trust the ladder"
    del q, k, v, idx_q, idx_k, buckets, mono, chunk
    recover()
    print("[oracle-sanity] chunked == monolithic reference at 8192: OK", flush=True)

    prefill_lane_open = "prefill" in args.lanes.split(",")
    stop_reason = None

    for dtype_name in args.dtypes.split(","):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[dtype_name]
        for ctx in [int(c) for c in args.ctxs.split(",")]:
            lanes = ["decode"] + (["prefill"] if prefill_lane_open else [])
            for lane in lanes:
                row = {"dtype": dtype_name, "context": ctx, "lane": lane,
                       "columns_est": columns_for(ctx), "phases": []}
                receipt["rows"].append(row)
                row["mem_before"] = mem_state()

                est_ref = estimate_bytes(ctx, lane, "reference", dtype_name)
                est_fused = estimate_bytes(ctx, lane, "fused", dtype_name)
                row["estimate_gib"] = {"reference": round(est_ref / 2**30, 2),
                                       "fused": round(est_fused / 2**30, 2)}
                free_b = row["mem_before"]["vram_free_gib"] * 2**30
                ref_skipped = est_ref > args.ref_arm_frac * free_b
                row["reference_arm"] = ("SKIPPED_PROJECTION" if ref_skipped else "RUN")
                if est_fused > args.budget_frac * free_b:
                    row["status"] = "BLOCKED_RESOURCE"
                    row["blocked_detail"] = (
                        f"fused projection {est_fused / 2**30:.1f} GiB > "
                        f"{args.budget_frac:.0%} of {row['mem_before']['vram_free_gib']:.1f} GiB free")
                    flush()
                    print(f"[{dtype_name} {ctx} {lane}] BLOCKED_RESOURCE "
                          f"({row['blocked_detail']})", flush=True)
                    if lane == "prefill":
                        prefill_lane_open = False
                        stop_reason = stop_reason or \
                            f"prefill lane closed at ctx={ctx} (fused projection)"
                    continue

                tensors = None
                try:
                    q, k, v = build_geometry(ctx, dtype, seed=args.seed + ctx, **GEOMETRY)
                    config = config_for(ctx)
                    idx_q, idx_k = make_index_space(q, k)
                    buckets = pipeline.build_buckets_gpu(idx_k, config)
                    scale = q.shape[-1] ** -0.5
                    gen = torch.Generator(device="cpu").manual_seed(args.seed + ctx + 7)
                    ragged = torch.sort(torch.randint(
                        1, ctx, (args.ragged_rows,), generator=gen)).values.to(q.device)
                    positions_all = torch.arange(ctx, device=q.device)
                    tensors = (q, k, v, idx_q, idx_k, buckets)
                    if lane == "decode":
                        shapes = [
                            ("decode_b1", positions_all[-1:], idx_q[-1:], q[:, -1:]),
                            ("decode_ragged", ragged, idx_q[ragged.long()],
                             q[0, ragged.long()].unsqueeze(0)),
                        ]
                    else:
                        shapes = [("prefill", positions_all, idx_q, q)]
                except Exception as e:  # noqa: BLE001
                    row["status"] = "OOM" if is_oom(e) else "FAILED"
                    row["error"] = f"setup: {type(e).__name__}: {str(e)[:300]}"
                    recover()
                    if tensors:
                        del tensors
                    recover()
                    flush()
                    if lane == "prefill":
                        prefill_lane_open = False
                    continue

                lane_failed = False
                for phase, positions, idx_rows, q_read in shapes:
                    prow = {"phase": phase, "rows": int(positions.shape[0])}
                    row["phases"].append(prow)
                    try:
                        # ---- correctness gate
                        oracle_mode = None
                        want = None
                        if not ref_skipped:
                            try:
                                want = pipeline.flat_select_reference(
                                    idx_rows, positions, buckets, config)
                                oracle_mode = "monolithic_reference"
                            except Exception as e:  # noqa: BLE001
                                if not is_oom(e):
                                    raise
                                recover()
                        if want is None:
                            want = chunked_oracle(idx_rows, positions, buckets, config,
                                                  args.chunk_rows)
                            oracle_mode = (oracle_mode or "") + "chunked_reference_oracle"
                        os.environ["LSA_SELECT_IMPL"] = "fused"
                        try:
                            got = pipeline.flat_select(idx_rows, positions, buckets, config)
                        finally:
                            os.environ.pop("LSA_SELECT_IMPL", None)
                        prow["oracle_mode"] = oracle_mode
                        prow["fused_exact"] = bool(
                            torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))
                        del got, want
                        recover()

                        # ---- determinism (3 repeats, fused select)
                        d0 = None
                        prow["fused_deterministic"] = True
                        for _ in range(3):
                            os.environ["LSA_SELECT_IMPL"] = "fused"
                            try:
                                d = pipeline.flat_select(idx_rows, positions, buckets, config)
                            finally:
                                os.environ.pop("LSA_SELECT_IMPL", None)
                            if d0 is None:
                                d0 = (d[0].clone(), d[1].clone())
                            elif not (torch.equal(d0[0], d[0]) and torch.equal(d0[1], d[1])):
                                prow["fused_deterministic"] = False
                        del d0
                        recover()

                        # ---- timing per stage
                        for stage_kind in ["select", "select_read"]:
                            entry = {"stage": stage_kind}
                            torch.cuda.reset_peak_memory_stats()
                            fn_fused = (select_fn("fused", idx_rows, positions, buckets, config)
                                        if stage_kind == "select" else
                                        select_read_fn("fused", q_read, k, v, idx_rows,
                                                       positions, buckets, config, scale))
                            fn_ref = (select_fn("reference", idx_rows, positions, buckets, config)
                                      if stage_kind == "select" else
                                      select_read_fn("reference", q_read, k, v, idx_rows,
                                                     positions, buckets, config, scale))
                            fn_fused()
                            sync()
                            t0 = time.perf_counter()
                            fn_fused()
                            sync()
                            entry["fused_first_call_ms"] = (time.perf_counter() - t0) * 1000
                            if ref_skipped:
                                fused_s = timed_single(fn_fused, args.warmup, args.reps)
                                ref_s = None
                                entry["mode"] = "single_arm_reference_skipped_projection"
                            else:
                                try:
                                    fn_ref()
                                    ref_s, fused_s = timed_pair(fn_ref, fn_fused,
                                                                args.warmup, args.reps)
                                    entry["mode"] = "interleaved_pair"
                                except Exception as e:  # noqa: BLE001
                                    if not is_oom(e):
                                        raise
                                    recover()
                                    fused_s = timed_single(fn_fused, args.warmup, args.reps)
                                    ref_s = None
                                    entry["mode"] = "single_arm_fused_after_reference_oom"
                            entry["reference"] = summarize(ref_s) if ref_s else None
                            entry["fused"] = summarize(fused_s)
                            if ref_s:
                                entry["speedup_p50"] = (statistics.median(ref_s) /
                                                        statistics.median(fused_s))
                            entry["launches_reference"] = (
                                launches_per_iter(fn_ref) if ref_s else None)
                            entry["launches_fused"] = launches_per_iter(fn_fused)
                            entry["peak_alloc_mib"] = round(
                                torch.cuda.max_memory_allocated() / 2**20, 1)
                            entry["peak_reserved_mib"] = round(
                                torch.cuda.max_memory_reserved() / 2**20, 1)
                            prow.setdefault("stages", {})[stage_kind] = entry
                            ref_p50 = entry["reference"]["p50_ms"] if entry["reference"] else float("nan")
                            print(f"[{dtype_name} {ctx} {phase} {stage_kind}] "
                                  f"{entry['mode']} ref {ref_p50:.3f} fused "
                                  f"{entry['fused']['p50_ms']:.3f} peak "
                                  f"{entry['peak_alloc_mib']:.0f}MiB "
                                  f"({entry.get('speedup_p50', float('nan')):.3f}x)", flush=True)
                            recover()
                    except Exception as e:  # noqa: BLE001
                        prow["status"] = "OOM" if is_oom(e) else "FAILED"
                        prow["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                        recover()

                    if "status" not in prow:
                        prow["status"] = (
                            "PASS" if prow.get("fused_exact") and prow.get("fused_deterministic")
                            and all(s.get("fused") for s in prow.get("stages", {}).values())
                            else "FAILED")
                    if prow["status"] != "PASS":
                        lane_failed = True
                    row["mem_after"] = mem_state()
                    flush()

                row["status"] = ("FAILED" if lane_failed else
                                 ("OOM" if all(p.get("status") == "OOM"
                                               for p in row["phases"]) else "PASS"))
                del tensors, q, k, v, idx_q, idx_k, buckets
                recover()
                flush()
                if lane == "prefill" and row["status"] == "OOM":
                    prefill_lane_open = False
                    stop_reason = stop_reason or f"prefill lane OOM at ctx={ctx}"

    receipt["stop_reason"] = stop_reason
    receipt["summary"] = {
        "steps_total": len(receipt["rows"]),
        "pass": sum(1 for r in receipt["rows"] if r.get("status") == "PASS"),
        "blocked": sum(1 for r in receipt["rows"] if r.get("status") == "BLOCKED_RESOURCE"),
        "oom_or_failed": sum(1 for r in receipt["rows"]
                             if r.get("status") in ("OOM", "FAILED")),
    }
    flush()
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}\nsha256 {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
