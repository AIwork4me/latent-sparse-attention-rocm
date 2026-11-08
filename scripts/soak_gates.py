#!/usr/bin/env python3
"""M6 (overnight_v1) - bounded soak: repeat the 17-gate manifest until a wall-clock
budget, watching for drift, failures and allocator growth. Fail-closed: two consecutive
non-green rounds stop the soak; a final leak check compares reserved memory to round 1.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--env", default="", help="extra KEY=VAL for the gates subprocess")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    py = "/home/amd/Desktop/lsa-rocm-eval-starter/.venv/bin/python"
    deadline = time.monotonic() + args.minutes * 60
    import torch  # after arg parse so --help works CPU-only
    rounds, consecutive_fail, first_reserved = 0, 0, None
    log = []
    while time.monotonic() < deadline:
        rounds += 1
        t0 = time.perf_counter()
        env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
        if args.env:
            for kv in args.env.split(";"):
                k, _, v = kv.partition("=")
                env[k] = v
        r = subprocess.run([py, "-m", "lsa_rocm_lab.port.gates"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=600)
        wall = round(time.perf_counter() - t0, 1)
        n_pass = r.stdout.count('"status": "PASS"')
        n_all = r.stdout.count('"status"')
        ok = r.returncode == 0 and n_all == 17 and n_pass == 17
        torch.cuda.empty_cache()
        reserved = round(torch.cuda.memory_reserved() / 2**20, 1)
        free = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
        if first_reserved is None:
            first_reserved = reserved
        temp = ""
        try:
            temp = subprocess.run(["rocm-smi", "--showtemp"], capture_output=True,
                                  text=True, timeout=10).stdout
            temp = next((l.split(":", 1)[1].strip() for l in temp.splitlines()
                         if "Temperature" in l), "")
        except Exception:  # noqa: BLE001
            pass
        log.append({"round": rounds, "ok": ok, "gates_pass": n_pass, "gates_total": n_all,
                    "wall_s": wall, "reserved_mib_after_cache_empty": reserved,
                    "vram_free_gib": free, "temp": temp, "rc": r.returncode})
        print(f"[round {rounds}] ok={ok} {n_pass}/{n_all} {wall}s reserved={reserved}MiB "
              f"free={free}GiB {temp}", flush=True)
        consecutive_fail = 0 if ok else consecutive_fail + 1
        if consecutive_fail >= 2:
            log[-1]["stop"] = "two consecutive non-green rounds"
            break
    leak_ratio = (log[-1]["reserved_mib_after_cache_empty"] / first_reserved
                  if first_reserved else 0)
    verdict = {
        "rounds": rounds, "green_rounds": sum(1 for e in log if e["ok"]),
        "stopped_early": "stop" in log[-1], "leak_ratio_vs_round1": round(leak_ratio, 2),
        "leak_ok": leak_ratio < 1.3,
        "verdict": "PASS" if all(e["ok"] for e in log) and leak_ratio < 1.3 else "FAIL",
        "minutes": args.minutes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"run_id": datetime.now(timezone.utc).isoformat(), "rounds_log": log, **verdict},
        indent=2))
    print(json.dumps(verdict), flush=True)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
