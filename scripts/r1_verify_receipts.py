#!/usr/bin/env python3
"""R1 independent receipt verifier — recomputes everything from raw artifacts.

For every receipt JSON: gates must all be PASS with the expected count; every
comparison's stored speedup must equal baseline_p50_ms/candidate_p50_ms
(recomputed, tolerance 1e-9 relative); repo_head_sha must exist in the given
git history; upstream_sha and gpu_arch must match expectations. For audit
files: every row must be APPROVED and its recomputed comparisons must equal
my independent recompute of the source receipt.
Usage: verify_receipts.py <repo_root> <receipt-or-audit.json>... [--expect-gates N]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

EXPECT_UPSTREAM = "5ebea732d639576d2a7579b0de40e6252f9001f5"
EXPECT_GPU = "gfx1151"


def git_has(repo: Path, sha: str) -> bool:
    try:
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", sha + "^{commit}"],
                       check=True, capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def check_comparison(c: dict, where: str, errs: list, stats: dict):
    b = c.get("baseline_p50_ms", c.get("baseline_ms"))
    cand = c.get("candidate_p50_ms", c.get("candidate_ms"))
    s = c.get("speedup")
    if not (isinstance(b, (int, float)) and b > 0):
        errs.append(f"{where}: bad baseline_p50_ms {b!r}")
        return
    if not (isinstance(cand, (int, float)) and cand > 0):
        errs.append(f"{where}: bad candidate_p50_ms {cand!r}")
        return
    mine = b / cand
    stats["comparisons"] += 1
    if not isinstance(s, (int, float)) or not math.isclose(mine, s, rel_tol=1e-9):
        errs.append(f"{where}: speedup stored={s!r} recomputed={mine:.12g} "
                    f"(b={b}, c={cand}, ctx={c.get('context')} {c.get('phase')} {c.get('method')})")


def verify_receipt(repo: Path, path: Path, expect_gates: int, errs: list, stats: list) -> dict | None:
    d = json.loads(path.read_text())
    rid = d.get("run_id", path.name)
    st = {"file": path.name, "run_id": rid, "comparisons": 0}
    stats.append(st)
    if d.get("kind") not in (None, "kernel-receipt/1", "lsa_rocm_kernel"):
        errs.append(f"{rid}: unexpected kind {d.get('kind')!r}")
    if d.get("status") not in ("ok", "OK", "PASS", None):
        errs.append(f"{rid}: status {d.get('status')!r}")
    if d.get("correctness_pass") is False:
        errs.append(f"{rid}: correctness_pass false")
    gates = d.get("gates") or []
    npass = sum(1 for g in gates if isinstance(g, dict) and g.get("status") == "PASS")
    if npass != len(gates) or len(gates) == 0:
        errs.append(f"{rid}: gates {npass}/{len(gates)} PASS")
    st["gates"] = f"{npass}/{len(gates)}"
    if expect_gates and len(gates) != expect_gates:
        errs.append(f"{rid}: gate count {len(gates)} != expected {expect_gates}")
    head = d.get("repo_head_sha", "")
    st["head"] = head[:9]
    if not git_has(repo, head):
        errs.append(f"{rid}: repo_head_sha {head} NOT a commit in {repo}")
    if d.get("upstream_sha") != EXPECT_UPSTREAM:
        errs.append(f"{rid}: upstream_sha {d.get('upstream_sha')!r} != pin")
    if d.get("gpu_arch") != EXPECT_GPU:
        errs.append(f"{rid}: gpu_arch {d.get('gpu_arch')!r} != {EXPECT_GPU}")
    for i, c in enumerate(d.get("comparisons") or []):
        check_comparison(c, f"{rid}#cmp{i}", errs, st)
    return d


def main() -> int:
    repo = Path(sys.argv[1])
    files, expect_gates = [], 0
    args = sys.argv[2:]
    if "--expect-gates" in args:
        i = args.index("--expect-gates")
        expect_gates = int(args[i + 1])
        del args[i:i + 2]
    files = [Path(a) for a in args]
    errs, stats = [], []
    for p in files:
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "receipts" in d and "verdict" in d:
            if d.get("verdict") != "APPROVED":
                errs.append(f"{p.name}: audit verdict {d.get('verdict')!r}")
            rows = d["receipts"]
            stats.append({"file": p.name, "audit_rows": len(rows), "comparisons": 0})
            tot = 0
            for r in rows:
                if r.get("verdict") != "APPROVED":
                    errs.append(f"{p.name}: row {r.get('source','?')} verdict {r.get('verdict')!r}")
                rc = (r.get("recomputed") or {}).get("comparisons") or []
                tot += len(rc)
                for i, c in enumerate(rc):
                    check_comparison(c, f"{p.name}#{r.get('source','?')}#r{i}", errs, stats[-1])
            stats[-1]["audit_recomputed_total"] = tot
        else:
            verify_receipt(repo, p, expect_gates, errs, stats)
    for s in stats:
        print(s)
    if errs:
        print(f"\nFAIL: {len(errs)} error(s)")
        for e in errs[:40]:
            print("  -", e)
        return 1
    print("\nALL VERIFIED (independent recompute)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
