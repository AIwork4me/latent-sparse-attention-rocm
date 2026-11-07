"""Independent receipt and benchmark auditor (F12).

This module trusts **nothing** the adapter asserted: ``status``,
``correctness_pass``, reported ``speedup``, sample counts, provenance and any
derived table are recomputed or cross-checked from the raw ``stages`` rows, the
``gate_manifest`` and the pinned upstream SHA. A receipt is APPROVED only when
every structural, provenance, gate and recomputation check passes; any single
finding REJECTS it. Callers (render_tables, CI, reviews) must treat a
non-APPROVED receipt as unusable for any claim.

Design rules:
- SHA checks establish *integrity within the saved bundle*; trusted CI
  artifacts / repo SHAs establish *provenance*. The auditor never asserts
  external truth from a hash alone - expected SHAs are inputs.
- Comparison rows are matched to raw stage rows by identity keys
  (method/phase/stage/context/dtype); missing, duplicate or ambiguous
  matches are findings, never guesses.
- Sparse-budget fields (top_k/top_p/window/sink) must be consistent within a
  comparison's candidate row and are reported per comparison so methods with
  different budgets stay distinguishable.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from lsa_rocm_lab.gate_manifest import REQUIRED_GATE_IDS, describe, verdict_of

SCHEMA_VERSION = "kernel-receipt/1"
REL_TOL = 1e-9
RUN_ID_RE = re.compile(r"^(\d{8}T\d{6}Z(-adapter)?|gh-\d+|rep\d+-\d{8}T\d{6}Z)$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DTYPES = ("fp16", "bf16")
PHASES = ("prefill", "decode")
SPARSE_METHODS = ("block_sparse", "lsa_flat", "lsa_tree")


class Finding:
    """One audit finding; any finding rejects the receipt."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail

    def __repr__(self):
        return f"[{self.code}] {self.detail}"

    def to_dict(self):
        return {"code": self.code, "detail": self.detail}


class AuditReport:
    def __init__(self, source: str, findings: list[Finding], recomputed: dict):
        self.source = source
        self.findings = findings
        self.recomputed = recomputed

    @property
    def verdict(self) -> str:
        return "APPROVED" if not self.findings else "REJECTED"

    def to_dict(self) -> dict:
        return {"source": str(self.source), "schema": SCHEMA_VERSION,
                "verdict": self.verdict,
                "findings": [f.to_dict() for f in self.findings],
                "recomputed": self.recomputed}

    def summary(self) -> str:
        lines = [f"{self.source}: {self.verdict} ({len(self.findings)} finding(s))"]
        lines += [f"  {f!r}" for f in self.findings]
        return "\n".join(lines)


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value > 0


def _close(actual, expected) -> bool:
    return math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=0.0)


def _scan_nonfinite(node, path: str, findings: list[Finding]):
    """Any NaN/Inf anywhere in the receipt is evidence corruption."""
    if isinstance(node, dict):
        for key, value in node.items():
            _scan_nonfinite(value, f"{path}.{key}", findings)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_nonfinite(value, f"{path}[{index}]", findings)
    elif isinstance(node, float) and not math.isfinite(node):
        findings.append(Finding("NAN_INF", f"non-finite number at {path}: {node}"))


def _check_structure(data: dict, findings: list[Finding]) -> None:
    local: list[Finding] = []
    required = {
        "kind": str, "status": str, "correctness_pass": bool, "gates_skipped": bool,
        "run_id": str, "repo_head_sha": str, "upstream_sha": str,
        "gpu_arch": str, "gpu_name": str, "torch": str,
        "command": list, "gates": list, "stages": list, "comparisons": list, "env": dict,
    }
    for field, typ in required.items():
        if not isinstance(data.get(field), typ):
            local.append(Finding("SCHEMA", f"field '{field}' missing or not {typ.__name__}"))
    findings.extend(local)
    if local:
        return
    if data["kind"] != "lsa_rocm_kernel":
        findings.append(Finding("SCHEMA", f"kind '{data['kind']}' is not a kernel receipt"))
    if not data["gpu_arch"].startswith("gfx1151"):
        findings.append(Finding("DEVICE", f"gpu_arch '{data['gpu_arch']}' is not gfx1151"))
    if not RUN_ID_RE.match(data["run_id"]):
        findings.append(Finding("SCHEMA",
                                f"run_id '{data['run_id']}' is not a timestamped or CI traceable id"))
    for sha_field in ("repo_head_sha", "upstream_sha"):
        if not SHA_RE.match(data[sha_field]):
            findings.append(Finding("PROVENANCE",
                                    f"{sha_field} '{data[sha_field]}' is not a 40-hex SHA"))
    if not data["command"]:
        findings.append(Finding("SCHEMA", "empty command - exact CLI not recorded"))
    env = data["env"]
    for field in ("dtype", "reps", "warmup", "seed"):
        if field not in env:
            findings.append(Finding("SCHEMA", f"env.{field} not recorded"))
    if env.get("dtype") not in DTYPES:
        findings.append(Finding("SCHEMA", f"env.dtype '{env.get('dtype')}' not in {DTYPES}"))
    if not isinstance(env.get("reps"), int) or env.get("reps", 0) < 10:
        findings.append(Finding("SCHEMA", "env.reps below the 10-sample evidence bar"))


def _check_gates(data: dict, findings: list[Finding]) -> None:
    if data.get("gates_skipped") is True:
        findings.append(Finding("UNVERIFIED_GATES",
                                "gates_skipped=true: benchmark cannot be treated as verified"))
    if data.get("status") == "UNVERIFIED":
        findings.append(Finding("UNVERIFIED_GATES", "receipt status is UNVERIFIED"))
    if data.get("status") != "PASS":
        findings.append(Finding("STATUS_NOT_PASS", f"receipt status '{data.get('status')}'"))
    if data.get("correctness_pass") is not True:
        findings.append(Finding("UNVERIFIED_GATES", "correctness_pass is not true"))
    verdict = verdict_of(data.get("gates") or [])
    if not verdict["ok"]:
        findings.append(Finding("GATE_MANIFEST", "gate manifest: " + describe(verdict)))
    # the decode-baseline parity gates are the anchor of every decode claim
    gate_ids = {g.get("gate_id") for g in data.get("gates") or []}
    dtype = (data.get("env") or {}).get("dtype")
    if dtype in DTYPES and f"G:{dtype}" not in gate_ids:
        findings.append(Finding("GATE_MANIFEST",
                                f"dense-decode parity gate G:{dtype} absent for dtype {dtype}"))


def _stage_key(row: dict) -> tuple:
    return (row.get("method"), row.get("phase"), row.get("stage"), row.get("context"),
            row.get("dtype"))


def _check_stage_rows(data: dict, findings: list[Finding]) -> dict:
    """Validate raw rows; return them keyed for comparison recomputation."""
    rows_by_key: dict[tuple, list[dict]] = {}
    identity = {}
    for index, row in enumerate(data.get("stages") or []):
        path = f"stages[{index}]"
        if not isinstance(row, dict):
            findings.append(Finding("SCHEMA", f"{path}: not an object"))
            continue
        row_bad = [field for field in ("method", "phase", "p50_ms", "p95_ms", "samples",
                                       "context", "dtype") if field not in row]
        for field in row_bad:
            findings.append(Finding("SCHEMA", f"{path}: missing '{field}'"))
        if row_bad:
            continue
        if row["phase"] not in PHASES:
            findings.append(Finding("SCHEMA", f"{path}: phase '{row['phase']}' unknown"))
        for stat in ("p50_ms", "p95_ms"):
            if not _finite_positive(row.get(stat)):
                findings.append(Finding("TIMING", f"{path}: {stat}={row.get(stat)!r} not finite positive"))
        for stat in ("mad_ms", "min_ms", "max_ms"):
            value = row.get(stat)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                findings.append(Finding("TIMING", f"{path}: {stat}={value!r} not finite non-negative"))
        if not isinstance(row.get("samples"), int) or row.get("samples", 0) < 10:
            findings.append(Finding("SAMPLES", f"{path}: samples={row.get('samples')} < 10"))
        if row.get("min_ms") is not None and row.get("max_ms") is not None \
                and not (row["min_ms"] <= row["p50_ms"] <= row["max_ms"]):
            findings.append(Finding("TIMING", f"{path}: p50 {row['p50_ms']} outside [{row['min_ms']}, {row['max_ms']}]"))
        # identity consistency: one seed/dtype/geometry per receipt
        for field in ("seed", "dtype", "heads", "kv_heads", "dim"):
            value = row.get(field)
            if field == "seed" and not isinstance(value, int):
                findings.append(Finding("SCHEMA", f"{path}: seed not an integer"))
                continue
            if field == "dtype" and value not in DTYPES:
                findings.append(Finding("SCHEMA", f"{path}: dtype '{value}' unknown"))
                continue
            if field in ("heads", "kv_heads", "dim") and (not isinstance(value, int) or value <= 0):
                findings.append(Finding("SCHEMA", f"{path}: {field}={value!r} not positive"))
                continue
            if field in identity and identity[field] != value:
                findings.append(Finding(f"MIXED_{field.upper()}",
                                        f"{path}: {field}={value!r} but earlier rows said {identity[field]!r}"))
            else:
                identity[field] = value
        # sparse methods must record their budget
        if row.get("method") in SPARSE_METHODS:
            for field in ("top_k", "top_p", "window", "sink"):
                if not isinstance(row.get(field), int):
                    findings.append(Finding("SCHEMA", f"{path}: sparse row missing integer {field}"))
        # the repaired dense-decode baseline reads the FULL causal history at Q=1
        if row.get("method") == "dense_sdpa" and row.get("phase") == "decode" \
                and row.get("query_len") != 1:
            findings.append(Finding("DECODE_BASELINE",
                                    f"{path}: dense decode query_len={row.get('query_len')!r}, "
                                    "must be 1 (full-history single-token decode)"))
        rows_by_key.setdefault(_stage_key(row), []).append(row)

    for key, rows in rows_by_key.items():
        if len(rows) > 1:
            findings.append(Finding("DUPLICATE_ROW",
                                    f"ambiguous stage rows for {key}: {len(rows)} entries"))
    return rows_by_key


def _recompute_comparisons(data: dict, rows_by_key: dict, findings: list[Finding]) -> list[dict]:
    """Every comparison must be rebuildable from raw rows; report what we computed."""
    recomputed = []
    for index, comp in enumerate(data.get("comparisons") or []):
        path = f"comparisons[{index}]"
        context, dtype, phase, method = (comp.get("context"), comp.get("dtype"),
                                         comp.get("phase"), comp.get("method"))
        dense_rows = rows_by_key.get(("dense_sdpa", phase, None, context, dtype), [])
        if len(dense_rows) != 1:
            findings.append(Finding("MISSING_BASELINE" if not dense_rows else "AMBIGUOUS_BASELINE",
                                    f"{path}: {len(dense_rows)} dense baseline rows for "
                                    f"context={context} dtype={dtype} phase={phase}"))
            continue
        cand_rows = rows_by_key.get((method, phase, comp.get("stage", "select+read"), context, dtype), [])
        if len(cand_rows) != 1:
            findings.append(Finding("MISSING_CANDIDATE" if not cand_rows else "AMBIGUOUS_CANDIDATE",
                                    f"{path}: {len(cand_rows)} candidate rows for method={method} "
                                    f"stage={comp.get('stage')} context={context} phase={phase}"))
            continue
        dense, cand = dense_rows[0], cand_rows[0]
        row = {"context": context, "dtype": dtype, "phase": phase, "method": method,
               "stage": comp.get("stage", "select+read"),
               "baseline_p50_ms": dense["p50_ms"], "candidate_p50_ms": cand["p50_ms"]}
        if not _close(comp.get("baseline_ms", float("nan")), dense["p50_ms"]):
            findings.append(Finding("BASELINE_MISMATCH",
                                    f"{path}: baseline_ms={comp.get('baseline_ms')} != dense row "
                                    f"p50 {dense['p50_ms']} (swapped or edited baseline?)"))
        if not _close(comp.get("candidate_ms", float("nan")), cand["p50_ms"]):
            findings.append(Finding("CANDIDATE_MISMATCH",
                                    f"{path}: candidate_ms={comp.get('candidate_ms')} != candidate "
                                    f"row p50 {cand['p50_ms']}"))
        speedup = dense["p50_ms"] / cand["p50_ms"]
        row["speedup"] = speedup
        if not _close(comp.get("speedup", float("nan")), speedup):
            findings.append(Finding("SPEEDUP_MISMATCH",
                                    f"{path}: reported speedup={comp.get('speedup')} but raw rows "
                                    f"recompute to {speedup}"))
        if comp.get("samples") != min(dense["samples"], cand["samples"]):
            findings.append(Finding("SAMPLES",
                                    f"{path}: samples={comp.get('samples')} != "
                                    f"min({dense['samples']}, {cand['samples']})"))
        # the honesty marker: select+read timing on a prebuilt index is NOT an E2E claim
        if "select+read" not in str(comp.get("note", "")) and comp.get("stage") != "select+read":
            findings.append(Finding("MISLABELED_SCOPE",
                                    f"{path}: neither stage nor note says select+read on a "
                                    "prebuilt index - a partial timing cannot masquerade as full LSA"))
        for field in ("top_k", "top_p", "window", "sink"):
            if field in cand:
                row[field] = cand[field]
        recomputed.append(row)
    if not data.get("comparisons"):
        findings.append(Finding("NO_COMPARISONS", "receipt carries no comparisons"))
    return recomputed


def _check_bundle_checksum(receipt_path: Path, findings: list[Finding]) -> None:
    """If the bundle ships a SHA256SUMS, the receipt line must still match."""
    for candidate in (receipt_path.parent / "SHA256SUMS", receipt_path.parent.parent / "SHA256SUMS"):
        if not candidate.is_file():
            continue
        covered = False
        for line in candidate.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and (candidate.parent / parts[1]).resolve() == receipt_path.resolve():
                covered = True
                digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                if digest != parts[0]:
                    findings.append(Finding("CHECKSUM_MISMATCH",
                                            f"{parts[1]}: SHA256SUMS says {parts[0]}, file hashes {digest}"))
        if covered:
            return


def audit_receipt(data: dict, source: str = "<memory>",
                  expected_upstream_sha: str | None = None,
                  expected_repo_sha: str | None = None,
                  seen_run_ids: set[str] | None = None) -> AuditReport:
    """Audit one parsed kernel receipt; REJECTED on any finding."""
    findings: list[Finding] = []
    recomputed: dict = {"schema": SCHEMA_VERSION}

    _scan_nonfinite(data, "", findings)
    _check_structure(data, findings)
    if not findings:
        _check_gates(data, findings)
        if expected_upstream_sha is not None and data.get("upstream_sha") != expected_upstream_sha:
            findings.append(Finding("UPSTREAM_SHA_MISMATCH",
                                    f"receipt upstream {data.get('upstream_sha')} != pinned "
                                    f"{expected_upstream_sha}"))
        if expected_repo_sha is not None and data.get("repo_head_sha") != expected_repo_sha:
            findings.append(Finding("REPO_SHA_MISMATCH",
                                    f"receipt repo HEAD {data.get('repo_head_sha')} != expected "
                                    f"{expected_repo_sha} (evidence produced by different code?)"))
        if seen_run_ids is not None:
            run_id = data.get("run_id")
            if run_id in seen_run_ids:
                findings.append(Finding("REPLAYED_RUN_ID",
                                        f"run_id '{run_id}' already audited in this set - replay"))
            seen_run_ids.add(run_id)
        rows_by_key = _check_stage_rows(data, findings)
        recomputed["comparisons"] = _recompute_comparisons(data, rows_by_key, findings)
        recomputed["gate_count"] = len(data.get("gates") or [])
        recomputed["required_gate_count"] = len(REQUIRED_GATE_IDS)
    return AuditReport(source, findings, recomputed)


def audit_file(path: Path, **kwargs) -> AuditReport:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except Exception as error:
        return AuditReport(str(path), [Finding("UNREADABLE", repr(error))], {})
    report = audit_receipt(data, source=str(path), **kwargs)
    _check_bundle_checksum(path, report.findings)
    return report


def audit_files(paths, expected_upstream_sha: str | None = None,
                expected_repo_sha: str | None = None) -> list[AuditReport]:
    """Audit a set, rejecting replayed run ids across the whole set."""
    seen: set[str] = set()
    return [audit_file(Path(p), expected_upstream_sha=expected_upstream_sha,
                       expected_repo_sha=expected_repo_sha, seen_run_ids=seen)
            for p in paths]
