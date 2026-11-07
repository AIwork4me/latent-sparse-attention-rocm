"""The required correctness-gate manifest - fail-closed evidence's single source.

A receipt is only claimable when its ``gates`` array carries *exactly* these ids,
each with verdict PASS. The adapter checks this before writing a PASS receipt and
``lsa_rocm_lab.run.validate_receipt`` re-checks it independently, so a skipped
suite (empty list), a truncated suite, a renamed gate or a failed verdict can
never yield ``correctness_pass=true`` anywhere in the chain.
"""

REQUIRED_GATE_IDS = [
    "A",
    "A2",
    "B1",
    "B2",
    "C",
    "D1:fp32",
    "D2:fp32",
    "D1:fp16",
    "D2:fp16",
    "D1:bf16",
    "D2:bf16",
    "E",
    "F",
    "G:fp32",
    "G:fp16",
    "G:bf16",
]


def verdict_of(gates: list[dict]) -> dict:
    """Classify a gate-result list against the manifest - no list is vacuously ok.

    Returns ``{ok, missing, unexpected, duplicate, failed}``; ``ok`` is True only
    for the exact required id set with every verdict PASS.
    """
    ids = [g.get("gate_id") for g in gates]
    seen: set[str] = set()
    duplicate = sorted({i for i in ids if i is not None and (i in seen or seen.add(i))})
    failed = sorted(g.get("gate_id") or "<no id>" for g in gates if g.get("status") != "PASS")
    missing = sorted(set(REQUIRED_GATE_IDS) - set(ids))
    unexpected = sorted(set(ids) - set(REQUIRED_GATE_IDS))
    ok = not (missing or unexpected or duplicate or failed)
    return {"ok": ok, "missing": missing, "unexpected": unexpected,
            "duplicate": duplicate, "failed": failed}


def describe(verdict: dict) -> str:
    """A human-readable reason for a failed manifest check."""
    parts = []
    if verdict["missing"]:
        parts.append("missing: " + ", ".join(verdict["missing"]))
    if verdict["unexpected"]:
        parts.append("unexpected: " + ", ".join(verdict["unexpected"]))
    if verdict["duplicate"]:
        parts.append("duplicate: " + ", ".join(verdict["duplicate"]))
    if verdict["failed"]:
        parts.append("not PASS: " + ", ".join(verdict["failed"]))
    return "; ".join(parts) or "gates match the manifest"
