#!/usr/bin/env python3
"""Static negative checks that keep untrusted PR code off the persistent GPU runner.

Enforced invariants (docs/SECURITY_RUNNER_POLICY.md):

  R1  No workflow triggered by ``pull_request`` or ``pull_request_target`` may
      run on a self-hosted runner. This repository is public; fork PR code is
      hostile and must only ever execute on GitHub-hosted ephemeral runners.
  R2  ``pull_request_target`` is prohibited entirely in this repository - it
      grants fork PRs read access to repo secrets and is too easy to misuse.
  R3  ``workflow_run``-triggered workflows must not reach self-hosted runners
      either (they can be tripped by public PR activity).
  R4  Every ``uses:`` step must be pinned to a full 40-hex commit SHA (or a
      ``docker://`` image pinned by ``@sha256:`` digest) - no floating tags.
  R5  Self-hosted jobs must not reference ``${{ secrets.* }}``; the persistent
      runner keeps no repo secrets and must never be handed one.

Exit code 0 only when every workflow file passes all five rules.
"""

import re
import sys
from pathlib import Path

import yaml

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UNTRUSTED_TRIGGERS = ("pull_request", "pull_request_target", "workflow_run")


def _triggers(workflow: dict) -> set:
    """Normalize the `on:` key (YAML parses bare `on` as boolean True)."""
    on = workflow.get("on", workflow.get(True))
    if on is None:
        return set()
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    return set(on.keys() or ())


def _jobs(workflow: dict) -> dict:
    jobs = workflow.get("jobs") or {}
    return jobs if isinstance(jobs, dict) else {}


def _is_self_hosted(job: dict) -> bool:
    runs_on = job.get("runs-on")
    if isinstance(runs_on, list):
        return any(str(label) == "self-hosted" for label in runs_on)
    return str(runs_on) == "self-hosted"


def _step_uses(job: dict) -> list[str]:
    return [str(step["uses"]) for step in job.get("steps") or [] if step.get("uses")]


def _walk_node(node):
    """Yield every string value nested in a parsed YAML node."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _walk_node(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_node(item)


def check_workflow(path: Path) -> list[str]:
    findings: list[str] = []
    workflow = yaml.safe_load(path.read_text()) or {}
    name = workflow.get("name") or path.name
    triggers = _triggers(workflow)
    jobs = _jobs(workflow)

    for job_id, job in jobs.items():
        self_hosted = _is_self_hosted(job)
        where = f"{path.name}:{name}/{job_id}"

        # R1/R3: untrusted-event triggers must never reach a self-hosted runner
        for trigger in sorted(triggers & set(UNTRUSTED_TRIGGERS)):
            if self_hosted:
                findings.append(
                    f"R1 {where}: trigger '{trigger}' routes to a self-hosted runner - "
                    "untrusted PR code must never execute on the persistent GPU host"
                )
        # R2: pull_request_target prohibited outright
        if "pull_request_target" in triggers:
            findings.append(f"R2 {where}: pull_request_target is prohibited in this repository")

        # R4: pinned actions only
        for uses in _step_uses(job):
            if uses.startswith("docker://"):
                if "@sha256:" not in uses:
                    findings.append(f"R4 {where}: docker action not digest-pinned: {uses}")
                continue
            ref = uses.split("@")[-1] if "@" in uses else ""
            if not SHA_RE.match(ref):
                findings.append(f"R4 {where}: action not pinned to a commit SHA: {uses}")

        # R5: no secrets for self-hosted jobs
        if self_hosted:
            for text in _walk_node(job):
                for match in re.findall(r"\$\{\{\s*secrets\.[A-Za-z0-9_]+", text):
                    findings.append(f"R5 {where}: self-hosted job references {match.strip()}")

    return findings


def check_all(workflows_dir: Path) -> list[str]:
    findings: list[str] = []
    files = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    if not files:
        return [f"R0 {workflows_dir}: no workflow files found - misplaced guard?"]
    for path in files:
        findings.extend(check_workflow(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1] / ".github" / "workflows"
    findings = check_all(root)
    for finding in findings:
        print(f"::error::{finding}")
        print(f"VIOLATION {finding}", file=sys.stderr)
    if findings:
        print(f"\nworkflow security check FAILED: {len(findings)} violation(s)")
        return 1
    print("workflow security check passed: no untrusted route to a self-hosted runner, "
          "all actions SHA-pinned, no secrets on self-hosted jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
