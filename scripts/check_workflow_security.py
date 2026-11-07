#!/usr/bin/env python3
"""Static negative checks that keep untrusted PR code off the persistent GPU runner.

Enforced invariants (docs/SECURITY_RUNNER_POLICY.md):

  R1  No workflow triggered by ``pull_request`` or ``pull_request_target`` may
      run outside GitHub-hosted ephemeral runners. This repository is public;
      fork PR code is hostile and must only ever execute on GitHub-hosted
      ephemeral runners.
  R2  ``pull_request_target`` is prohibited entirely in this repository - it
      grants fork PRs read access to repo secrets and is too easy to misuse.
  R3  ``workflow_run``-triggered workflows must not reach non-hosted runners
      either (they can be tripped by public PR activity).
  R4  Every ``uses:`` step must be pinned to a full 40-hex commit SHA (or a
      ``docker://`` image pinned by ``@sha256:`` digest) - no floating tags.
  R5  Non-hosted jobs must not be handed ``${{ secrets.* }}`` - including via
      workflow-level ``env:`` / ``defaults:`` (which apply to every job).
  R6  A ``workflow_call`` (reusable) workflow containing non-hosted jobs is
      rejected unless the file carries the reviewed-allowlist marker comment
      ``# guard-r6-allow: reviewed-reusable-workflow`` - composition must not
      let an untrusted caller reach the persistent host through a side file.

Runner classification is deliberately conservative (R1 adversarial review
2026-11-07): a job counts as NON-HOSTED unless every ``runs-on`` label matches
the GitHub-hosted allowlist (``ubuntu-*``, ``macos-*``, ``windows-*``). Custom
label sets, unknown labels, expressions such as ``${{ matrix.runner }}`` and a
missing ``runs-on`` are all treated as potentially persistent/self-hosted.

Exit code 0 only when every workflow file passes all six rules.
"""

import re
import sys
from pathlib import Path

import yaml

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UNTRUSTED_TRIGGERS = ("pull_request", "pull_request_target", "workflow_run")
HOSTED_LABEL_RE = re.compile(r"^(ubuntu|macos|windows)-[a-z0-9][a-z0-9.\-]*$")
R6_ALLOW_MARKER = "# guard-r6-allow: reviewed-reusable-workflow"
SECRET_RE = re.compile(r"\$\{\{\s*secrets\.[A-Za-z0-9_]+")


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
    """Conservative: anything not provably GitHub-hosted is persistent-host.

    Expressions (``${{ ... }}``), unknown/custom labels and a missing
    ``runs-on`` all count as non-hosted - matrix-injected runners and custom
    label sets must never slip past R1/R3/R5 (adversarial finding F1).
    """
    runs_on = job.get("runs-on")
    labels = runs_on if isinstance(runs_on, list) else [runs_on]
    if not labels:
        return True
    for label in labels:
        text = str(label)
        if "${{" in text or not HOSTED_LABEL_RE.match(text):
            return True
    return False


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


def _secret_refs(*nodes) -> list[str]:
    refs: list[str] = []
    for node in nodes:
        for text in _walk_node(node):
            refs.extend(m.rstrip() for m in SECRET_RE.findall(text))
    return refs


def check_workflow(path: Path) -> list[str]:
    findings: list[str] = []
    raw = path.read_text()
    workflow = yaml.safe_load(raw) or {}
    name = workflow.get("name") or path.name
    triggers = _triggers(workflow)
    jobs = _jobs(workflow)

    any_non_hosted = any(_is_self_hosted(job) for job in jobs.values() if isinstance(job, dict))

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        self_hosted = _is_self_hosted(job)
        where = f"{path.name}:{name}/{job_id}"

        # R1/R3: untrusted-event triggers must never reach a non-hosted runner
        for trigger in sorted(triggers & set(UNTRUSTED_TRIGGERS)):
            if self_hosted:
                findings.append(
                    f"R1 {where}: trigger '{trigger}' routes to a non-GitHub-hosted runner - "
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

        # R5: no secrets handed to non-hosted jobs (job-level scan)
        if self_hosted:
            for match in _secret_refs(job):
                findings.append(f"R5 {where}: non-hosted job references {match}")

    # R5 (file scope): workflow-level env:/defaults: apply to every job - if any
    # job is non-hosted, secrets there reach the persistent host (finding F2).
    if any_non_hosted:
        for match in _secret_refs(workflow.get("env"), workflow.get("defaults")):
            findings.append(
                f"R5 {path.name}:{name}: workflow-level env/defaults references {match} "
                "while a non-hosted job exists"
            )

    # R6: reusable workflows must not launder untrusted callers onto the
    # persistent host through composition (finding F1-b3).
    if "workflow_call" in triggers and any_non_hosted and R6_ALLOW_MARKER not in raw:
        findings.append(
            f"R6 {path.name}:{name}: workflow_call workflow with non-hosted jobs - "
            "composition bypass; add the reviewed allowlist marker or use trusted triggers"
        )

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
    print("workflow security check passed: no untrusted route to a persistent runner, "
          "all actions SHA-pinned, no secrets on non-hosted jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
