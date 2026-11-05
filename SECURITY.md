# Security Policy

## Supported versions

The `main` branch is the only supported line. Benchmark receipts record the
commit they were produced at.

## Reporting a vulnerability

Please report vulnerabilities privately by opening a GitHub security advisory
("Report a vulnerability" under the Security tab) rather than a public issue.
Include reproduction steps and, where relevant, the environment manifest from
`python -m lsa_rocm_lab.preflight`.

## Scope notes specific to this repository

- This project **executes code from the pinned upstream checkout**
  (`third_party/latent-sparse-attention`). The checkout is fetched over HTTPS
  from `https://github.com/Prefrontal-AI/latent-sparse-attention` and pinned by
  SHA. Do not run the harness against an untrusted checkout or a moved SHA.
- Never install upstream's CUDA-pinned dependencies (`uv sync --locked`) into a
  shared environment; use the isolated venv procedure in the README.
- The harness writes results and logs under `results/` only; it does not read
  credentials, tokens, or files outside the repository, the venv, and the
  upstream checkout. `scripts/upstream_watch.py` reads `GH_TOKEN` from the
  environment only when `--issue` is explicitly requested.
