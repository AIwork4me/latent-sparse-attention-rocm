#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/third_party/latent-sparse-attention"
# Default to the REVIEWED pin in tracking/UPSTREAM_BASELINE.json, never floating
# upstream main: CI must test the SHA the campaign evidence was produced against.
# UPSTREAM_REF overrides explicitly (e.g. an approved compatibility-update run);
# "main" is only the bootstrap default when no reviewed pin has been recorded yet.
PIN_FILE="$ROOT/tracking/UPSTREAM_BASELINE.json"
if [[ -n "${UPSTREAM_REF:-}" ]]; then
  REF="$UPSTREAM_REF"
elif [[ -f "$PIN_FILE" ]]; then
  REF="$(python3 -c "import json;print(json.load(open('$PIN_FILE'))['reviewed_head_sha'])")"
else
  REF="main"
fi
mkdir -p "$ROOT/third_party"
if [[ ! -d "$DIR/.git" ]]; then git clone https://github.com/Prefrontal-AI/latent-sparse-attention.git "$DIR"; fi
git -C "$DIR" fetch --tags --prune origin
if [[ "$REF" == "main" ]]; then REF="origin/main"; fi
git -C "$DIR" checkout --detach "$(git -C "$DIR" rev-parse "${REF}^{commit}")"
git -C "$DIR" rev-parse HEAD > "$ROOT/third_party/upstream.sha"
echo "Pinned upstream SHA $(cat "$ROOT/third_party/upstream.sha")"
echo "Do NOT run uv sync --locked in the upstream checkout: it pins CUDA packages."
