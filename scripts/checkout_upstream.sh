#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/third_party/latent-sparse-attention"
REF="${UPSTREAM_REF:-main}"
mkdir -p "$ROOT/third_party"
if [[ ! -d "$DIR/.git" ]]; then git clone https://github.com/Prefrontal-AI/latent-sparse-attention.git "$DIR"; fi
git -C "$DIR" fetch --tags --prune origin
if [[ "$REF" == "main" ]]; then REF="origin/main"; fi
git -C "$DIR" checkout --detach "$(git -C "$DIR" rev-parse "${REF}^{commit}")"
git -C "$DIR" rev-parse HEAD > "$ROOT/third_party/upstream.sha"
echo "Pinned upstream SHA $(cat "$ROOT/third_party/upstream.sha")"
echo "Do NOT run uv sync --locked in the upstream checkout: it pins CUDA packages."
