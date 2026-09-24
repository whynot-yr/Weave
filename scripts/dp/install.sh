#!/usr/bin/env bash
# Separate from WEAVE's existing simulator installer: no PPO dependency changes.
set -euo pipefail
DP_WEAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_LEROBOT_ROOT="${1:-$(dirname "$DP_WEAVE_ROOT")/lerobot}"
DP_ENV_ROOT="${DP_ENV_ROOT:-$DP_WEAVE_ROOT/.venv-dp}"
if [ ! -f "$DP_LEROBOT_ROOT/pyproject.toml" ]; then
    echo "LeRobot source not found: $DP_LEROBOT_ROOT" >&2
    exit 1
fi
if [ ! -x "$DP_ENV_ROOT/bin/python" ]; then
    uv venv --python 3.12 "$DP_ENV_ROOT"
fi
uv pip install --python "$DP_ENV_ROOT/bin/python" \
    -e "$DP_LEROBOT_ROOT[dataset,training,diffusion]" \
    -e "$DP_WEAVE_ROOT/source/weave_data" pytest ruff
"$DP_ENV_ROOT/bin/python" -c 'import torch, lerobot; print("torch:", torch.__version__, "lerobot:", lerobot.__version__, "CUDA:", torch.cuda.is_available())'
