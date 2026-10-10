#!/usr/bin/env bash
# Launch from any directory; dataset/output paths are relative to the caller.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "${PYTHON:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  "$script_dir/train.py" "$@"
