#!/usr/bin/env bash
# num_envs is PER RANK; two processes each own a simulator and student.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
torchrun_args=(--standalone --nnodes=1 --nproc-per-node=2)
if [[ -n "${DAGGER_TORCHRUN_LOG_DIR:-}" ]]; then
  torchrun_args+=(--log-dir "$DAGGER_TORCHRUN_LOG_DIR" --redirects 3 --tee 3)
fi
exec "${PYTHON:-python}" -m torch.distributed.run \
  "${torchrun_args[@]}" \
  "$script_dir/train.py" --distributed "$@"
