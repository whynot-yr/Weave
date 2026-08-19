#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "$REPO_DIR")}"
VENV="$WORKSPACE/sim51"
PYTHON_VERSION=3.11
# pick up an existing install under any isaacsim* name
if [ -z "${ISAACSIM:-}" ]; then
    ISAACSIM="$WORKSPACE/isaacsim"
    for candidate in "$WORKSPACE"/isaacsim*; do
        if [ -x "$candidate/isaac-sim.sh" ]; then ISAACSIM="$candidate"; break; fi
    done
fi
ISAACSIM_VERSION=5.1.0
ISAACLAB="${ISAACLAB:-$WORKSPACE/IsaacLab}"
ISAACLAB_COMMIT=e17312889676ed229b986d56c9e0b23a01cf0ab7

echo "🚀 Installing g1_hoi_learning into $WORKSPACE"

# ---------------------------------------------------------------- prerequisites
MISSING=""
for cmd in curl g++ git nvidia-smi tar unzip; do
    command -v "$cmd" >/dev/null 2>&1 || MISSING="$MISSING $cmd"
done
if [ -n "$MISSING" ]; then
    echo "❌ Missing required commands:$MISSING" >&2
    exit 1
fi

if [ -z "${GCC_EXEC_PREFIX:-}" ]; then unset GCC_EXEC_PREFIX; fi

case "$(uname -m)" in
    x86_64 | amd64)  ARCH=x86_64 ;;
    aarch64 | arm64) ARCH=aarch64 ;;
    *) echo "❌ Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
echo "📍 Architecture: $ARCH"

# ---------------------------------------------------------------- CUDA toolchain
echo "🔍 Probing CUDA toolchain..."
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if   [ "${DRIVER%%.*}" -ge 580 ]; then DRIVER_MAX_CUDA=13
elif [ "${DRIVER%%.*}" -ge 525 ]; then DRIVER_MAX_CUDA=12
else
    echo "❌ Driver $DRIVER is too old: CUDA 12 needs >= 525, CUDA 13 needs >= 580" >&2
    exit 1
fi
echo "📍 Driver $DRIVER, compute capability $COMPUTE_CAP -> CUDA <= $DRIVER_MAX_CUDA"

NVCC=""
if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
else
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc \
                     "$WORKSPACE"/cuda-*/bin/nvcc \
                     "$VENV"/lib/python*/site-packages/nvidia/cu13/bin/nvcc; do
        if [ -x "$candidate" ]; then NVCC="$candidate"; break; fi
    done
fi

if [ -n "$NVCC" ]; then
    NVCC_MAJOR="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
    echo "📍 Found nvcc (CUDA $NVCC_MAJOR) at $NVCC"
    if [ "$NVCC_MAJOR" -gt "$DRIVER_MAX_CUDA" ]; then
        echo "⚠️  nvcc is newer than the driver supports, ignoring it"
        NVCC=""
    fi
fi
if [ -n "$NVCC" ]; then
    CUDA_MAJOR="$NVCC_MAJOR"
else
    echo "📍 No usable nvcc found, one will be installed"
    CUDA_MAJOR="$DRIVER_MAX_CUDA"
fi
echo "✅ Target: CUDA $CUDA_MAJOR"

# ---------------------------------------------------------------- uv
if command -v uv >/dev/null 2>&1; then
    echo "✅ uv: $(uv --version)"
else
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    export PATH="${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "❌ uv installed but not on PATH; add its bin directory and re-run" >&2
        exit 1
    fi
    echo "✅ uv: $(uv --version)"
fi

# ---------------------------------------------------------------- venv
if [ -x "$VENV/bin/python" ]; then
    echo "✅ Virtualenv already exists: $VENV"
else
    echo "📦 Creating virtualenv $VENV..."
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --quiet --python "$VENV/bin/python" pip setuptools wheel

# ---------------------------------------------------------------- Isaac Sim
if [ -x "$ISAACSIM/isaac-sim.sh" ]; then
    echo "✅ Isaac Sim already installed: $ISAACSIM"
else
    ZIP="$WORKSPACE/isaac-sim-standalone-$ISAACSIM_VERSION-linux-$ARCH.zip"
    if [ ! -f "$ZIP" ]; then
        echo "📥 Downloading Isaac Sim $ISAACSIM_VERSION for $ARCH (~9 GB)..."
        curl -fL -C - -o "$ZIP.part" \
            "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-$ISAACSIM_VERSION-linux-$ARCH.zip"
        mv "$ZIP.part" "$ZIP"
    fi
    echo "📦 Extracting Isaac Sim to $ISAACSIM..."
    mkdir -p "$ISAACSIM"
    unzip -q "$ZIP" -d "$ISAACSIM"
    (cd "$ISAACSIM" && ./post_install.sh)
    echo "✅ Isaac Sim ready (delete $ZIP to reclaim ~9 GB)"
fi

# ---------------------------------------------------------------- IsaacLab
if [ ! -d "$ISAACLAB/.git" ]; then
    echo "📥 Cloning IsaacLab..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
fi
git -C "$ISAACLAB" checkout --quiet "$ISAACLAB_COMMIT"
echo "📍 IsaacLab at $(git -C "$ISAACLAB" rev-parse --short HEAD)"

# -n replaces the existing symlink instead of creating a link inside what it points at
ln -sfn "$ISAACSIM" "$ISAACLAB/_isaac_sim"

if grep -q ISAACLAB_PATH "$VENV/bin/activate"; then
    echo "✅ Virtualenv already linked to IsaacLab"
else
    echo "📦 Linking virtualenv to IsaacLab..."
    (set +u; . "$VENV/bin/activate"; set -u; cd "$ISAACLAB" && ./isaaclab.sh --uv "../$(basename "$VENV")")
fi
echo "📦 Installing IsaacLab extensions..."
(set +u; . "$VENV/bin/activate"; set -u; cd "$ISAACLAB" && ./isaaclab.sh -i rsl_rl)

# ---------------------------------------------------------------- nvcc
if [ -z "$NVCC" ]; then
    if [ "$CUDA_MAJOR" -eq 13 ]; then
        echo "📦 Installing the CUDA 13 compiler into the venv..."
        SP="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
        # All four pinned to 13.0.x, else cicc and ptxas disagree on the PTX ISA version.
        uv pip install --python "$VENV/bin/python" \
            nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
            nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88
        ln -sf libcudart.so.13 "$SP/nvidia/cu13/lib/libcudart.so"   # pip ships no unversioned .so
        NVCC="$SP/nvidia/cu13/bin/nvcc"
    elif [ "$(id -u)" -eq 0 ]; then
        echo "📦 Installing the CUDA 12.8 toolkit via apt..."
        distro="$(. /etc/os-release && echo "$ID${VERSION_ID//./}")"
        curl -fL -o /tmp/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$ARCH/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq
        apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8
        NVCC=/usr/local/cuda-12.8/bin/nvcc
    else
        # pip's nvidia-cuda-nvcc-cu12 is a stub (ptxas only, no nvcc/cicc); NVIDIA's
        # redistributable tarballs are the complete toolchain and need no root.
        echo "📥 Unpacking the CUDA 12.8 redistributable toolchain (~81 MB)..."
        CUDA12_NVCC=12.8.93
        CUDA12_LIB=12.8.90   # cudart / cccl ship at a different patch level than nvcc
        CUDA12_HOME="$WORKSPACE/cuda-12.8"
        mkdir -p "$CUDA12_HOME"
        # linux-sbsa is the ARM server build; Jetson boards would need linux-aarch64
        case "$ARCH" in x86_64) plat=linux-x86_64 ;; aarch64) plat=linux-sbsa ;; esac
        for c in "cuda_nvcc-$plat-$CUDA12_NVCC" "cuda_cudart-$plat-$CUDA12_LIB" "cuda_cccl-$plat-$CUDA12_LIB"; do
            curl -fsL "https://developer.download.nvidia.com/compute/cuda/redist/${c%%-*}/$plat/$c-archive.tar.xz" \
                | tar xJ -C "$CUDA12_HOME" --strip-components=1
        done
        NVCC="$CUDA12_HOME/bin/nvcc"
    fi
fi
CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
echo "✅ nvcc: $NVCC"

# ---------------------------------------------------------------- torch
# isaaclab.sh -i installs a torch of its own (2.7.x)
# Muon needs >= 2.10
CURRENT_TORCH="$("$VENV/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ -n "$CURRENT_TORCH" ] && "$VENV/bin/python" -c "
import sys, torch
version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
cuda = (torch.version.cuda or '').split('.')[0]
sys.exit(0 if version >= (2, 10) and cuda == '$CUDA_MAJOR' else 1)
" 2>/dev/null; then
    echo "✅ torch $CURRENT_TORCH already matches CUDA $CUDA_MAJOR"
else
    if [ -n "$CURRENT_TORCH" ]; then
        echo "⚠️  Replacing torch $CURRENT_TORCH (need >= 2.10 built for CUDA $CUDA_MAJOR)"
    fi
    echo "📦 Installing torch for CUDA $CUDA_MAJOR..."
    if [ "$CUDA_MAJOR" -eq 13 ]; then
        uv pip install --python "$VENV/bin/python" \
            --index-url https://download.pytorch.org/whl/cu130 'torch>=2.10' torchvision
    else
        uv pip install --python "$VENV/bin/python" \
            --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
    fi
fi
uv pip install --quiet --python "$VENV/bin/python" 'numpy<2'   # torch bumps numpy; IsaacLab needs <2

# ---------------------------------------------------------------- Isaac Sim prebundle
PREBUNDLE="$ISAACSIM/exts/omni.isaac.ml_archive/pip_prebundle"
for d in torch torchvision nvidia; do
    [ -d "$PREBUNDLE/$d" ] || continue
    # mv into an existing .bak would nest the directory inside it instead of replacing it
    if [ -e "$PREBUNDLE/$d.bak" ]; then
        echo "❌ Both $d and $d.bak exist in $PREBUNDLE" >&2
        echo "   Isaac Sim was probably re-extracted over a sidelined install." >&2
        echo "   Keep whichever copy you want, remove the other, then re-run." >&2
        exit 1
    fi
    mv "$PREBUNDLE/$d" "$PREBUNDLE/$d.bak"
    echo "✅ Sidelined prebundled $d"
done

# ---------------------------------------------------------------- this extension
echo "📦 Installing g1_hoi_learning..."
uv pip install --quiet --python "$VENV/bin/python" -e "$REPO_DIR/source/g1_hoi_learning"

# ---------------------------------------------------------------- verify
echo ""
echo "🔍 Verifying installation..."
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

import numpy
import torch
from torch.optim import Muon  # noqa: F401  (torch >= 2.10)

print(f"   ✅ torch          {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"   ✅ numpy          {numpy.__version__}")
assert numpy.__version__ < "2", "IsaacLab requires numpy < 2"
print(f"   ✅ isaaclab       {version('isaaclab')}")
print(f"   ✅ rsl-rl         {version('rsl-rl-lib')}")
print(f"   ✅ g1_hoi_learning {version('g1_hoi_learning')}")
print("   ✅ Muon optimizer available")

assert torch.cuda.is_available(), "CUDA is not available to torch"
print(f"   ✅ CUDA device    {torch.cuda.get_device_name(0)}")
# Conv2d is the only cuDNN user in this pipeline, so it is what exposes cu12/cu13 mixups
torch.nn.Conv2d(1, 1, 3).cuda()(torch.randn(1, 1, 8, 8, device="cuda"))
print(f"   ✅ cuDNN          {torch.backends.cudnn.version()}")
PY

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Paths:"
echo "   workspace   $WORKSPACE"
echo "   venv        $VENV"
echo "   Isaac Sim   $ISAACSIM"
echo "   IsaacLab    $ISAACLAB"
echo "   CUDA        $CUDA_HOME"
echo ""
echo "📋 Next steps:"
echo "   1. source $VENV/bin/activate"
echo "   2. python scripts/list_envs.py"
