#!/usr/bin/env bash
set -euo pipefail

# Pinned MLX (Python/C++) wheel version — validated for detection parity
# (578 findings vs the frozen baseline).
#
# This wheel provides the prebuilt `libmlx` (+ `mlx.metallib`) that our C++ shim
# (csrc/ch_mlx.cpp) links against, via the vendored mlx-c wrapper (vendor/mlx-c).
# The only version pairing to keep aligned is the vendored mlx-c release ↔ this
# prebuilt MLX ABI (vendor/mlx-c is currently 0.2.0, targeting MLX 0.25.x).
# Re-validate parity before bumping either (see vendor/mlx-c/PROVENANCE.md).
MLX_VERSION="0.25.2"
INSTALL_DIR="${CLONEHUNTER_MLX_DIR:-$HOME/.local/share/clonehunter/mlx}"

echo "=== CloneHunter MLX Setup ==="
echo "MLX version: $MLX_VERSION"
echo "Install dir:  $INSTALL_DIR"

# 1. Create a temp venv and install the pinned MLX wheel.
# mlx==0.25.2 is on PyPI with prebuilt macOS-arm64 wheels. We prefer uv (fast,
# reliable resolver) and fall back to python3 -m venv + pip.
if ! command -v uv &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "ERROR: need either 'uv' or 'python3' to fetch the MLX wheel."
    echo "       Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
# Note: do not name this TMPDIR — that shadows the system variable mktemp/venv read.
# trap ensures the temp venv is removed even on an early `set -e` exit.
CH_TMPDIR=$(mktemp -d)
trap 'rm -rf "${CH_TMPDIR:-}"' EXIT
TMPVENV="$CH_TMPDIR/mlx-env"

if command -v uv &>/dev/null; then
    uv venv "$TMPVENV" --quiet
else
    python3 -m venv "$TMPVENV"
fi

# Require Python 3.11+ (matches README) — checked on the VENV interpreter so it
# covers both the uv and python3 paths (uv may select a Python not on PATH).
"$TMPVENV/bin/python3" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
    || { echo "ERROR: Python 3.11+ required (venv has $("$TMPVENV/bin/python3" --version 2>&1)); see README."; exit 1; }

if command -v uv &>/dev/null; then
    VIRTUAL_ENV="$TMPVENV" uv pip install --quiet "mlx==$MLX_VERSION"
else
    "$TMPVENV/bin/pip" install --quiet "mlx==$MLX_VERSION"
fi

# 2. Find the mlx site-package (mlx.__file__ may be None; use mlx.core.__file__)
MLX_PKG=$("$TMPVENV/bin/python3" -c "import mlx.core; import os; print(os.path.dirname(mlx.core.__file__))")

# 3. Validate package layout and copy to stable location
for subdir in lib include share; do
    [ -d "$MLX_PKG/$subdir" ] || { echo "ERROR: $MLX_PKG/$subdir not found — unexpected mlx package layout"; exit 1; }
done

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp -R "$MLX_PKG/lib" "$INSTALL_DIR/lib"
cp -R "$MLX_PKG/include" "$INSTALL_DIR/include"
cp -R "$MLX_PKG/share" "$INSTALL_DIR/share"

# 4. Patch MLXTargets.cmake — replace Xcode-specific Accelerate path
CMAKE_FILE="$INSTALL_DIR/share/cmake/MLX/MLXTargets.cmake"
if [ -f "$CMAKE_FILE" ]; then
    sed -i '' 's|/Applications/Xcode[^;]*Accelerate.framework|-framework Accelerate|g' "$CMAKE_FILE"
fi

# 5. Temp venv is removed by the EXIT trap set above.

echo ""
echo "=== MLX prebuilt installed to: $INSTALL_DIR ==="
echo ""
echo "Build CloneHunter with MLX:"
echo "  CLONEHUNTER_MLX_PREBUILT=$INSTALL_DIR cargo build --release --features mlx"
echo ""
echo "After building, fix the runtime library path:"
echo "  install_name_tool -add_rpath $INSTALL_DIR/lib target/release/clonehunter"
echo ""
echo "Then run:"
echo "  ./target/release/clonehunter scan . --embedder mlx"
