#!/usr/bin/env bash
set -euo pipefail

# Pinned MLX version — validated for detection parity (578 findings, 2.88e-12 numerics)
MLX_VERSION="0.25.2"
INSTALL_DIR="${CLONEHUNTER_MLX_DIR:-$HOME/.local/share/clonehunter/mlx}"

echo "=== CloneHunter MLX Setup ==="
echo "MLX version: $MLX_VERSION"
echo "Install dir:  $INSTALL_DIR"

# 1. Create temp venv and install mlx
# Uses uv if available (required for mlx==0.25.2 which is no longer on the live
# PyPI index but is available from uv's local cache).
TMPDIR=$(mktemp -d)
TMPVENV="$TMPDIR/mlx-env"

if command -v uv &>/dev/null; then
    uv venv "$TMPVENV" --quiet
    VIRTUAL_ENV="$TMPVENV" uv pip install --quiet "mlx==$MLX_VERSION"
else
    python3 -m venv "$TMPVENV"
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

# 5. Clean up temp venv
rm -rf "$TMPDIR"

echo ""
echo "=== MLX prebuilt installed to: $INSTALL_DIR ==="
echo ""
echo "Build CloneHunter with MLX:"
echo "  MLX_SYS_PREBUILT=$INSTALL_DIR cargo build --release --features mlx"
echo ""
echo "After building, fix the runtime library path:"
echo "  install_name_tool -add_rpath $INSTALL_DIR/lib target/release/clonehunter"
echo ""
echo "Then run:"
echo "  ./target/release/clonehunter scan . --embedder mlx"
