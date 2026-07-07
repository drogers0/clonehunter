# Vendored: `mlx-c` (Apple MLX C API)

- **Upstream:** https://github.com/ml-explore/mlx-c
- **Version:** `MLX_C_VERSION 0.2.0` (see `CMakeLists.txt`)
- **Targets MLX core:** the `FetchContent` `GIT_TAG` in `CMakeLists.txt` is `v0.25.1`, but that
  path is only used for a from-source build, which we do NOT use. We build against a
  **prebuilt** `libmlx` from the `mlx==0.25.2` wheel (`scripts/setup-mlx.sh`) — the operative
  MLX runtime is 0.25.2; the 0.25.1 tag is inert here.
- **Modifications:** none. This is the unmodified upstream Apple C API source. It was
  previously vendored transitively inside the `mlx-sys` crate at
  `vendor/mlx-sys-0.2.0/src/mlx-c`; when this project dropped the `mlx-rs`/`mlx-sys`
  dependency stack in favour of a self-owned C++ shim (`csrc/ch_mlx.cpp`), the `mlx-c`
  source was relocated here and everything else in `mlx-sys` was deleted.

## How it is built

`mlx-c` is a thin C wrapper (produces `libmlxc.a`) over the MLX C++ core (`libmlx`). The
crate's root `build.rs` (active only under `--features mlx`) builds `mlxc` here via cmake
with `MLX_C_USE_SYSTEM_MLX=ON`, pointing `CMAKE_PREFIX_PATH` at a **prebuilt `libmlx`**
(installed by `scripts/setup-mlx.sh` to `~/.local/share/clonehunter/mlx`, located at build
time via the `CLONEHUNTER_MLX_PREBUILT` env var). This avoids compiling the huge MLX core
from source (which would require Xcode's Metal toolchain). Our C++ shim links `static=mlxc`
+ `dylib=mlx`.

## How to bump

1. Replace this tree with a newer `ml-explore/mlx-c` tag (keep it unmodified).
2. Install a **matching** prebuilt MLX (`scripts/setup-mlx.sh`, adjust the wheel version so
   the MLX ABI matches the mlx-c release).
3. Update `csrc/ch_mlx.cpp` if any C API signature changed.
4. Re-validate parity: the detection contract is **578 findings + scores within 1e-4** of
   `benchmark/baseline.json`; the in-crate
   `src/embedding/mlx_backend.rs::tests::mlx_parity_vs_pytorch` (`#[ignore]`) is the
   embedding-level gate. Do not tighten its tolerance band.
