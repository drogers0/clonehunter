//! Build script — active only under `--features mlx`.
//!
//! Builds Apple's `mlx-c` (vendored at `vendor/mlx-c`) against a **prebuilt**
//! `libmlx` (installed by `scripts/setup-mlx.sh`, located via the
//! `CLONEHUNTER_MLX_PREBUILT` env var), compiles our C++ shim (`csrc/ch_mlx.cpp`),
//! and links them. For every other build (default / onnx / cuda) this is a no-op
//! and the `cc`/`cmake` crates are compiled out entirely (they are optional
//! build-deps gated into the `mlx` feature).

#[cfg(feature = "mlx")]
fn build_mlx() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=csrc/ch_mlx.cpp");
    println!("cargo:rerun-if-changed=csrc/ch_mlx.h");
    println!("cargo:rerun-if-env-changed=CLONEHUNTER_MLX_PREBUILT");

    let prebuilt = std::env::var("CLONEHUNTER_MLX_PREBUILT").unwrap_or_else(|_| {
        panic!(
            "CLONEHUNTER_MLX_PREBUILT is not set. The `mlx` feature links a prebuilt \
             libmlx (Apple Silicon only); building MLX from source needs Xcode's Metal \
             toolchain and is not supported. Run ./scripts/setup-mlx.sh, then build with \
             CLONEHUNTER_MLX_PREBUILT=~/.local/share/clonehunter/mlx"
        )
    });

    // Build the thin mlx-c wrapper (libmlxc.a) against the prebuilt libmlx via
    // find_package(MLX) — never compiles the MLX core from source.
    let mut cfg = cmake::Config::new("vendor/mlx-c");
    cfg.define("CMAKE_INSTALL_PREFIX", ".")
        .define("MLX_C_USE_SYSTEM_MLX", "ON")
        .define("CMAKE_PREFIX_PATH", &prebuilt)
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_METAL", "OFF")
        .define("MLX_BUILD_ACCELERATE", "OFF");
    #[cfg(debug_assertions)]
    cfg.define("CMAKE_BUILD_TYPE", "Debug");
    #[cfg(not(debug_assertions))]
    cfg.define("CMAKE_BUILD_TYPE", "Release");
    let dst = cfg.build();
    let cmake_lib_dir = format!("{}/build/lib", dst.display());

    // A stale libmlx.a from a prior from-source build in a reused OUT_DIR would
    // shadow the prebuilt dynamic lib — remove it. (Moot under find_package, but
    // harmless and cheap insurance.)
    let stale = PathBuf::from(&cmake_lib_dir).join("libmlx.a");
    if stale.exists() {
        let _ = std::fs::remove_file(&stale);
    }

    // Compile the shim FIRST so its `-l` directive precedes mlxc/mlx on the link
    // line: libch_mlx.a has undefined references *into* mlxc/mlx, and a
    // consumer-before-provider order is correct on any linker.
    cc::Build::new()
        .cpp(true)
        .file("csrc/ch_mlx.cpp")
        .include("csrc")
        .include("vendor/mlx-c")
        .flag_if_supported("-std=c++17")
        .compile("ch_mlx"); // emits rustc-link-lib=static=ch_mlx

    // Then the mlx-c wrapper (static) and the prebuilt MLX core (dynamic).
    println!("cargo:rustc-link-search=native={cmake_lib_dir}");
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-search=native={prebuilt}/lib");
    println!("cargo:rustc-link-lib=dylib=mlx");

    // Frameworks the prebuilt libmlx depends on, plus the C++/ObjC runtimes.
    for fw in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");

    // Note: cargo build scripts cannot embed an rpath. At runtime set
    // DYLD_LIBRARY_PATH=$CLONEHUNTER_MLX_PREBUILT/lib, or run
    // `install_name_tool -add_rpath <lib> <binary>` on the release binary.
}

fn main() {
    #[cfg(feature = "mlx")]
    build_mlx();
}
