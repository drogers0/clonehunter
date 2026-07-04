extern crate cmake;

use bindgen::RustTarget;
use cmake::Config;
use std::{env, path::PathBuf};

fn build_and_link_mlx_c() {
    // When MLX_SYS_PREBUILT is set to a directory containing a prebuilt MLX
    // (e.g. from `pip install mlx`), skip the full MLX source build and just
    // build the thin mlx-c wrapper against the prebuilt libmlx.dylib.
    // This bypasses the need for `xcrun metal` (Xcode.app) when Metal is enabled.
    let prebuilt_dir = env::var("MLX_SYS_PREBUILT").ok();

    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    if let Some(ref prebuilt) = prebuilt_dir {
        // Use prebuilt MLX: tell mlx-c cmake to find_package(MLX) instead of
        // FetchContent (building from source). The prebuilt dir must contain:
        //   lib/libmlx.dylib, lib/mlx.metallib,
        //   include/mlx/*, share/cmake/MLX/MLXConfig.cmake
        config.define("MLX_C_USE_SYSTEM_MLX", "ON");
        config.define("CMAKE_PREFIX_PATH", prebuilt);
        config.define("MLX_C_BUILD_EXAMPLES", "OFF");
        // Metal/Accelerate are already baked into the prebuilt dylib;
        // these flags affect only the mlx-c compilation, not the MLX library.
        config.define("MLX_BUILD_METAL", "OFF");
        config.define("MLX_BUILD_ACCELERATE", "OFF");
    } else {
        config.define("MLX_BUILD_METAL", "OFF");
        config.define("MLX_BUILD_ACCELERATE", "OFF");

        #[cfg(feature = "metal")]
        {
            config.define("MLX_BUILD_METAL", "ON");
        }

        #[cfg(feature = "accelerate")]
        {
            config.define("MLX_BUILD_ACCELERATE", "ON");
        }
    }

    // build the mlx-c project
    let dst = config.build();

    let cmake_lib_dir = format!("{}/build/lib", dst.display());

    if let Some(ref prebuilt) = prebuilt_dir {
        // In prebuilt mode, a stale libmlx.a from a previous from-source build
        // may linger in the cmake build dir. The linker would prefer it over the
        // dynamic prebuilt lib, silently losing Metal support. Delete it.
        let stale = PathBuf::from(&cmake_lib_dir).join("libmlx.a");
        if stale.exists() {
            let _ = std::fs::remove_file(&stale);
        }
        // Prebuilt mode: link mlxc statically (just built), mlx dynamically (prebuilt)
        println!("cargo:rustc-link-search=native={}", cmake_lib_dir);
        println!("cargo:rustc-link-lib=static=mlxc");
        let lib_dir = format!("{}/lib", prebuilt);
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=dylib=mlx");
        // Note: rpath is NOT embedded automatically from build scripts (cargo
        // limitation). Use `install_name_tool -add_rpath <lib_dir> <binary>`
        // after building, or set DYLD_LIBRARY_PATH at runtime.

        // The prebuilt libmlx.dylib already links these frameworks, but we
        // must declare them for the final link step.
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        println!("cargo:rustc-link-search=native={}", cmake_lib_dir);
        println!("cargo:rustc-link-lib=static=mlx");
        println!("cargo:rustc-link-lib=static=mlxc");

        println!("cargo:rustc-link-lib=framework=Foundation");

        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Metal");
        }

        #[cfg(feature = "accelerate")]
        {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target(RustTarget::Stable_1_73)
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
