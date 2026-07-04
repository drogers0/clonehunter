//! Build script.
//!
//! The only thing this does is supply the link flags the ONNX Runtime CoreML
//! execution provider needs when statically linked on macOS (`--features
//! onnx-coreml`). The prebuilt static `libonnxruntime` bundles the CoreML
//! Objective-C++ objects, but `ort-sys` does not emit the framework links they
//! require, so the final link fails with undefined `onnxruntime::coreml::*`
//! symbols and the `___isPlatformVersionAtLeast` compiler-rt builtin. We add
//! them here. This is a no-op for every other build configuration.

fn main() {
    // Only relevant when the CoreML EP is compiled in, and only on macOS.
    let coreml = std::env::var_os("CARGO_FEATURE_ONNX_COREML").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !coreml || target_os != "macos" {
        return;
    }

    // CoreML EP Objective-C++ objects reference these frameworks.
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // `___isPlatformVersionAtLeast` (emitted for @available checks in the
    // CoreML objects) lives in the clang compiler-rt runtime. Locate it via
    // `clang -print-runtime-dir` and link the darwin builtins archive.
    if let Some(dir) = clang_runtime_dir() {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

/// Ask clang where its runtime libs live (`.../lib/clang/<v>/lib/darwin`).
fn clang_runtime_dir() -> Option<String> {
    let clang = std::env::var("CLANG").unwrap_or_else(|_| "clang".to_string());
    let out = std::process::Command::new(&clang)
        .arg("-print-runtime-dir")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if dir.is_empty() { None } else { Some(dir) }
}
