# CloneHunter Phase 5: Embedding Backend Landscape Research

**Date:** 2026-07-04
**Scope:** Evaluate Rust embedding-inference backends for `microsoft/codebert-base` (RoBERTa-family,
125M params, 768-dim) against two goals: (1) match/beat PyTorch MPS inference speed, (2) distribute
as a single self-contained binary.

**Baseline measurements (M-series Mac, ~3 400 snippets, `batch_size=16`):**
| Backend | Time | Throughput |
|---|---|---|
| candle CPU (current) | ~1 108 s | ~3.1 snippets/s |
| candle Metal (current, unstable) | ~241 s | ~14 snippets/s |
| PyTorch MPS (Python, target) | ~50 s | ~68 snippets/s |

The 5-8× candle-vs-PyTorch gap is well-documented for BERT-family models ([candle #2877]).
Root causes: element-wise kernels (`ucopy`/`badd`) lack vectorized memory access; no fused
attention; per-op overhead dominates small-batch inference ([candle #1139]).

---

## 1. Comparison Table

| Strategy | Speed vs PyTorch MPS | Single binary? | Binary size delta | Cross-platform | GPU portability | BERT maturity | Numeric reproducibility | Implementation effort |
|---|---|---|---|---|---|---|---|---|
| **ORT** (`ort` crate, being prototyped) | ~parity CPU; faster with CoreML | No — requires ~18 MB shared lib sidecar | +18 MB shared lib (or very large static binary if statically linked) | Prebuilt macOS/Linux/Windows | CPU, CUDA/TensorRT (Linux/Win prebuilt), CoreML (macOS), DirectML (Win), XNNPACK | Excellent — battle-tested BERT | High (ONNX spec) | Medium (being explored) |
| **tract** (Sonos pure-Rust ONNX) | Unknown/poor for BERT — "awful" per maintainers in non-optimized mode | YES — pure Rust, zero C++ runtime | +5–15 MB Rust crate code | macOS/Linux/Windows (pure Rust) | Metal + CUDA backends exist, transformer maturity unverified | Poor — `SkipLayerNormalization` (Microsoft domain) missing; blocks RoBERTa/BERT | Good (pure Rust, deterministic) | Very High — dead-end (see below) |
| **burn + wgpu** | Unknown — wgpu shader compilation unoptimized for BERT | Partial — wgpu adds WebGPU overhead; libtorch needed for existing BERT impl | Large (wgpu shader compilation, burn runtime) | macOS/Linux/Windows/WASM via WebGPU | Metal/Vulkan/DX12 in one build | Poor — burn-import ONNX coverage insufficient for BERT; sentence-transformers-burn requires `burn-tch` (libtorch) | Unknown — GPU ops may diverge across backends | Very High — not production-ready |
| **candle + Accelerate/MKL** (quick win on current stack) | ~2–4× over current candle CPU; ~3–6× slower than PyTorch MPS | YES — Accelerate is a macOS system framework (no new deps); MKL needs lib download on Linux | Zero on macOS (system lib); MKL ~200 MB on Linux | macOS arm64/x64 (Accelerate); Linux x86 (MKL); not portable across | None — CPU only | Excellent — unchanged code | Exact match (no logic change, only matmul backend) | Very Low — feature flag already in Cargo.toml |
| **Smaller model** (MiniLM-L6/CodeRankEmbed) | ~5–6× faster than current candle CPU (fewer params); ~1.2–2× slower than PyTorch MPS | YES — candle already embedded; tokenizer bundles too | Zero additional (smaller model JSON weights; download path unchanged) | Full (same candle stack) | Same as candle (CPU; Metal if stable) | Good — candle BERT/XLM-RoBERTa path already works | Different — new frozen baseline required | Medium — fix broken `faster` preset + re-freeze baseline |
| **GGUF / llama.cpp** | Fast GPU (1 795–30 843 tok/s at Metal for 33M model); BERT-class quantized model competitive | No — C++ shared lib required (or complex static link) | +5–20 MB (llama.cpp shared lib + quantized model) | macOS/Linux/Windows via CMake; per-platform build complexity | Metal, CUDA, Vulkan (compile-time flags) | Moderate — GGUF BERT conversion works; nomic-embed-code has GGUF variants | Quantization → numeric drift from fp32 baseline | High — C FFI bindings, CMake, llama.cpp API churn |

---

## 2. Strategy Deep Dives

### 2.1 ORT (`ort` crate) — being prototyped separately

**Speed:** ONNX Runtime CPU achieves ~400 sentences/s vs Python/PyTorch's ~100 sentences/s (4×
faster). For our workload, ORT CPU should bring the ~1 108 s wall time to roughly **55–110 s**
(assuming the canonical 5–8× candle-vs-PyTorch gap inverts via ORT's optimized BERT kernels). With
**CoreML on macOS** (Apple Neural Engine), 2–3× additional speedup over ORT CPU is plausible,
potentially reaching **20–50 s** — on par with or faster than PyTorch MPS.

**Distribution:** The central challenge. `ort` downloads a prebuilt ONNX Runtime shared library
(~18 MB for macOS arm64 v1.27). Static linking is supported but the static ONNX Runtime archive
is large (~100–200 MB compiled). The `load-dynamic` feature (recommended by `ort` maintainers)
loads the lib at runtime — simplifying linking but requiring the sidecar `.dylib`/`.so` be
shipped alongside the binary. The prebuilt download covers **CUDA+TensorRT only** for Linux/Windows;
CoreML and DirectML are "available in any build if the platform supports it." GPU execution providers
(CUDA, TensorRT) require dynamically loaded runtime libs regardless of linking strategy.

**BERT maturity:** Excellent. Production deployments widely documented. ONNX export of CodeBERT
is straightforward (`optimum` or manual `torch.onnx.export`).

**Reproducibility:** High — ONNX spec governs op semantics. Slight float ordering differences
vs candle possible but within existing frozen baseline tolerance (1e-4).

**`ort` crate status:** v2.0.0-rc.12 (March 2026), described by maintainers as "production-ready,
just not API-stable."

---

### 2.2 tract (Sonos pure-Rust ONNX) — DEAD-END

**Status:** Passes ~85% of ONNX backend tests, but almost entirely classical CV (ResNet,
Inception, VGG). Transformer support has been a known gap since 2020.

**Blocking issue for CodeBERT/RoBERTa:**
- `SkipLayerNormalization` (Microsoft domain operator) — present in every HuggingFace BERT/RoBERTa
  ONNX export; not implemented in tract ([tract #331]).
- `ConstantOfShape` with dynamic inputs — requires full symbolic dimension support.
- In the maintainers' own words: BERT in "incorporated" (non-optimized) mode "runs but performance
  is awful."

**GPU:** Metal and CUDA backends exist as of 2025/2026, but transformer-layer operator coverage
on those backends is unverified.

**Distribution:** tract's genuine strength — pure Rust, zero C++ runtime, true static binary.
If BERT support were complete and fast, this would be the ideal path.

**Verdict: Do not prototype.** Closing the Microsoft-domain operator gap would require either
forking tract or contributing a non-trivial amount of work upstream. Even then, symbolic dimension
support and optimized kernel paths for transformers need validation. ORT solves all of this today.

---

### 2.3 burn + wgpu — PREMATURE

**wgpu promise:** Single cross-platform GPU build (Metal/Vulkan/DX12/WebGPU/WASM) with no
per-platform C++ compilation. Architecturally compelling.

**Current BERT reality:**
- `burn-import` ONNX has limited operator coverage — insufficient for full BERT/RoBERTa graph.
- The one concrete Rust BERT implementation (`sentence-transformers-burn`) still requires
  `burn-tch` (libtorch backend), defeating the no-PyTorch goal.
- Importing via safetensors is possible but requires manually implementing the BERT computation
  graph in burn primitives.
- wgpu shader compilation overhead and kernel optimization for transformer shapes is not at
  production-grade maturity.

**Distribution:** Complex. wgpu itself is pure Rust + WGSL shaders. But getting BERT weights
in requires either libtorch (giant dep) or a custom ONNX importer (immature).

**Verdict: Do not prototype now.** Monitor burn's ONNX import operator coverage. Worth
revisiting in 12–18 months if burn-import reaches BERT-class coverage without libtorch.

---

### 2.4 candle + Accelerate/MKL — HIGHEST-PRIORITY QUICK WIN

**Root cause recap:** The per-op kernel overhead (`ucopy`, `badd`) is part of the story, but
the dominant cost in BERT inference is GEMM (attention projections, FFN layers). On CPU without
BLAS, candle falls back to a naive matmul. Enabling `Accelerate` (macOS) or `MKL` (Linux x86)
routes those GEMMs through Apple's BLAS or Intel's MKL, which use AMX coprocessors or AVX-512.

**Expected speedup:** 2–4× for matmul-dominated workloads on Apple Silicon; up to 4× on Intel
x86 with MKL. For our ~1 108 s baseline: estimated **280–550 s** with Accelerate on macOS.
Still ~6–11× slower than PyTorch MPS, but much more usable for occasional/cached embedding runs.

**Key fact: The features are already scaffolded in Cargo.toml (lines 87–93):**
```toml
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
mkl = ["candle-core/mkl", "candle-nn/mkl", "candle-transformers/mkl"]
```
Enabling them requires **zero code changes** — just `cargo build --release --features accelerate`
on macOS or `--features mkl` on Linux x86.

**Single binary:** YES on macOS — `Accelerate.framework` is a system-provided framework, no
new runtime deps. On Linux, MKL requires a separate runtime library download (not single-binary).

**Frozen baseline:** Unchanged — same logic, same model, same numeric path. Frozen baseline
remains intact.

**Verdict: Do this immediately.** Low effort, guaranteed partial win, no risk to detection
contract.

---

### 2.5 Smaller/Faster Model — HIGH-IMPACT MEDIUM EFFORT

**The math:** If model is 5–6× smaller → 5–6× fewer FLOP → proportional speedup in candle.
With Accelerate already enabled, combined speedup could reach **~10–15×** over current baseline:
1 108 s → **~75–110 s** CPU. That approaches PyTorch MPS territory.

**Candidate models:**

| Model | Params | Dim | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22 M | 384 | General text, not code-specific. ~5× smaller than CodeBERT. Very well-tested on CPU. |
| `CodeRankEmbed` (nomic) | ~521 MB weights | TBD | Code-specific, lightweight. Newer (2025). Limited production data. |
| `UniXCoder` | 125 M | 768 | Same size as CodeBERT but **higher F1 for clone detection** (0.918 vs CodeBERT's lower score). Drop-in swap but no speed gain. |
| `GraphCodeBERT` | 125 M | 768 | Best quality (F1 ~0.917), same size. No speed gain. |

**Quality tradeoff for clone detection:** Research ([ACM ASE 2022]) shows CodeBERT's F1 drops
when evaluated on code snippets outside training distribution. UniXCoder (same size) consistently
outperforms CodeBERT on clone detection. MiniLM trained on general text may miss code-specific
patterns (control flow, identifier semantics). The actual recall impact for CloneHunter's
use-case needs measurement — the frozen baseline would need re-freezing regardless.

**The broken `faster` preset:** CloneHunter already has a `faster` preset intended for this
use case, but it's broken — it loads BERT-family weights via the XLMRobertaModel architecture
(mismatch), and may fail at runtime. Fixing this preset and validating a working small model
is the right vehicle for this prototype.

**Single binary:** YES — the candle stack doesn't change. A smaller `tokenizer.json` would be
bundled instead of the 3.5 MB CodeBERT tokenizer.

**Verdict: Worth prototyping — fix `faster` preset, validate MiniLM-L6 first (well-understood),
then CodeRankEmbed.**

---

### 2.6 GGUF / llama.cpp — NOT RECOMMENDED

**Performance is real:** A 33M BERT-class model achieves 1 795–30 843 tok/s on Apple Silicon
Metal (from llama.cpp discussion #7712). Quantized CodeBERT-class models would be fast.
`nomic-embed-code` has GGUF quantized versions available.

**Distribution is the problem:**
- `llama-cpp-rs` crate is C FFI bindings to llama.cpp's C/C++ codebase — not pure Rust.
- Requires CMake build of llama.cpp or a prebuilt shared lib (same story as ORT but with faster
  API churn — llama.cpp releases new versions weekly).
- Metal and CUDA acceleration require compile-time flags; not a single cross-platform binary.
- The llama.cpp embedding API has changed repeatedly across versions; binding maintenance cost is
  high.

**Verdict: Not worth prototyping over ORT.** If you need llama.cpp-style quantized inference,
ORT with INT8 quantized ONNX model provides the same efficiency story with a stable API and
better Rust bindings. The GGUF ecosystem's true advantage (quantized LLM generation) doesn't
apply to encoder-only embedding models.

---

## 3. Ranked Recommendations

### Tier 1 — Do now, zero risk: candle + Accelerate/MKL

Enable the already-scaffolded feature flags. Expected 2–4× CPU speedup with no code changes,
no baseline impact, no distribution change, pure single-binary.

```bash
# macOS (Accelerate — system framework, no download)
cargo build --release --features accelerate

# Linux x86_64 (Intel MKL — requires oneAPI Base Toolkit or MKL runtime)
cargo build --release --features mkl
```

This should be the very next thing done before any larger backend work.

### Tier 2 — Prototype in parallel with ORT: Fix the `faster` preset (smaller model)

Validate `all-MiniLM-L6-v2` in the existing candle stack (6-layer BERT, 384-dim, candle's
BERT model path rather than XLMRoberta). Measure:
1. Inference speedup (expected: ~5×)
2. Clone detection quality on the four benchmark repos (compare finding counts vs frozen baseline)
3. If quality is acceptable, re-freeze baseline and ship as the default for the `faster` preset

Combined with Accelerate: ~10–15× CPU speedup over current → approaching PyTorch MPS parity.

### Do not prototype

- **tract**: Missing `SkipLayerNormalization` is a hard blocker for RoBERTa/BERT. Awful
  non-optimized performance. No production BERT deployments documented. Revisit only if they
  ship a transformer-complete operator set.
- **burn+wgpu**: Architecturally appealing but BERT via ONNX import is not working today.
  The only working BERT implementation requires libtorch. Revisit when burn-import achieves
  full BERT coverage without libtorch.
- **GGUF/llama.cpp**: Adds C++ build complexity and API churn for no benefit over ORT's INT8
  quantized ONNX approach.

---

## 4. Distribution Verdict

**Realistic path to fast + single-binary CodeBERT embedder in Rust:**

There is no free lunch. The options and their tradeoffs:

| Path | Speed (est., macOS arm64) | Single binary? | Ready today? |
|---|---|---|---|
| candle CPU + Accelerate (current model) | ~280–550 s | YES | Yes (feature flag) |
| candle CPU + Accelerate + MiniLM-L6 | ~55–110 s | YES | After preset fix + re-freeze |
| ORT + CoreML + CodeBERT | ~20–50 s (matches PyTorch MPS) | No (~18 MB shared lib sidecar) | After ORT prototype |
| ORT statically linked + CodeBERT | ~55–110 s CPU, no CoreML accel possible via static | Single binary (large, ~150–200 MB) | After ORT prototype, complex build |

**Recommended distribution target:**
- **macOS:** Ship `accelerate` as the default release build feature. Single binary, best effort
  on CPU. If ORT prototype proves out, ship ORT as an optional `--embedder onnx` preset that
  downloads/bundles the shared lib (following the HuggingFace Hub download pattern already
  used for model weights).
- **Linux x86:** `mkl` feature in CI for package releases; MKL runtime is a standard dependency
  in ML environments. Or ORT prebuilt lib bundled.
- **Linux arm64 / Windows:** ORT prebuilt libs cover these platforms. candle without BLAS
  works but slowly.
- **For the stub embedder (CI/testing):** No change — already single binary, fast, deterministic.

**The single-binary absolute floor:** candle + Accelerate (macOS) + MiniLM-L6 — this combination
is fast enough for most users' cached workflows (embedding is one-time cost) and distributes as a
true single binary. It's achievable with the existing candle stack plus fixing the `faster` preset.

---

## 5. Sources

- [ort — Fast ML inference for ONNX models in Rust](https://ort.pyke.io/)
- [ort — Linking documentation](https://ort.pyke.io/setup/linking)
- [ort — Execution providers](https://ort.pyke.io/perf/execution-providers)
- [ort crate on crates.io (v2.0.0-rc.12)](https://crates.io/crates/ort)
- [ONNX Runtime CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime releases (v1.27.0)](https://github.com/microsoft/onnxruntime/releases)
- [sonos/tract — GitHub](https://github.com/sonos/tract)
- [tract — BERT support issue #331](https://github.com/sonos/tract/issues/331)
- [FOSDEM 2026 — tract and torch-to-nnef](https://fosdem.org/2026/schedule/event/YJJQTD-tract-and-torch-to-nnef/)
- [tracel-ai/burn — GitHub](https://github.com/tracel-ai/burn)
- [sentence-transformers-burn — GitHub](https://github.com/tvergho/sentence-transformers-burn)
- [Building Sentence Transformers in Rust (Burn/ORT/Candle) — DEV Community](https://dev.to/mayu2008/building-sentence-transformers-in-rust-a-practical-guide-with-burn-onnx-runtime-and-candle-281k)
- [huggingface/candle — GitHub](https://github.com/huggingface/candle)
- [candle — Performance issues vs PyTorch, issue #1139](https://github.com/huggingface/candle/issues/1139)
- [candle — Slow generation vs transformers+PyTorch, issue #1683](https://github.com/huggingface/candle/issues/1683)
- [Comparing PyTorch & ONNX inference (Python vs Rust) — Stackademic](https://blog.stackademic.com/comparing-inference-performance-for-pytorch-onnx-models-in-python-and-rust-34e766c0f121)
- [Evaluating Small-Scale Code Models for Clone Detection (2025)](https://arxiv.org/pdf/2506.10995)
- [Generalizability of CodeBERT for Clone Detection (ASE 2022)](https://arxiv.org/abs/2208.12588)
- [On the Use of Deep Learning Models for Semantic Clone Detection (2024)](https://arxiv.org/pdf/2412.14739)
- [nomic-embed-code on HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-code)
- [Nomic Embed Code announcement — Simon Willison](https://simonwillison.net/2025/Mar/27/nomic-embed-code/)
- [llama.cpp GGUF embedding tutorial — Discussion #7712](https://github.com/ggml-org/llama.cpp/discussions/7712)
- [llama-cpp-rs crate](https://crates.io/crates/llama_cpp_rs)
- [ONNX Runtime BERT optimization blog (Microsoft)](https://opensource.microsoft.com/blog/2021/03/01/optimizing-bert-model-for-intel-cpu-cores-using-onnx-runtime-default-execution-provider/)
- [Benchmarking On-Device ML on Apple Silicon (2025)](https://arxiv.org/html/2510.18921v1)

---

## LINUX / NVIDIA RESULTS (click cold, ~3386 snippets, aorus-2: RTX 5090 + Ryzen 9 9950X, Ubuntu 24.04) — Phase 7

| Backend | embed | findings | shape | verdict |
|---|---|---|---|---|
| **candle-CUDA (GPU) ★** | **13.6s** | **578 (exact)** | binary + CUDA runtime | **FASTEST measured anywhere — 3.4× MLX-Metal, beats PyTorch-MPS 50s. Existing candle `cuda` feature (no new dep). The Linux GPU tier.** |
| ort-CPU (codebert) | 221.8s | 578 (exact) | ✅ single static binary (35 MB, ldd-confirmed) | The fast portable CPU tier (also cross-platform); faster than Mac's 356s (Zen5 AVX-512). |
| ort-CUDA (GPU) | 345.1s | 578 (exact) | sidecar | **LOST the bake-off** — no GPU speedup (Blackwell sm_120 + CUDA 12.0 toolkit + no-sudo cuDNN → GPU underused/CPU-fallback). `onnx-cuda` feature dropped. |

**Linux takeaway:** candle's CUDA backend is *far* more mature than its Metal backend — so Linux gets PyTorch-crushing GPU speed (13.6s) from the default framework with just `--features cuda`, while Mac needed the whole MLX effort. ort-CPU remains the fast single-binary portable CPU option. ort-CUDA is not worth keeping.

## MEASURED RESULTS (click cold, ~3386 snippets, M-series Mac) — Phase 5 synthesis

| Backend | embed | findings | single-binary | numerics vs PyTorch | notes |
|---|---|---|---|---|---|
| candle-CPU (codebert, default) | 1129s | 578 | ✅ | 2.68e-6 | frozen baseline |
| candle-CPU + Accelerate | **750s** | 578 | ✅ | exact (578) | free ~1.5x (not 2-4x); zero risk |
| candle-Metal (GPU) | 241s | 578 | ✅ | unvalidated | candle-metal instability (#2659) |
| **ort-CPU (codebert)** | **356s** | **578** | ✅ static | **2.38e-7** | 3x candle-CPU, exact detection |
| ort-CoreML (GPU/ANE) | 827s | 578 | ✅ static | exact (578) | **works now** (`e9d051a`: MLProgram+static export); SLOWER than CPU — CoreML dispatch overhead on small BERT batches |
| **MLX-Metal (GPU, prebuilt) ★** | **~46s** | **578** | ❌ sidecar (libmlx.dylib 16MB + mlx.metallib 85MB); Apple-Silicon only | 2.88e-12 (best) | **FASTEST — beats PyTorch-MPS (50s)**; exact parity; GPU-confirmed (user 5.58s / wall 47s); no Xcode via pip-wheel prebuilt link |
| MLX-CPU (codebert, spike) | 257s | 578 | ⚠️ builds MLX from source, Apple-Silicon only | **2.88e-12 (best of all)** | 4.3x candle-CPU; ≈candle-Metal *without* GPU (pip 0.25.2 prebuilt gives 578; earlier from-source 0.25.1 gave 532) |
| MiniLM faster-preset (candle BERT) | 271s | 547 | ✅ | new model | ~4x; **74% pair overlap w/ codebert** |
| MiniLM + Accelerate | 268s | 547 | ✅ | new model | Accelerate doesn't help small model |
| PyTorch-MPS (original) | 50s | — | — | reference | — |

### Verdict
- **Accelerate**: real but modest ~1.5x (corrects earlier inconclusive test); zero-risk single-binary, exact detection. Enable on macOS/Linux.
- **ort-CPU = the standout for single-binary + exact detection**: 3x faster than candle-CPU, best numerics (could share frozen baseline), exact 578 findings, single static binary (~18MB lib linked in). Recommend as the go-to fast backend (`--embedder onnx`, feature `onnx`).
- **MiniLM faster-preset** (now FIXED via candle BertModel — was broken): ~4x faster but only 74% detection overlap → a genuine speed/fidelity tradeoff. Appropriate as the explicit `faster` preset; needs its own re-frozen baseline if promoted.
- **ort-CoreML now works but is NOT fast** (`e9d051a`, 2026-07-04): build.rs framework links + MLProgram flag + fixed-seq static re-export make it run as a single static binary with exact parity (578) — but at 827s it is *slower* than ort-CPU. Per-inference CoreML dispatch overhead dominates small BERT batches (matches the general finding that CoreML EP suits fixed small CV graphs, not dynamic-shape transformers). Not recommended.
- **MLX-CPU (spike, `3446677`)**: best numerics of any backend (2.88e-12 vs PyTorch — MLX-CPU fp32 ≈ torch-CPU fp32), 4.3x faster than candle-CPU, comparable to candle-Metal *without* a GPU. But Apple-Silicon-only and builds MLX C++ from source (~15-20 min). As a *CPU* backend it's a marginal win over the cross-platform ort-CPU — not obviously worth a third CPU path.
- **★ MLX-Metal is the speed winner (`94a5596`, 2026-07-04): ~46s embed / 578 findings — it BEATS PyTorch-MPS (50s) as a native Rust binary, with exact detection parity and best-in-class numerics (2.88e-12).** GPU execution confirmed (user 5.58s vs 47s wall). Achieved WITHOUT Xcode by linking the pip `mlx` wheel's precompiled `libmlx.dylib` + `mlx.metallib` (patched vendored mlx-sys build.rs: `MLX_SYS_PREBUILT` env → cmake `find_package(MLX)` + link prebuilt; MLX's `load_colocated_library` resolves the metallib next to the dylib via dladdr). **Cost: NOT a single binary** — ships a ~114MB sidecar (13MB binary + 16MB dylib + 85MB metallib) with a post-build `install_name_tool -add_rpath`, and Apple-Silicon-only.
- **The speed↔distribution tradeoff is now explicit:** ort-CPU = cross-platform single static binary, 356s, exact. MLX-Metal = Apple-only sidecar, ~46s (PyTorch-class), exact. candle stays the default.
- **candle stays the default** (validated frozen baseline); the rest are opt-in.
