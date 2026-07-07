# CloneHunter

CloneHunter finds duplicate code across mixed-language repositories. It uses function-aware analysis for Python and windows-based analysis for other code files, with evidence so you can decide what to refactor.

Under the hood, CloneHunter combines snippet generation (function/window/call-expansion), transformer-based code embeddings (CodeBERT via [candle](https://github.com/huggingface/candle)), vector similarity search, and lexical scoring before rolling matches up into findings and HTML/JSON/SARIF reports. This is intentionally not a lightweight grep-style checker: it runs a semantic retrieval pipeline with model inference and indexing to catch harder, non-trivial duplicate patterns.

![CloneHunter HTML report screenshot](https://raw.githubusercontent.com/drogers0/clonehunter/master/assets/clonehunter-report-demo.png)

## Install

### Download binary (recommended)

Download the pre-built binary for your platform from [GitHub Releases](https://github.com/drogers0/clonehunter/releases):

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `clonehunter-aarch64-apple-darwin.tar.gz` |
| Linux x86_64 | `clonehunter-x86_64-unknown-linux-musl.tar.gz` |

```bash
# macOS / Linux example
curl -L https://github.com/drogers0/clonehunter/releases/latest/download/clonehunter-aarch64-apple-darwin.tar.gz | tar xz
sudo mv clonehunter /usr/local/bin/
```

### Cargo install (from source)

```bash
cargo install --git https://github.com/drogers0/clonehunter
```

### Build from source

```bash
git clone https://github.com/drogers0/clonehunter
cd clonehunter
cargo build --release
# Binary is at target/release/clonehunter
```

> **First run:** CodeBERT model weights (~440 MB) are downloaded from Hugging Face on first use
> and cached in `~/.cache/huggingface/hub/`. Subsequent runs use the local cache.
> Use `--embedder stub` for instant runs without any model download.

### Embedding backend support matrix

| Platform | Recommended backend | Build flag | Speed (click benchmark) |
|---|---|---|---|
| macOS arm64 + Metal GPU | MLX (opt-in sidecar) | `--features mlx` | ~46s |
| macOS arm64, no sidecar | candle-CPU + Accelerate | `--features accelerate` | ~750s |
| Linux x86_64 / arm64, CPU | ORT CPU (`--embedder onnx`) | `--features onnx` | ~357s |
| Linux x86_64 / arm64, NVIDIA GPU | candle-CUDA | `--features cuda` | ~14s |

The default build (no features) uses candle-CPU and works everywhere.

### Apple MLX backend (Apple Silicon, optional)

The MLX backend uses Apple's Metal GPU for embedding inference. It is the fastest
backend (~46s vs ~50s PyTorch-MPS on the click benchmark) with exact detection parity.

**Requirements:** Apple Silicon Mac, Python 3.11+ (for setup only, not at runtime).

```bash
# One-time setup: download and install prebuilt MLX library
./scripts/setup-mlx.sh

# Build with MLX support
CLONEHUNTER_MLX_PREBUILT=~/.local/share/clonehunter/mlx cargo build --release --features mlx

# Fix runtime library path
install_name_tool -add_rpath ~/.local/share/clonehunter/mlx/lib target/release/clonehunter

# Run with MLX
./target/release/clonehunter scan . --embedder mlx
```

> **Note:** The MLX build is not a single binary — it requires `libmlx.dylib` (~16 MB) and
> `mlx.metallib` (~85 MB) at runtime. If you see `Library not loaded: @rpath/libmlx.dylib`,
> run the `install_name_tool` command above or set
> `DYLD_LIBRARY_PATH=~/.local/share/clonehunter/mlx/lib`.

### ONNX Runtime backend (CPU, optional)

The ORT backend is the fast CPU-only alternative — ~3× faster than candle-CPU and ships
as a single statically-linked binary (no runtime dylib). It requires a pre-exported
`model.onnx`.

```bash
# Build
cargo build --release --features onnx

# Export model (requires PyTorch + transformers):
#   python -c "
#     from transformers import AutoModel
#     import torch
#     model = AutoModel.from_pretrained('microsoft/codebert-base')
#     dummy = torch.zeros(1, 8, dtype=torch.long)
#     torch.onnx.export(model, (dummy, dummy, dummy),
#       '~/.cache/clonehunter/onnx/codebert-base/model.onnx',
#       input_names=['input_ids','attention_mask','token_type_ids'],
#       output_names=['last_hidden_state'], opset_version=18,
#       dynamic_axes={n: {0:'batch',1:'seq'} for n in
#                     ['input_ids','attention_mask','token_type_ids','last_hidden_state']})"

# Run
./target/release/clonehunter scan . --embedder onnx
```

### NVIDIA CUDA backend (Linux, optional)

```bash
# Build (requires CUDA toolkit)
cargo build --release --features cuda

# Run
./target/release/clonehunter scan . --embedder codebert --device cuda
```

---

## Quickstart

```bash
clonehunter scan . --format html --out clonehunter_report.html
```

---

## Basic Usage

Scan a repository (defaults to HTML and `clonehunter_report.html`; output extension follows `--format`):

```bash
clonehunter scan .
```

Generate a JSON report:

```bash
clonehunter scan . --format json --out report.json
```

Generate an HTML report:

```bash
clonehunter scan . --format html --out report.html
```

Generate a SARIF report (for integration with GitHub Code Scanning):

```bash
clonehunter scan . --format sarif --out report.sarif
```

Diff scan (scan only changed files relative to a git ref):

```bash
clonehunter diff --base HEAD --format json --out diff.json
```

### Language Support

* **Python files**: parsed with tree-sitter AST extraction and analyzed with function/window snippets.
* **Other code files**: analyzed in windows-only mode by file content.
* **Cross-file-type comparisons** are supported.
* Results can vary by language and repository shape; tune thresholds/windows for best quality.

---

## How Scoring Works

Scores are **composite**: embedding similarity blended with lexical similarity.

```
composite = (1 - lexical_weight) * embedding + lexical_weight * lexical
```

Matches are filtered by `lexical_min_ratio`, and then the composite score is compared against the relevant threshold (`func`, `win`, or `exp`).

---

## Configuration (clonehunter.toml)

Place a `clonehunter.toml` file in your repository root to configure CloneHunter:

```toml
engine = "semantic"
cluster_findings = false
cluster_min_size = 2

[thresholds]
func = 0.92
win = 0.90
exp = 0.90
min_window_hits = 1
lexical_min_ratio = 0.5
lexical_weight = 0.3

[windows]
window_lines = 12
stride_lines = 6
min_nonempty = 4

[expansion]
enabled = false
depth = 1
max_chars = 4000

[index]
name = "brute"
top_k = 25

[cache]
path = "~/.cache/clonehunter"

[embedder]
name = "codebert"
model = "microsoft/codebert-base"
revision = "main"
max_length = 256
batch_size = 16
device = "auto"
```

> **Migrating from pyproject.toml:** Move your `[tool.clonehunter]` settings into a standalone
> `clonehunter.toml` file in your repository root. The key names are unchanged; drop the
> `[tool.clonehunter]` prefix (e.g., `[tool.clonehunter.thresholds]` → `[thresholds]`).

By default, CLI scans apply the `monorepo` repotype preset unless overridden with `--repotype` or `--repotype none`.

---

## CLI Options (selected)

```
clonehunter scan [PATHS...] [--format json|html|sarif] [--out FILE]
  --engine semantic|sonarqube
  --embedder codebert|stub|onnx|mlx
  --index brute|faiss
  --threshold-func FLOAT
  --threshold-win FLOAT
  --threshold-exp FLOAT
  --min-window-hits INT
  --lexical-min-ratio FLOAT
  --lexical-weight FLOAT
  --window-lines INT
  --stride-lines INT
  --min-nonempty INT
  --expand-calls
  --expand-depth INT
  --expand-max-chars INT
  --cache-path PATH
  --cluster
  --cluster-min-size INT
  --repotype dotnet|go|java|kotlin|monorepo|node|none|php|python|react|ruby|rust|swift|cpp
                                  # repeatable preset globs
  --include-globs GLOB   # repeatable; merged with config includes
  --exclude-globs GLOB   # repeatable; merged with config excludes
  --device auto|cpu|cuda

clonehunter diff --base REF [--format json|html|sarif] [--out FILE]
```

`--repotype` is additive and can be mixed (for example `--repotype python --repotype react`).
If `--repotype` is omitted, CloneHunter defaults to `monorepo` (scans all supported languages).
Use `--repotype none` to disable all repotype presets (scans 0 files unless `--include-globs` is also passed).
Merge order: `clonehunter.toml` globs → `--repotype` presets → explicit `--include-globs`/`--exclude-globs`.
When the same glob appears in both include/exclude sets, the most recent CLI layer wins.

Example mixed-language scan with custom overrides:

```bash
clonehunter scan . \
  --repotype python \
  --repotype react \
  --repotype cpp \
  --exclude-globs "**/__generated__/**" \
  --include-globs "**/*.txt" \
  --format html --out report.html
```

---

## Example Outputs

Example reports live in the `examples/` folder:

* [`examples/clonehunter_report.html`](examples/clonehunter_report.html)
* [`examples/clonehunter_report.json`](examples/clonehunter_report.json)
* [`examples/clonehunter_report.sarif`](examples/clonehunter_report.sarif)
* [`examples/clonehunter_diff.json`](examples/clonehunter_diff.json)
* [`examples/clonehunter_diff_report.html`](examples/clonehunter_diff_report.html)

Generate the example reports:

```bash
clonehunter scan . --format html --out examples/clonehunter_report.html
clonehunter scan . --format json --out examples/clonehunter_report.json
clonehunter scan . --format sarif --out examples/clonehunter_report.sarif
clonehunter diff --base HEAD --format json --out examples/clonehunter_diff.json
clonehunter diff --base HEAD --format html --out examples/clonehunter_diff_report.html
```

---

## Tuning Tips

* If you see too many false positives, increase `lexical_min_ratio` and/or `lexical_weight`.
* Increase `min_window_hits` to require stronger evidence.
* Exclude tests or generated code via `exclude_globs`.

---

## Known Limitations

* Semantic similarity is approximate, not guaranteed equivalence.
* Python findings are generally richer due to AST/function context.
* Non-Python findings use windows-only analysis and may require threshold/window tuning.
* Very small functions are harder to compare meaningfully.
* FAISS index is not implemented; the `--index faiss` flag is accepted for CLI compatibility but always uses the brute-force index.
* Embedding arithmetic uses f32 (candle default); near-equal cosine scores may differ slightly from the Python baseline.
* The `mlx` embedder backend is Apple Silicon only and requires a sidecar library (~101 MB).
  See the [Apple MLX backend](#apple-mlx-backend-apple-silicon-optional) section for build instructions.

---

## Development

```bash
# Run the full test suite
cargo test

# Format and lint
cargo fmt --check
cargo clippy --all-targets -- -D warnings

# Fast dev loop without model download (stub embedder is deterministic)
CLONEHUNTER_EMBEDDER=stub cargo run -- scan .

# Full scan with CodeBERT (downloads ~440 MB on first run)
cargo run --release -- scan . --format html
```
