# Phase 0: Cargo Scaffold + Parity Spike (T0 + T1)

Issue: #23 — Rewrite CloneHunter in Rust

## Problem Summary

CloneHunter is being rewritten from Python to Rust for single-binary distribution, faster scans, and tree-sitter parsing. Phase 0 establishes the Cargo project structure, validation toolchain, foundational error/logging types, and — critically — runs a parity spike that empirically validates the candle + tree-sitter + re-freeze approach before any production code is written. Everything downstream (T2–T15) depends on the spike's GO/NO-GO.

## Design Decisions

### DD1: Crate layout — single crate with library + binary

**Options considered:**
- **(A) Single crate, `src/lib.rs` + `src/main.rs`** — one `Cargo.toml`, library modules mirroring the Python package, one binary target.
- **(B) Cargo workspace** — separate `clonehunter-core` lib and `clonehunter` binary crate.

**Decision: (A) Single crate.**
- Pros: Simpler build, no cross-crate version coordination, integration tests have full access, matches the Python "one package" shape, easier initial development.
- Cons: If we later want a library API for external consumers, we'd refactor into a workspace. This is unlikely near-term.
- Rationale: The project has one binary consumer. A workspace adds coordination cost with no current benefit. The lib.rs/main.rs split still gives clean separation for testing.

### DD2: Module layout — mirror Python package structure with layer discipline

Modules under `src/`: `cli/`, `core/`, `parsing/`, `snippets/`, `embedding/`, `index/`, `similarity/`, `engines/`, `reporting/`, `io/`. Each becomes a Rust module directory with `mod.rs`. This preserves the mental model from CLAUDE.md and makes the port traceable file-by-file.

**Layer discipline:** `core/` is a leaf module — it contains only types, config, errors, logging, and fingerprints. It has no dependencies on other `clonehunter` modules. Top-level orchestration (the equivalent of `core/pipeline.py`) lives in `engines/` (specifically the semantic engine), not in `core/`. This prevents `core` from becoming a catch-all.

**Visibility:** Default to `pub(crate)` for module internals. Only `cli::run()` and the re-exports needed by integration tests are `pub`. The Python `model/interfaces.py` ABCs (only `Engine` and `VectorIndex` are subclassed) become traits defined in their subsystem modules (e.g., `VectorIndex` in `index/`, `Engine` in `engines/`), not a separate `model/` module.

### DD3: Error strategy — module-local error types with `thiserror`

**Options considered:**
- **(A) Single `CloneHunterError` enum** — one enum with `#[from]` conversions for all subsystems.
- **(B) Module-local error types** — `ConfigError`, `EmbeddingError`, `CacheError`, etc., each `thiserror`-derived, with a small top-level `AppError` at CLI boundaries.
- **(C) `anyhow` everywhere** — erased errors.

**Decision: (B) Module-local error types with thin boundary enums, `anyhow` in `main.rs`.**
- Rationale: A single enum would grow to a god-type by T15. Module-local errors keep each subsystem's failure modes contained and self-documenting. `ConfigError` is the only error type defined in Phase 0; others are added by the modules that need them (T4: `CacheError`, T5: `ParseError`, T7: `EmbeddingError`, etc.).
- **Boundary composition:** Orchestration layers (T10 `engines/pipeline`, T12 `cli`) define thin boundary enums (e.g., `PipelineError`) with `#[from]` wrappers around subsystem errors. This preserves typed control over messaging, exit codes, and reproducibility metadata without collapsing everything into `anyhow`.
- **Recoverable degradations:** Not propagated through the error channel. Instead, successful results carry a `Vec<Degradation>` (or similar structured diagnostics list) recording events like parse-skip, device fallback, cache self-heal, ANN→brute. These are logged via `tracing::warn!` and surfaced in stats/timing metadata. This prevents silent loss of fallback information.
- `main.rs` uses `anyhow::Result` for clean CLI error reporting. Library modules use their local `Result<T>` aliases.

### DD4: Logging — `tracing` crate

**Decision: `tracing` + `tracing-subscriber`.**
- Rationale: `tracing` is the modern Rust standard, supports structured fields (useful for timing/stats), zero-cost when disabled, and `tracing-subscriber` provides the same `[LEVEL] message` format as the Python logger. Minimal overhead vs `log`; better investment for the full port.

### DD5: Dependency map — Python → Rust

| Python dep | Rust crate | Justification | Risk |
|---|---|---|---|
| `torch` + `transformers` | `candle-core`, `candle-nn`, `candle-transformers` | Pure Rust RoBERTa/CodeBERT inference; single binary goal. Backend features: `cuda` and `metal` (Apple Silicon). | **Medium** — candle is younger than PyTorch; numerical parity is the spike's purpose (T1a). CodeBERT is a RoBERTa-family model (`architectures: ["RobertaModel"]`); the spike must validate candle's RoBERTa path specifically. |
| HF tokenizer (via transformers) | `tokenizers` (default-features = false) | Official HF crate, used by candle examples; BPE tokenization. Disable default features to minimize binary size (no `progressbar`, `onig`, `esaxx_fast`). | **Low** — mature, widely used. |
| HF model download | `hf-hub` | Download model weights/config/tokenizer from HuggingFace Hub. Required by candle loading pattern. | **Low** — official HF crate. |
| `ast` (stdlib) | `tree-sitter`, `tree-sitter-python` | Python function extraction; language-agnostic for future langs. **Verify ABI compatibility between tree-sitter and tree-sitter-python versions before committing pins.** | **Medium** — line-span semantics differ from CPython AST (T1b spike topic). |
| `numpy` (cosine, matmul) | `ndarray` | Brute-force cosine similarity via matrix multiply. | **Low** — mature, well-tested. |
| `multiprocessing` | `rayon` | Parallel candidate retrieval (issue #11). | **Low** — standard Rust parallelism. |
| `sqlite3` | `rusqlite` | Embedding cache (WAL, schema versioning). | **Low** — mature, full SQLite3 binding. |
| `argparse` | `clap` (derive) | CLI parsing with subcommands, defaults, env vars. | **Low** — ecosystem standard. |
| `tomllib`/`tomli` | `toml` + `serde` | Config file parsing. Standalone `clonehunter.toml` per alignment decision. | **Low**. |
| `difflib.SequenceMatcher` | `similar` | Side-by-side diffs for HTML/JSON reports. | **Low** — designed as difflib replacement. |
| `hashlib.sha256` | `sha2` + `hex` | Fingerprinting (`hash_text`, `embed_cache_key`). `hex` for hexdigest-format output matching Python's `hexdigest()`. | **Low** — RustCrypto standard. |
| `re.findall` | `regex` | Identifier tokenization for Jaccard lexical similarity (T9). | **Low** — ecosystem standard. |
| `tqdm` | `indicatif` | Progress bars. | **Low**. |
| `json` | `serde_json` | JSON report output + SARIF. | **Low**. |
| `os.walk` | `walkdir` | File collection with early directory pruning. | **Low**. |
| `glob` matching | `globset` | Custom `**` glob matching for include/exclude. | **Low** — from the ripgrep author. |
| — | `dirs` | Platform XDG cache directory (`~/.cache/clonehunter`). | **Low**. |

**Not included in Phase 0:** `faiss` equivalent (T8 will evaluate `hora` or `hnswlib-rs`; brute-force is the parity path). SARIF schema validation (T11). HTML templating (T11 — inline string building matches Python).

### DD6: MSRV, toolchain, and reproducibility

**Decision: Rust 2024 edition, MSRV 1.85.0** (first stable 2024-edition release). Pin via `rust-toolchain.toml`. Candle requires ≥1.75; 1.85 gives us 2024 edition features and is current stable. No nightly features.

**Reproducibility:** `Cargo.lock` is committed (binary crate convention). After the spike validates specific crate versions, exact patch versions are pinned. The HuggingFace model is pinned to a specific revision SHA (not `main`). The parity baseline is produced on CPU; GPU/Metal is best-effort behavior, not part of the frozen contract.

### DD7: Normalization contract (T1c output — defined here, validated by spike)

**Options considered:**
- **(A) tree-sitter docstring strip + `ast.unparse`-equivalent canonical re-emit** — build a custom AST printer from tree-sitter CST. Highest fidelity to Python behavior but enormous implementation surface.
- **(B) tree-sitter docstring strip + raw source passthrough** — strip docstrings by deleting their byte ranges from source, keep everything else as-is.
- **(C) tree-sitter docstring strip + comment removal + whitespace normalization** — strip docstrings, remove comments, collapse whitespace runs. Middle ground approaching `ast.unparse` semantics.

**Decision: (B) with explicit contract — tree-sitter docstring strip + source-preserving passthrough.**

The normalization contract specifies exactly what text drives embeddings, hashes (cache keys), lexical tokens, and rendered diffs:

1. **Docstrings:** Identified by tree-sitter as `expression_statement > string` at body position 0 of `function_definition`, `class_definition`, or module. The entire `expression_statement` node's byte range (from start of indentation to trailing newline, inclusive) is replaced with `{indent}pass\n` where `{indent}` matches the node's indentation. For multiline docstrings, this reduces the snippet's line count by `len(docstring_lines) - 1`. T6 must account for this line-count reduction when computing absolute line offsets in normalized snippets.
2. **Comments:** Comments are **stripped** from normalized text used for embeddings and lexical scoring. Preserving comments would add noise to lexical Jaccard (identifier tokens), potentially flipping near-threshold findings based on comment-only edits. However, comments are preserved in the **display text** used for rendered diffs in reports. This means normalized text has two forms: **(a) analysis text** (comments stripped, used for embeddings, hashes, lexical tokens) and **(b) display text** (comments preserved, used for rendered diffs). The analysis text is what drives detection; the display text is what users see. The spike must validate that comment stripping does not cause unexpected divergence from the Python baseline.
3. **Whitespace/indentation:** Preserved as-is in both analysis and display text (unlike `ast.unparse` which re-indents). Known divergence.
4. **String quotes:** Preserved as-is (unlike `ast.unparse` which normalizes). Known divergence.
5. **Decorators:** Not part of the extracted `FunctionRef.code` — extraction uses the `function_definition` node's span (starting at `def`), not `decorated_definition`. This matches Python's `ast.FunctionDef.lineno` which points at `def`.

**Implication:** The Rust baseline will not match the Python baseline byte-for-byte. The re-freeze (T13) produces a new `baseline.json`; the diff vs old must be fully explained by these documented normalization divergences. The spike must measure impact on both embedding cosines **and** lexical similarity scores, and must verify that comment stripping + whitespace preservation does not cause near-threshold findings to flip relative to the Python baseline.

### DD8: Spike structure — committed code + decision memo

The spike (T1) produces:
1. A `spike/` directory **committed to the repo** (not gitignored) with a `README.md` header stating "throwaway research code — do not port." This enables audit/reproduction of the GO/NO-GO verdict.
2. A checked-in `23_PHASE0_SPIKE_MEMO.md` with empirical results, GO/NO-GO, exact `cargo run` invocation, fixture file SHA-256 hashes, and crate versions used.
3. A `spike/fixtures/` directory with reference Python embeddings and parse results.

---

## Step-by-step instructions

### T0 — Cargo scaffold + validation toolchain

#### Step 0.1: Create `Cargo.toml`

Create `/Cargo.toml` at the repo root:

```toml
[package]
name = "clonehunter"
version = "0.1.0"
edition = "2024"
rust-version = "1.85.0"
description = "Find duplicate code across mixed-language repositories"
license = "MIT"
default-run = "clonehunter"

[[bin]]
name = "clonehunter"
path = "src/main.rs"

[lib]
name = "clonehunter"
path = "src/lib.rs"

[dependencies]
# Error handling
thiserror = "2"
anyhow = "1"

# Logging / tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# CLI
clap = { version = "4", features = ["derive", "env"] }

# Serialization / config
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"

# Hashing
sha2 = "0.10"
hex = "0.4"

# Embedding (candle) — CodeBERT is RoBERTa-family
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = { version = "0.21", default-features = false }
hf-hub = "0.3"

# Database (embedding cache)
rusqlite = { version = "0.32", features = ["bundled"] }

# Parsing
tree-sitter = "0.24"
tree-sitter-python = "0.23"

# Numeric
ndarray = "0.16"

# Parallelism
rayon = "1"

# Diff
similar = "2"

# Regex (lexical similarity tokenization)
regex = "1"

# File system
walkdir = "2"
globset = "0.4"
dirs = "5"

# Progress
indicatif = "0.17"

[dev-dependencies]
tempfile = "3"
assert_cmd = "2"
predicates = "3"

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]

[profile.release]
lto = true
opt-level = 3
strip = true
codegen-units = 1
```

**Note:** After the spike (T1) validates specific crate versions, replace semver ranges with exact patch pins and commit `Cargo.lock`. Verify `tree-sitter-python 0.23` builds against `tree-sitter 0.24` — if not, adjust to matching versions (test with `cargo build`; `Language::new(tree_sitter_python::LANGUAGE)` must compile).

#### Step 0.2: Create `rust-toolchain.toml`

```toml
[toolchain]
channel = "1.85.0"
components = ["rustfmt", "clippy"]
```

#### Step 0.3: Create module skeleton

Create the following files with minimal placeholder content:

**`src/lib.rs`** — root library, declares module tree:
```rust
pub(crate) mod cli;
pub(crate) mod core;
pub(crate) mod embedding;
pub(crate) mod engines;
pub(crate) mod index;
pub(crate) mod io;
pub(crate) mod parsing;
pub(crate) mod reporting;
pub(crate) mod similarity;
pub(crate) mod snippets;

// Re-export only the public entry point needed by main.rs
pub use cli::run;
```

**`src/main.rs`** — binary entry point:
```rust
use anyhow::Result;

fn main() -> Result<()> {
    clonehunter::run()
}
```

**Module stubs** — each module directory gets a `mod.rs`. For Phase 0, only `core/` has real content; all others get a minimal `mod.rs`:

- `src/cli/mod.rs` — `pub(crate) fn run() -> anyhow::Result<()> { todo!() }`
- `src/core/mod.rs` — declares `errors`, `logging` (implemented below)
- `src/embedding/mod.rs` — empty (comment: `// T7`)
- `src/engines/mod.rs` — empty (comment: `// T10`)
- `src/index/mod.rs` — empty (comment: `// T8`)
- `src/io/mod.rs` — empty (comment: `// T4`)
- `src/parsing/mod.rs` — empty (comment: `// T5`)
- `src/reporting/mod.rs` — empty (comment: `// T11`)
- `src/similarity/mod.rs` — empty (comment: `// T9`)
- `src/snippets/mod.rs` — empty (comment: `// T6`)

**Note:** `cargo run` will panic at the `cli::run()` `todo!()` — this is expected. `cargo test` is the T0 gate; runtime functionality begins in T12.

#### Step 0.4: Implement `src/core/errors.rs`

Phase 0 defines only `ConfigError`. Other module-local error types are added by the modules that need them (see DD3).

```rust
use thiserror::Error;

/// Configuration errors. Defined here because config is a core concern.
/// Other error types (EmbeddingError, CacheError, ParseError, etc.)
/// are defined in their respective modules.
#[derive(Debug, Error)]
pub(crate) enum ConfigError {
    #[error("invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },

    #[error("failed to read config: {0}")]
    ReadError(String),
}
```

#### Step 0.5: Implement `src/core/logging.rs`

Maps Python's `core/logging.py` singleton logger:

```rust
use tracing_subscriber::{fmt, EnvFilter};
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the global tracing subscriber.
/// Safe to call multiple times; only the first call takes effect.
pub fn init_logging() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("clonehunter=info"));
        fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_writer(std::io::stderr)
            .init();
    });
}
```

Design notes:
- `[LEVEL] message` format matches Python's `[%(levelname)s] %(message)s`.
- `RUST_LOG=clonehunter=debug` overrides the default.
- `Once` guard matches the Python "add handler only if none" pattern.
- Writes to stderr (same as Python's `StreamHandler` default).

#### Step 0.6: Wire up `src/core/mod.rs`

```rust
pub(crate) mod errors;
pub(crate) mod logging;
```

#### Step 0.7: Validation gates

Create a `Makefile` with the local validation commands. Since there is no CI, these are the gate:

```makefile
.PHONY: check
check: fmt-check lint test

.PHONY: fmt-check
fmt-check:
	cargo fmt --check

.PHONY: lint
lint:
	cargo clippy --all-targets -- -D warnings

.PHONY: test
test:
	cargo test
```

Equivalent to the Python repo's:
| Python | Rust |
|---|---|
| `uv run ruff format --check .` | `cargo fmt --check` |
| `uv run ruff check .` | `cargo clippy --all-targets -- -D warnings` |
| `uv run pyright` | (clippy + type system cover this) |
| `uv run pytest` | `cargo test` |

#### Step 0.8: Update `.gitignore` and commit policy

Append to existing `.gitignore`:
```
# Rust
/target/
```

**`Cargo.lock` is committed** (binary crate convention). Do NOT add it to `.gitignore`. This ensures reproducible builds — critical because the parity baseline depends on exact crate versions.

**`spike/` is committed** (not gitignored) — see DD8.

#### Step 0.9: Initial tests

Inline `#[cfg(test)]` in each file:

In `src/core/errors.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats_include_context() {
        let e = ConfigError::InvalidValue {
            field: "threshold".into(),
            reason: "must be in [0,1]".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("threshold"));
        assert!(msg.contains("must be in [0,1]"));
    }
}
```

In `src/core/logging.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_logging_is_idempotent() {
        init_logging();
        init_logging(); // must not panic
    }
}
```

#### Step 0.10: Verify T0

Run all three gates via the Makefile and confirm they pass:
```bash
make check
```
This runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test`.

`cargo run` will panic at `todo!()` — this is expected and does not block T0.

---

### T1 — Parity Spike (Hard Gate)

#### Step 1.0: Create spike directory structure and Cargo.toml

```
spike/
  README.md             # "Throwaway research code — do not port"
  Cargo.toml            # standalone crate (not a workspace member)
  src/
    main.rs             # entry point, runs all three checks with assertions
    candle_embed.rs     # T1a: candle RoBERTa/CodeBERT embedding
    treesitter_parse.rs # T1b: tree-sitter function extraction
    normalize.rs        # T1c: normalization comparison
  fixtures/
    snippets.txt        # code snippets for embedding comparison
    parse_targets/      # Python files for extraction comparison
  generate_references.py  # Python script to produce reference data
```

**`spike/Cargo.toml`:**
```toml
[package]
name = "clonehunter-spike"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = { version = "0.21", default-features = false }
hf-hub = "0.3"
tree-sitter = "0.24"
tree-sitter-python = "0.23"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
regex = "1"
similar = "2"
sha2 = "0.10"
hex = "0.4"
```

The spike is a standalone crate (not a workspace member of the main project). This keeps experimental code isolated. `spike/Cargo.lock` is committed alongside the spike code to ensure reproducibility.

**Anti-drift rule:** The crate versions and model revision SHA validated by the spike must be copied verbatim into the root `Cargo.toml` before T2 proceeds. The spike memo records exact versions; implementers must verify root pins match.

#### Step 1.1: Generate Python reference data

Before writing any Rust, produce reference data from the existing Python implementation. Create `spike/generate_references.py`.

**Fixture corpus — two tiers:**

**Tier 1: Curated edge cases (inline in `spike/fixtures/parse_targets/`):**
1. Simple function (3-5 lines, no docstring)
2. Function with a docstring
3. Class with two methods (dotted qualified_name test)
4. `async def` function
5. Nested function inside a class method
6. Decorated function (single `@decorator`)
7. Decorated async function (stacked decorators)
8. Syntax edge cases: walrus operator, match statement, multiline typed signature with defaults
9. Lambda inside a function default argument
10. Comprehension with shadowed names
11. Function with comments (to measure comment-preservation impact)
12. Near-max-token function (~256 tokens)
13. Empty/minimal function (`def f(): pass`)
14. Function with a `# comment` line between the `def` line and the docstring — tests that the tree-sitter docstring detector skips comment CST nodes when searching for the first expression statement

**Tier 2: Real functions sampled from the codebase (at least 20):**
Extract functions from `src/clonehunter/` using the existing `extract_functions`, sampling:
- `similarity/lexical.py::lexical_similarity` (simple, tokenization-heavy)
- `similarity/rollup.py::rollup_findings` (long, complex)
- `similarity/candidates.py::_retrieve_matches` (the scoring heart)
- `embedding/codebert_embedder.py::CodeBertEmbedder.embed` (embedding logic)
- `snippets/normalization.py::strip_docstrings` (meta — normalization of normalization)
- `parsing/python_ast.py::extract_functions` (the function we're replacing)
- `io/fs.py::collect_files`
- At least 13 more functions to reach 20+ real-code fixtures

This gives ~33+ fixtures with ~500+ pairwise comparisons — sufficient to catch near-threshold instability.

**Tier 3: Adversarial/edge-case fixtures (at least 5):**
- Invalid Python syntax (the extractor must return `[]`)
- Comment-only variant of a tier-2 function (same code, different comments — to test comment-stripping impact)
- Unicode identifiers and comments
- CRLF line endings
- File with no trailing newline

This gives ~39+ fixtures with ~700+ pairwise comparisons.

**Reference script requirements:**
- **Must use `CodeBertEmbedder` directly** (not the stub). Do not set `CLONEHUNTER_EMBEDDER=stub`. The `embedding` field in output must be 768-dimensional.
- **Must force CPU device** — pass `device='cpu'` to `CodeBertEmbedder` (or `CodeBertConfig(device='cpu')`). The memo must record the device used. GPU/MPS reference values would measure hardware differences, not Python-vs-Rust differences.
- Pin the model to a specific HuggingFace revision SHA (not `main`). Record this SHA — it becomes the Rust default.

**Reference script outputs:**

- `spike/fixtures/python_embeddings.json` — list of `{text, embedding: [768 floats], token_ids: [int...]}`.
- `spike/fixtures/python_cosines.json` — full pairwise cosine similarity matrix.
- `spike/fixtures/python_functions.json` — list of `{file, qualified_name, start_line, end_line, is_async, code}`.
- `spike/fixtures/python_normalized.json` — list of `{original, normalized}`.
- `spike/fixtures/python_lexical_scores.json` — pairwise lexical similarity (Jaccard) on normalized text.

#### Step 1.2: T1a — Candle RoBERTa/CodeBERT numeric parity

In `spike/src/candle_embed.rs`:

**Critical: `microsoft/codebert-base` is a RoBERTa-family model** (`architectures: ["RobertaModel"]`, `model_type: "roberta"`, `type_vocab_size: 1`, special tokens `<s>`/`</s>`/`<pad>`). The spike must use candle's RoBERTa implementation path, not BERT. If `candle-transformers` does not have a RoBERTa-compatible code path, this is a NO-GO condition.

1. Download `microsoft/codebert-base` weights via `hf_hub::api::sync::Api` at the pinned revision SHA. Obtain `model.safetensors`, `config.json`, `tokenizer.json`.
2. Load the tokenizer via the `tokenizers` crate from `tokenizer.json`.
3. Load the model via candle's RoBERTa/BERT path (RoBERTa is architecturally BERT with different tokenization/pretraining; the forward pass is identical if token type IDs are handled correctly — RoBERTa uses `type_vocab_size: 1`, all zeros).
4. Tokenize each fixture snippet with `padding=True, truncation=True, max_length=256`. Verify token IDs match `python_embeddings.json`'s `token_ids` exactly.
5. Run forward pass on CPU, extract `last_hidden_state`.
6. Mean-pool with attention mask: `masked = hidden * mask.unsqueeze(-1); summed = masked.sum(dim=1); counts = mask.sum(dim=1).clamp(min=1); pooled = summed / counts`.
7. Compare each embedding dimension against `python_embeddings.json`:
   - Compute **max absolute per-dimension difference** across all snippets.
   - Compute **mean absolute difference** across all dimensions/snippets.
8. Compute pairwise cosine similarity matrix from the Rust embeddings.
9. Compare against `python_cosines.json`:
   - **Max absolute cosine difference** across all pairs.
   - Identify any pair where the cosine difference > 1e-4.
   - For near-threshold pairs (cosine within 0.02 of any threshold value 0.90/0.92), flag whether the Rust cosine is on the same side of the threshold.
10. **Determinism check:** embed all fixtures twice (same device, same run) and assert identical results.
11. **Batch invariance check:** embed each fixture individually and in a batch of all fixtures; assert results match within f32 epsilon.

**Spike verdict architecture (`main.rs`):**

The spike does NOT use `assert!` as its top-level control flow. Instead, each subcheck (T1a, T1b, T1c) returns a structured verdict:

```rust
enum Verdict { Pass, Conditional(String), Fail(String), Inconclusive(String) }
```

All three subchecks run to completion regardless of individual failures. After all subchecks finish, `main.rs` prints a complete summary and maps the overall verdict to exit codes:
- `0` = GO (all Pass) or CONDITIONAL GO (any Conditional, none Fail)
- `1` = NO-GO (any Fail)
- `2` = INCONCLUSIVE (tooling/infrastructure failure — download error, corrupt fixture, tokenizer load crash — distinct from a measured design NO-GO)

T1a verdicts:
- `Fail` if `max_cosine_diff >= 1e-3` OR `token_id_match_rate < 1.0` OR determinism fails OR no RoBERTa code path
- `Conditional` if `max_cosine_diff` in [1e-4, 1e-3) — prints: `"T1a CONDITIONAL GO: max cosine diff {:.2e} in [1e-4, 1e-3) — orchestrator sign-off required before T2"`
- `Pass` otherwise

Determinism and batch-invariance are subverdicts within T1a:
```rust
// determinism: same input twice => identical embeddings
if !determinism_pass { return Verdict::Fail("non-deterministic embeddings".into()); }
// batch invariance: single vs batch embedding diff
if batch_invariance_max_diff >= f32::EPSILON * 10.0 {
    return Verdict::Fail(format!("batch invariance violation {batch_invariance_max_diff:.2e}"));
}
```

#### Step 1.3: T1b — Tree-sitter vs stdlib AST function extraction

In `spike/src/treesitter_parse.rs`:

1. Parse each fixture Python file with `tree-sitter-python`.
2. Extract functions by walking the CST for `function_definition` and `class_definition` nodes:
   - Build the qualified name stack (class.method dotted path) matching Python's `Visitor` pattern.
   - Handle `async` functions.
   - Emit nested functions individually.
   - **Explicitly skip `lambda` nodes** — Python's AST extractor only emits `FunctionDef`/`AsyncFunctionDef`, never lambdas.
3. Extract `start_line`, `end_line` for each function.
4. Extract `code` text by slicing the source from `start_line` to `end_line` (1-indexed, inclusive).
5. **Compare against `python_functions.json` on three dimensions:**
   - **Qualified name match rate** — must be 100%.
   - **Line span differences** — document each divergence.
   - **Extracted `code` text match** — byte-for-byte comparison. This is critical because `code` feeds `code_hash`, normalization, snippet generation, expansion, and reporting.

**Known divergences to investigate:**
- **Decorator handling:** CPython `ast.FunctionDef.lineno` points at the `def` keyword. Tree-sitter wraps decorated functions in a `decorated_definition` node. The spike must verify that the `function_definition` child node's `start_position` is at the `def` line, not the `@` line.
- **`end_lineno`:** CPython's `end_lineno` is the last line of the function body (1-indexed). Tree-sitter's `end_position.row` is 0-indexed. Verify: `end_position.row + 1` should equal `ast.end_lineno`.
- **Comprehension scoping:** Verify these don't introduce spurious qualified-name segments.
- **Multiline typed signatures:** Verify the span includes the full signature but starts at `def`.

**T1b verdicts:**
- `Fail` if `qualified_name_match_rate < 1.0` OR `code_text_match_rate < 1.0` after applying any documented span adjustments (e.g., decorator offset normalization)
- `Conditional` if code text matches only after a documented, named adjustment (prints: `"T1b CONDITIONAL GO: code text matches after {adjustment_name} — orchestrator sign-off required"`)
- `Pass` if all match byte-for-byte with no adjustments needed

**GO/NO-GO criteria for T1b (concrete):**
- **PASS:** Qualified name match rate = 100%; line spans match exactly for undecorated functions; decorated functions resolved by extracting from `function_definition` child; extracted `code` matches byte-for-byte with no adjustments.
- **CONDITIONAL GO:** Qualified name match = 100%; code text matches byte-for-byte only after a documented constant adjustment (e.g., decorator offset, trailing whitespace). Each adjustment must be named, have exact downstream impact documented, and be applied as a deterministic transformation in T5.
- **NO-GO:** Any qualified name mismatch; any code text mismatch that cannot be resolved to a named, constant adjustment; any span divergence that cannot be expressed as a constant offset.

#### Step 1.4: T1c — Normalization definition and impact

In `spike/src/normalize.rs`:

1. Implement the proposed Rust normalization per DD7:
   - Parse each fixture function's code with tree-sitter.
   - Identify docstrings: `expression_statement` child at body position 0 whose child is a `string` node.
   - Replace the docstring with `pass` (matching Python's `ast.Pass()` replacement behavior).
   - Return the remaining source (preserving whitespace/formatting/comments).
2. Compare against `python_normalized.json` (which has `ast.unparse` output):
   - Compute character-level diff between Rust-normalized and Python-normalized for each function.
   - Categorize differences: (a) comment preservation, (b) whitespace/indentation, (c) parenthesization, (d) quote style, (e) `pass` placeholder handling.
3. **Embedding impact assessment:**
   - Use the Rust candle embedder (from Step 1.2) to embed both Rust-normalized and Python-normalized versions.
   - Compute cosine similarity between the two normalization variants for each snippet.
   - If cosine > 0.99 for all snippets, normalization difference is negligible for detection.
   - If any cosine < 0.95, document which cases and whether it's acceptable for re-freeze.
4. **Lexical impact assessment:**
   - Compute Jaccard lexical similarity (identifier tokens) on Rust-normalized text.
   - Compare against `python_lexical_scores.json`.
   - Document max divergence and whether it crosses any gating threshold (0.5 `lexical_min_ratio`, 0.90/0.92 composite thresholds).
5. **Threshold stability check:**
   - For any pair where either the Python or Rust composite score is within 0.02 of a threshold (0.90, 0.92), verify both land on the same side.

**T1c verdicts:**
- `Fail` if any normalization cosine < 0.90 OR any lexical score divergence > 0.10 OR any near-threshold pair flips sides
- `Conditional` if any normalization cosine in [0.90, 0.99) or lexical divergence in (0.05, 0.10] — prints: `"T1c CONDITIONAL GO: normalization impact {metric} — orchestrator sign-off required"`
- `Pass` if all normalization cosines > 0.99 and lexical divergence < 0.05 and no threshold flips

#### Step 1.5: Write the decision memo

Create `23_PHASE0_SPIKE_MEMO.md` in the repo root with:

```markdown
# Phase 0 Parity Spike — Decision Memo

## Environment
- Crate versions: [list from Cargo.lock]
- Model revision SHA: ___
- Device: CPU
- Fixture corpus: ___ snippets (tier 1 edge cases + tier 2 real code)
- Fixture file SHA-256 hashes: [list]
- Exact invocation: `cd spike && cargo run --release`

## T1a: Candle RoBERTa/CodeBERT Numeric Parity
- Tokenizer ID match rate: ___
- Max per-dimension absolute difference: ___
- Mean absolute difference: ___
- Max pairwise cosine difference: ___
- Pairs exceeding 1e-4 cosine tolerance: ___ / ___
- Near-threshold pair stability: [all stable / N pairs flip]
- Determinism: [PASS / FAIL]
- Batch invariance: [PASS / FAIL]
- **Verdict:** [PASS / CONDITIONAL PASS / FAIL]

## T1b: Tree-sitter vs AST Function Extraction
- Qualified name match rate: ___
- Code text match rate: ___
- Line span divergences: [none / list with adjustments]
- Decorator handling: [spans match at function_definition / offset documented]
- Lambda exclusion: [confirmed / issue]
- **Verdict:** [PASS / CONDITIONAL PASS / FAIL]

## T1c: Normalization
- Strategy: tree-sitter docstring strip (→ pass) + source passthrough
- Categories of difference vs ast.unparse: [list]
- Embedding cosine (Rust-norm vs Python-norm): min ___, mean ___, max ___
- Lexical score divergence: max ___, mean ___
- **Verdict:** [PASS / CONDITIONAL PASS / FAIL]

## GO / NO-GO
- [ ] GO — proceed with candle + tree-sitter + re-freeze
- [ ] CONDITIONAL GO — proceed with mitigations: ___ (requires orchestrator sign-off before T2)
- [ ] NO-GO — escalate to orchestrator: ___

## Escalation risks
- [List any finding that would force escalation to orchestrator]
```

#### Step 1.6: GO/NO-GO criteria

- **GO:** All three verdicts PASS. Token IDs match exactly. Max cosine difference < 1e-4. Tree-sitter qualified names and code text match 100%. Normalization cosine impact > 0.99 for all snippets. Determinism and batch invariance pass.
- **CONDITIONAL GO:** Cosine differences in 1e-4 to 1e-3 range but no near-threshold pair flips. Tree-sitter spans differ on decorated functions only, with a documented constant adjustment. Normalization differences are categorized and do not cause lexical score changes > 0.05. **A CONDITIONAL GO requires explicit sign-off from the orchestrator before proceeding to T2.** Escalation mechanism: the implementer commits `23_PHASE0_SPIKE_MEMO.md` with a `## ESCALATION` section detailing the CONDITIONAL findings, posts a comment on GitHub issue #23, and the orchestrator must update `ORCHESTRATION.md`'s LOG with an explicit GO acknowledgement before T2 begins.
- **NO-GO (escalate):** Cosine differences > 1e-3 OR near-threshold pairs flip. Any qualified name mismatch. Any span divergence that cannot be expressed as a constant offset. Candle has no working RoBERTa code path. Normalization causes any embedding cosine < 0.90.

---

## Testing

### T0 validation
- `cargo fmt --check` — passes (code is formatted).
- `cargo clippy --all-targets -- -D warnings` — passes (no warnings, including test code).
- `cargo test` — passes:
  - `ConfigError` display strings include context.
  - `init_logging` is idempotent.
  - All module stubs compile.

### T1 validation
- `spike/generate_references.py` runs against the existing Python implementation (CodeBERT embedder, CPU device) and produces fixture JSON files.
- `cd spike && cargo run --release` runs all three subchecks (T1a, T1b, T1c) to completion regardless of individual failures. Each returns a structured `Verdict` (Pass/Conditional/Fail/Inconclusive).
- Exit codes: `0` = GO or CONDITIONAL GO, `1` = NO-GO, `2` = INCONCLUSIVE (tooling failure).
- On CONDITIONAL GO, the spike prints explicit warnings identifying which metrics are in the conditional range and reminding that orchestrator sign-off is required.
- Results are transcribed into `23_PHASE0_SPIKE_MEMO.md` with the concrete numbers.
- The memo's GO/NO-GO is checked before proceeding to T2. CONDITIONAL GO triggers the escalation mechanism (issue #23 comment + ORCHESTRATION.md LOG update).

### Ongoing gates (all phases)
Every change must pass all three gates before merge:
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

---

## Constraints on later phases

The following Phase 0 decisions constrain T2–T15:

1. **Error strategy (DD3):** Module-local error types with `thiserror`. Each subsystem defines its own error enum. Orchestration layers (T10 `engines/pipeline`, T12 `cli`) define thin boundary enums with `#[from]` wrappers. Recoverable degradations are not propagated through errors — instead, successful results carry a `Vec<Degradation>` diagnostics list (logged via `tracing::warn!`, surfaced in stats). `main.rs` uses `anyhow`.
2. **Logging (DD4):** All modules use `tracing::{info, warn, debug, error}` macros. `init_logging()` is called once in `main.rs`. No module initializes its own subscriber.
3. **Normalization contract (DD7):** T6 produces two text forms from the normalized source:
   - **Analysis text** (drives embeddings, `SnippetRef.text`, cache keys, lexical scoring): docstring→`pass`, **comments stripped**, whitespace preserved, decorators excluded.
   - **Display text** (used only by reporters for rendered diffs): docstring→`pass`, comments preserved, whitespace preserved, decorators excluded.
   The analysis text is what `SnippetRef.text` carries into the pipeline; the display text is a separate field or derived on demand in the reporter. If the spike produces CONDITIONAL GO with mitigations, those mitigations apply to the analysis text form.
4. **Module layout (DD2):** T2–T12 each implement their module(s) within the declared structure. `core/` is leaf-only (types, config, errors, logging, fingerprints). Pipeline orchestration lives in `engines/`. No new top-level modules unless justified.
5. **Single crate (DD1):** All code lives in one crate. If a workspace split becomes necessary (unlikely), it's a separate decision.
6. **MSRV 1.85 (DD6):** No nightly-only features. All dependencies must build on 1.85.0.
7. **Stub embedder (cross-task theme):** T7 (embedding) must implement a `StubEmbedder` activated by `CLONEHUNTER_EMBEDDER=stub` env var. The stub must be deterministic (SHA-256-based 16-dim vector matching `stub_embedder.py`) and run the entire detection pipeline without candle weights. Most tests rely on it.
8. **Reproducibility (DD6):** `Cargo.lock` is committed. Model pinned to a specific HuggingFace revision SHA. Parity baseline produced on CPU.
9. **Visibility (DD2):** Default to `pub(crate)`. Only re-export what tests or the binary need. Traits (`VectorIndex`, `Engine`) are defined in their subsystem modules, not a separate `model/` bucket.
10. **Lambda exclusion (T1b):** T5 extraction explicitly whitelists `function_definition`/`class_definition` and never emits lambdas.
11. **Parity device policy:** CPU is the parity baseline. GPU/Metal is best-effort behavior, not part of the frozen contract.
