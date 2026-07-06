# CloneHunter

CloneHunter finds duplicate code across mixed-language repositories and emits evidence-rich HTML/JSON/SARIF reports. It is a semantic retrieval pipeline, not a grep: Python files are parsed to functions (tree-sitter AST), every file is also windowed, snippets are embedded with a transformer (CodeBERT via candle), neighbors are retrieved by vector similarity (brute-force), and each candidate is re-scored by blending embedding similarity with lexical (identifier-Jaccard) similarity before rollup into findings.

Binary entry point: `[[bin]] name = "clonehunter" path = "src/main.rs"` ([Cargo.toml](Cargo.toml)); the library crate is `src/lib.rs`. Rust edition 2024, MSRV 1.85.0.

## CLI surface

Two subcommands, both defined with clap derive in [src/cli/mod.rs](src/cli/mod.rs). The full flag list is in [README.md](README.md); the load-bearing shape:

```
clonehunter scan [PATHS...] [--format json|html|sarif] [--out FILE]   # default format: html; default out: clonehunter_report.<ext>
    --engine semantic|sonarqube   --embedder codebert|faster|stub   --index brute|faiss   --device auto|cpu|mps|cuda
    --threshold-func/-win/-exp FLOAT   --min-window-hits INT   --lexical-min-ratio/-weight FLOAT
    --window-lines/-stride-lines/-min-nonempty INT   --expand-calls [--expand-depth/-max-chars INT]
    --cache-path PATH   --cluster [--cluster-min-size INT]
    --repotype <lang>...   --include-globs GLOB...   --exclude-globs GLOB...     # repeatable; layered (see below)

clonehunter diff --base REF [--format ...] [--out FILE] [--engine/-embedder/-index/-device ...]
```

**`scan`** carries the full tuning surface. **`diff`** carries only `--base` (default `HEAD`) plus the common report/override args. Every override field is `Option<T>` so it only overrides config/defaults when actually passed.

## End-to-end flow

`run_scan` → `get_engine(config.engine).scan(...)` → for the semantic engine, [src/engines/pipeline.rs](src/engines/pipeline.rs) `run_pipeline`, in this exact order (each stage timed):

1. **collect files** — [src/io/fs.rs](src/io/fs.rs) `collect_files(paths, include_globs, exclude_globs)` → `Result<Vec<FileRef>>`. `.py` → `Language::Python`, everything else → `Language::Text`.
2. **extract units** — python files → `extract_functions` (tree-sitter CST walk) go into *both* `python_functions` and `window_units`; every other file → one whole-file unit into `window_units` only.
3. **generate snippets** — FUNC (one per function) + WIN (sliding windows over every unit) + EXP (call-expansion, only if `expansion.enabled`), concatenated into one list. Each snippet's `text` is **tree-sitter comment-stripped** (analysis text); `display_text` keeps comments.
4. **embed** — `StubEmbedder` if `embedder.name==stub` else `CodeBertEmbedder` (candle XLMRobertaModel); results memoized in SQLite [src/embedding/cache.rs](src/embedding/cache.rs). Only cache-misses are embedded, in `batch_size` batches.
5. **similarity** — build the brute index, `retrieve_candidates` → `rollup_findings` → optional clustering.
6. **assemble** — `ScanResult { findings, stats, config_snapshot, timing, degradations }` → the matching reporter.

## Repo layout

- [src/core/](src/core/) — the spine.
  - `types.rs` — all data types: `FileRef`, `FunctionRef` (`identity() = "{path}:{qname}:{start}:{end}"`), `SnippetRef` (`kind` ∈ Func/Win/Exp, `text` = analysis, `display_text` = display, `snippet_hash = hash_text(text)`), `Embedding` (f32 vec), `CandidateMatch`, `Finding`, `ScanStats`, `ScanResult`, `ScanRequest`, `Language`.
  - `config.rs` — `CloneHunterConfig` nested structs + enums + repotype presets + `EMBEDDER_PRESETS`.
  - `config_loader.rs` — `ConfigOverride`, `load_config` layers **defaults → `clonehunter.toml` → CLI overrides**, `validate_config` (enum membership, unit intervals). `find_config_root` walks up from any path looking for `clonehunter.toml`.
  - `errors.rs` (`CloneHunterError`/`ConfigError`), `logging.rs` (tracing subscriber init).
- [src/io/](src/io/) — `fs.rs` (`collect_files`: globset+walkdir, early dir pruning, canonical-path dedupe). `git.rs` (`changed_files(base, paths, cwd)` = `git diff --name-only <base>` ∪ `git ls-files --others`). `fingerprints.rs` (`hash_text` = SHA-256 hex, `embed_cache_key = sha256("{model}:{revision}:{max_tokens}:{snippet_hash}")`).
- [src/parsing/](src/parsing/) — `python_ast.rs` (`extract_functions`: tree-sitter CST walk with class/func name stack for dotted qualified name, handles `async def`, nested functions, decorators; swallows all parse errors → `[]`). `text_units.rs` (non-python file → one whole-file `FunctionRef`).
- [src/snippets/](src/snippets/) — `normalization.rs` (`normalize_analysis`: tree-sitter comment-strip + passthrough; `normalize_display`: docstrings→pass, comments preserved). `generators.rs` (`generate_function_snippets` FUNC, `generate_window_snippets` WIN — window emitted only when non-empty-line count ≥ `min_nonempty`). `expansion.rs` (`expand_calls` BFS-inlines called helper bodies up to `depth`/`max_chars`; resolves names, `self`/`cls` methods, local classes, imports, constructors via per-file ImportMap).
- [src/embedding/](src/embedding/) — `codebert.rs` (`CodeBertEmbedder`: candle `XLMRobertaModel`, bundled `tokenizer.json` @ pinned CODEBERT_REVISION, local HF cache lookup, CPU fallback on error; mean-pools `last_hidden_state` with attention mask). `stub.rs` (`StubEmbedder`: deterministic 16-dim SHA-256 embedder using f64 arithmetic for parity). `cache.rs` (`EmbeddingCache`: SQLite WAL, self-healing on version mismatch/corruption, chunked IN reads, legacy JSON migration). `mod.rs` (`Embedder` trait, `create_embedder`, `embed_with_cache`).
- [src/index/](src/index/) — `brute.rs` (cosine via ndarray f32 matrix → f64 cast, `argsort` with `partial_cmp` stable sort). `mod.rs` (`VectorIndex` trait: `build(&mut self, ids, embeddings)` + `query(&self, embedding, top_k) -> Vec<(String, f64)>`).
- [src/similarity/](src/similarity/) — **the heart** (same logic as Python, rayon replaces multiprocessing).
  - `candidates.rs` (`retrieve_candidates`: rayon par_iter, shared `&dyn VectorIndex` (pre-built by pipeline); applies composite score + lexical gate + per-kind threshold; **skips `neighbor_id == snip.snippet_hash`** self-match).
  - `lexical.rs` (`lexical_similarity` = Jaccard over lowercased `[A-Za-z0-9_]+` tokens).
  - `scoring.rs` (`best_score`).
  - `ranking.rs` (`kind_rank`, `best_match` — deterministic tie-break via `to_bits()`).
  - `rollup.rs` (`rollup_findings`: filter-overlap → filter-lexical → dedupe → normalize a/b orientation → group by function pair → emit only if ≥1 reason; `_duplicated_lines`).
  - `occurrences.rs` (`SelfCloneOccurrences` — union-find over overlapping spans; `covered_lines` uses adjacency; `occurrence_for` is `&self`).
  - `clustering.rs` (union-find over `function.identity`; only runs when `cluster_findings` is on).
- [src/reporting/](src/reporting/) — `schema.rs` (`SCHEMA_VERSION = env!("CARGO_PKG_VERSION")`). `compare.rs` (`select_compare` → `best_match` for rendering). `json.rs` (`write_json`: `{schema_version, findings, stats, config, timing}`, each finding with a `similar`-crate unified diff). `sarif.rs` (`write_sarif`: SARIF 2.1.0, `note`-level results). `html.rs` (`write_html`: self-contained inline CSS/JS, `DiffOp` side-by-side diff, client-side sort; self-clone aware via `SelfCloneOccurrences`).
- [src/engines/](src/engines/) — `pipeline.rs` (`run_pipeline`: the 6-stage impl). `semantic.rs` (one-line delegate). `sonarqube.rs` (adapter: reads `CLONEHUNTER_SONAR_REPORT` env var, maps `duplications[]` → `Finding`s with `score=1.0`; no embedding/index). `mod.rs` (`get_engine`, `PipelineError`).
- [src/cli/](src/cli/) — `mod.rs` (clap derive; `Commands::Scan(Box<ScanArgs>)` boxed to avoid large-enum-variant; `run_scan` = build overrides → `resolve_config_root` walk-up → `load_config` → two-pass glob merge → engine.scan → reporter; `run_diff` = `changed_files` → full scan → filter findings to changed paths → reporter). `glob_merge.rs` (`REPO_TYPE_PRESETS`, `effective_repotypes`, `resolve_repotype_globs`, `merge_globs`, `validate_repotype`).
- [src/main.rs](src/main.rs), [src/lib.rs](src/lib.rs).

## Scoring, thresholds & config layering

**Composite score** ([src/similarity/candidates.rs](src/similarity/candidates.rs)): `composite = (1 − lexical_weight)·embedding + lexical_weight·lexical`. A candidate is kept when `lexical ≥ lexical_min_ratio` **and** `composite ≥` the per-kind threshold (Func→`func`, Win→`win`, else→`exp`).

**Config defaults** ([src/core/config.rs](src/core/config.rs)): `engine="semantic"`; thresholds `func=0.92, win=0.90, exp=0.90, min_window_hits=1, lexical_min_ratio=0.5, lexical_weight=0.3`; windows `window_lines=40, stride=6, min_nonempty=4`; expansion `enabled=false, depth=1, max_chars=4000`; index `name="brute", top_k=25`; embedder `name="codebert", model="microsoft/codebert-base", revision=<pinned SHA>, max_length=256, batch_size=16, device="auto"`; cache `~/.cache/clonehunter`; `include_globs=["**/*.py"]`; `cluster_findings=false, cluster_min_size=2`.

**Glob layering** (`scan` only, applied after `load_config` in [src/cli/mod.rs](src/cli/mod.rs)): when `--repotype` is explicitly passed, the repotype preset **replaces** the config's include_globs entirely; when `--repotype` is omitted, the `monorepo` expansion is merged on top of config globs. Then `--include/--exclude-globs` are merged as the final CLI layer, with conflicts resolved in favour of the CLI layer. `--repotype none` produces empty include_globs → 0 files collected.

## Design principles

Grounded in how the code actually behaves — respect these when changing it:

- **Determinism/parity is a first-class constraint.** The brute index sorts stably, `best_match` and `SelfCloneOccurrences` are order-independent, the stub embedder is deterministic. This exists because outputs are a frozen contract (below). Do not introduce nondeterminism into detection.
- **Config overrides are additive and safe.** Unset CLI fields are `None` and dropped; nested override sub-structs are only set when the user touched their option-group. Never let an unset flag clobber a config/default value.
- **One canonicalization point per concept.** a/b pair orientation is normalized in exactly one place (`rollup._normalize_orientation`); analysis text is normalized once (`normalize_analysis`). Everything downstream depends on these — don't add a second.
- **Degrade gracefully, never crash the scan.** CUDA/MPS→CPU, missing progress display→no-op, unparseable file→skipped, corrupt/old cache→self-heal. New external dependencies should follow suit.
- **Testable without heavy deps.** `CLONEHUNTER_EMBEDDER=stub` runs the entire detection pipeline without downloading model weights; all integration tests rely on it. Keep detection logic independent of the concrete embedder.

## Known limitations & gotchas

- **`diff` and `scan` do not scan the same files.** `diff` skips repotype/glob merging entirely and calls `load_config(cwd, ...)`, so it obeys only the config default `**/*.py` — whereas a bare `scan` scans all languages via the `monorepo` default. Cross-language diff needs explicit config.
- **`lexical_min_ratio` gates twice** — in candidate retrieval *and* again in rollup. `min_window_hits` is not a per-match filter; it earns a finding the `min_window_hits` *reason*, and a finding is emitted only if it has ≥1 reason.
- **Self-match filtering is two-layered:** retrieval skips a snippet matching its own `snippet_hash`; rollup keeps *same-function* self-clones only when line ranges are disjoint and drops *same-file cross-function* range overlaps as containment.
- **Finding *order* is not stable across rayon runs** (par_iter is unordered); scores/pairs are identical, only ordering drifts. The benchmark sorts before comparing — do the same in any parity check.
- **Non-python files get WIN snippets only** (FUNC/EXP derive from python functions), so cross-language detection is window-based; `stats.function_count` counts python functions only.
- **`--index faiss`** is not implemented; the flag is accepted for CLI compatibility but always uses the brute index.
- **`faster` embedder preset** uses the XLM-RoBERTa architecture; loading BERT-family (MiniLM) weights may fail at runtime. Use `codebert` (the default) if `faster` fails.
- **f32 precision.** Rust uses f32 for embedding arithmetic (candle default); Python used f64 (torch default). Near-equal embeddings (cosine ~1.0) may produce different top-k ordering. The re-frozen `benchmark/baseline.json` is the Rust detection contract.
- **Normalization differs from Python `ast.unparse`.** Rust strips tree-sitter comment nodes; Python normalized via `ast.unparse`. The re-frozen baseline documents all divergences (Cat-A through Cat-E).
- **Snippet text ≠ source.** Embeddings, hashes, lexical tokens, and the rendered diff all operate on the normalized (comment-stripped) form; `FunctionRef.code` keeps the original.
- **Report format contract:** `tests/snapshots/` golden files lock the JSON/SARIF schema. After any intentional schema change: `INSTA_UPDATE=new cargo test --test golden_fixtures` then review and accept the new snapshots.
- **`--embedder mlx`** requires `--features mlx` build with a prebuilt MLX library
  (Apple Silicon only). NOT a single binary — requires libmlx.dylib + mlx.metallib
  sidecar (~101 MB). Setup: `./scripts/setup-mlx.sh`. Fastest backend (~46s click,
  beats PyTorch-MPS) with exact frozen-baseline parity and best numerics (2.88e-12).

## Working in this repo

This project uses **cargo**. There is no CI workflow for pushes — local validation is the gate. Run all four before declaring a change done:

```bash
~/.cargo/bin/cargo fmt --check              # format (rustfmt)
~/.cargo/bin/cargo clippy --all-targets -- -D warnings   # lint
~/.cargo/bin/cargo test                     # tests (~268)
~/.cargo/bin/cargo build --release          # verify release build
```

- **Fast dev loop without model download:** `CLONEHUNTER_EMBEDDER=stub cargo run -- scan .`. The stub embedder is deterministic and covers the detection pipeline; reach for `codebert` only when embedding quality is under test.
- **Run with real embedder:** `cargo run --release -- scan . --format html` (downloads ~440 MB on first run, cached in `~/.cache/huggingface/hub/`).
- **Update golden snapshots** after an intentional schema change: `INSTA_UPDATE=new cargo test --test golden_fixtures`.
- **Env vars:** `CLONEHUNTER_EMBEDDER=stub` (force stub); `CLONEHUNTER_SONAR_REPORT=<path>` (required by the `sonarqube` engine); `MLX_SYS_PREBUILT=<dir>` (prebuilt MLX library location for `--features mlx` build).
- **Merge convention:** PRs are **squash-merged to `master`**.

## The frozen Rust baseline (detection contract)

Tag **`rust-baseline`** and the tracked file [benchmark/baseline.json](benchmark/baseline.json) are the parity contract for the Rust implementation. They supersede the Python baseline at `v1.1.0-python-baseline`.

- [benchmark/run_benchmark.py](benchmark/run_benchmark.py) is a Python harness that clones four pinned repos (click 8.1.8, requests 2.32.3, attrs 24.3.0, rich 13.9.4) and calls the **Rust binary** for each scan. It records sorted `finding_scores` (6dp) and `finding_pairs` per repo. The harness uses `CLONEHUNTER_BINARY` env var to locate the binary (defaults to `target/release/clonehunter`).
- `--save-baseline` writes `baseline.json` (keys: `timestamp`, `environment`, `results`; `environment` captures `rust_version`, `candle_version`, `resolved_device`, full config, hardware). `--compare-baseline` requires **detection counts and finding pairs to match exactly, scores within 1e-4**; timing is informational.
- **Any change that alters detection output** (candidate generation, scoring, thresholds, rollup, expansion, parsing) must be deliberate: regenerate the baseline with `run_benchmark.py --save-baseline`, confirm the diff is fully explained by your change, and re-freeze.

Known divergences from Python baseline (fully explained, Cat-A through Cat-E):
- **Cat-A** — comment stripping in analysis text (tree-sitter vs `ast.unparse`)
- **Cat-B** — f32 vs f64 embedding precision (~1e-4 cosine delta)
- **Cat-C** — tree-sitter extraction differences (minor)
- **Cat-D** — `config_snapshot` format differences
- **Cat-E** — file collection globset vs Python `os.walk` corner cases
