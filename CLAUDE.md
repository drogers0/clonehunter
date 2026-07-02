# CloneHunter

CloneHunter finds duplicate code across mixed-language repositories and emits evidence-rich HTML/JSON/SARIF reports. It is a semantic retrieval pipeline, not a grep: Python files are parsed to functions (AST), every file is also windowed, snippets are embedded with a transformer (CodeBERT by default), neighbors are retrieved by vector similarity (brute-force or FAISS), and each candidate is re-scored by blending embedding similarity with lexical (identifier-Jaccard) similarity before rollup into findings.

Console entry point: `clonehunter = clonehunter.cli.main:main` ([pyproject.toml](pyproject.toml)); `python -m clonehunter` runs the same `main` ([\_\_main\_\_.py](src/clonehunter/__main__.py)). Package requires Python ≥ 3.10; the dev checkout pins **3.13** ([.python-version](.python-version)). Runtime deps: numpy, torch, transformers, tqdm; `faiss-cpu` is an optional extra; `tomli` only on 3.10.

## CLI surface

Two subcommands, both defined by pure argparse wiring in [cli/main.py](src/clonehunter/cli/main.py) (no logic there). The full flag list is in [README.md](README.md); the load-bearing shape:

```
clonehunter scan [PATHS...] [--format json|html|sarif] [--out FILE]   # default format: html; default out: clonehunter_report.<ext>
    --engine semantic|sonarqube   --embedder codebert|faster|stub   --index brute|faiss   --device auto|cpu|mps|cuda
    --threshold-func/-win/-exp FLOAT   --min-window-hits INT   --lexical-min-ratio/-weight FLOAT
    --window-lines/-stride-lines/-min-nonempty INT   --expand-calls [--expand-depth/-max-chars INT]
    --cache-path PATH   --cluster [--cluster-min-size INT]
    --repotype <lang>...   --include-globs GLOB...   --exclude-globs GLOB...     # repeatable; layered (see below)

clonehunter diff --base REF [--format ...] [--out FILE] [--engine/-embedder/-index/-device ...]
```

**`scan`** carries the full tuning surface. **`diff`** deliberately carries none of the threshold/glob/repotype knobs — only `--base` (default `HEAD`) plus the common report/override args. Every override flag defaults to `None` so it only overrides config/defaults when actually passed ([cli/commands/overrides.py](src/clonehunter/cli/commands/overrides.py) drops `None`s via `clean_overrides` — this is what makes layering safe).

## End-to-end flow

`run_scan` → `get_engine(config.engine).scan(...)` → for the semantic engine, [core/pipeline.py](src/clonehunter/core/pipeline.py) `run_pipeline`, in this exact order (each stage timed):

1. **collect files** — [io/fs.py](src/clonehunter/io/fs.py) `collect_files(paths, include, exclude)` → `list[FileRef]`. `.py` → `python`, everything else → `text`.
2. **extract units** — python files → `extract_functions` (AST) go into *both* `python_functions` and `window_units`; every other file → one whole-file unit into `window_units` only.
3. **generate snippets** — FUNC (one per function) + WIN (sliding windows over every unit) + EXP (call-expansion, only if `expansion.enabled`), concatenated into one list. Each snippet's `text` is **docstring-stripped and `ast.unparse`d** ([snippets/normalization.py](src/clonehunter/snippets/normalization.py)).
4. **embed** — `StubEmbedder` if `embedder.name=="stub"` else `CodeBertEmbedder`; results memoized in the SQLite [embedding/cache.py](src/clonehunter/embedding/cache.py). Only cache-misses are embedded, in `batch_size` batches.
5. **similarity** — build the index (faiss with **brute fallback on `RuntimeError`**), `retrieve_candidates` → `rollup_findings` → optional `cluster_findings`/`filter_clusters`.
6. **assemble** — `ScanResult(findings, stats, config_snapshot, timing)` → the matching reporter.

## Repo layout

- [cli/](src/clonehunter/cli/) — `main.py` argparse only; `commands/scan.py` is the real orchestration (`run_scan`: build nested overrides only for touched option-groups → force engine registration via `__import__("clonehunter.engines")` → resolve config root by walking up to the nearest `pyproject.toml` → `load_config` → **two-pass glob merge** → `get_engine(...).scan(...)` → reporter). `commands/diff.py` (`run_diff`: `git` changed-files → full scan of paths → filter findings to those touching a changed file; **if nothing changed it re-runs scan with `paths=[]` → empty report**). `commands/overrides.py` maps flat CLI args → the nested override dict and honors `CLONEHUNTER_EMBEDDER=stub`.
- [core/](src/clonehunter/core/) — the spine. `pipeline.py` (`run_pipeline`, see flow above). `types.py` — all frozen/slots dataclasses: `FileRef`, `FunctionRef` (its `identity = "{path}:{qname}:{start}:{end}"` is **the pervasive grouping/dedupe key**), `SnippetRef` (`kind` ∈ FUNC/WIN/EXP, absolute line spans, normalized `text`), `Embedding`, `CandidateMatch`, `Finding`, `ScanStats`, `ScanResult`, `ScanRequest`. `config.py` — nested config dataclasses + defaults (see below) + `EMBEDDER_PRESETS`. `config_loader.py` — `load_config` layers **defaults → `[tool.clonehunter]` in pyproject → CLI overrides**, then `validate_config` (enum membership, positivity, `[0,1]` unit intervals). `errors.py` (`CloneHunterError`/`ConfigError`), `logging.py` (singleton `"clonehunter"` logger).
- [model/](src/clonehunter/model/) — `interfaces.py` defines six ABCs (`Engine`, `Extractor`, `SnippetGenerator`, `Embedder`, `VectorIndex`, `Reporter`) but **only `Engine` and `VectorIndex` are actually subclassed** — embedders/reporters/extractors/generators are duck-typed. `registry.py` is a name→factory map **for engines only** (`get_engine` raises `ConfigError` listing supported names); embedder and index are selected by hardcoded `if/else` in `pipeline.py`.
- [parsing/](src/clonehunter/parsing/) — `python_ast.py` (`parse_file` reads UTF-8 `errors="replace"`; `extract_functions` walks with a class/func name stack for dotted `qualified_name`, handles `async def`, emits nested functions individually, **swallows all parse errors → `[]`**). `text_units.py` (turns any non-python file into one whole-file `FunctionRef` — how non-python code enters windowing).
- [snippets/](src/clonehunter/snippets/) — `normalization.py` (`strip_docstrings` via `ast.NodeTransformer`, then `ast.unparse`, falling back to raw source on `SyntaxError`). `generators.py` (`generate_function_snippets`, `generate_window_snippets` — window emitted only when non-empty-line count ≥ `min_nonempty`, offsets mapped back to absolute lines). `expansion.py` (the most complex module: `expand_calls` BFS-inlines called helper bodies into a caller's text up to `depth`/`max_chars` so extract-method refactors still register as clones; resolves names/`self`·`cls` methods/local classes/imported functions/constructors via a per-file `ImportMap`).
- [embedding/](src/clonehunter/embedding/) — `codebert_embedder.py` (`resolve_device`: `auto` → mps→cuda→cpu; lazy torch/transformers import; **mean-pools `last_hidden_state` with the attention mask**; **falls back to CPU on `RuntimeError`**; serves both `codebert` and `faster` presets). `stub_embedder.py` (deterministic 16-dim SHA-256 embedder — the reason detection logic is testable without torch). `cache.py` (`EmbeddingCache`: SQLite WAL, schema-versioned with self-healing on version-mismatch/corruption, chunked `IN` reads, lazy migration from legacy `{key}.json`; **cache key excludes device/batch_size** — see fingerprints).
- [index/](src/clonehunter/index/) — `brute_index.py` (cosine via matrix mult; `np.argsort(-scores, kind="stable")` — the **stable sort is deliberate for parity**). `faiss_index.py` (exact `IndexFlatIP` when `N < nlist`, else approximate `IndexIVFFlat`; raises `RuntimeError` when faiss is absent → caught by the pipeline fallback; IVF is **approximate → not parity-stable**).
- [engines/](src/clonehunter/engines/) — `semantic_engine.py` (one-line delegate to `run_pipeline`). `sonarqube_engine.py` (an **adapter, not a detector**: reads a precomputed report path from `CLONEHUNTER_SONAR_REPORT`, maps `duplications[]` → `Finding`s with `score=1.0`; runs no embedding/index/similarity). Both register at import in `engines/__init__.py`.
- [similarity/](src/clonehunter/similarity/) — **the heart** (see scoring below). `candidates.py` (`retrieve_candidates`, multiprocessed across `cpu_count()-1` workers, each building its own full index; applies the composite score + lexical gate + per-kind threshold; **skips self-hash**). `lexical.py` (`lexical_similarity` = Jaccard over lowercased `[A-Za-z0-9_]+` identifier tokens). `scoring.py` (`best_score`). `ranking.py` (`kind_rank`, `best_match` — the representative pair, chosen with an **order-independent deterministic tie-break**). `rollup.py` (`rollup_findings`: filter-overlap → filter-lexical → dedupe → normalize a/b orientation → group by function pair → emit only if it has ≥1 reason; `_duplicated_lines`). `occurrences.py` (`SelfCloneOccurrences` — recent N-way self-clone aggregator; union-finds overlapping spans into connected components and counts `sum(lengths) − max(length)` per component to avoid over-counting chained self-clones). `clustering.py` (union-find over `function.identity`; only runs when `cluster_findings` is on).
- [reporting/](src/clonehunter/reporting/) — `schema.py` (`SCHEMA_VERSION` tracks the installed package version). `compare.py` (`select_compare` — **not** a report-vs-report diff; picks the single `best_match` evidence pair to render, shared by JSON+HTML). `json_reporter.py` (`{schema_version, findings, stats, config, timing}`, each finding with a unified `difflib` diff). `sarif_reporter.py` (SARIF 2.1.0, `note`-level results). `html_reporter.py` (self-contained inline CSS/JS, `SequenceMatcher`-opcode side-by-side diff, client-side sort; **self-clone aware** — threads `SelfCloneOccurrences` into evidence bounds and hidden-line markers).
- [io/](src/clonehunter/io/) — `fs.py` (`collect_files`: `os.walk` pruning excluded dirs early, custom `**` glob matching, dedupe by canonical path). `git.py` (`changed_files` = `git diff --name-only <base>` ∪ `git ls-files --others`; only `diff` uses it). `fingerprints.py` (`hash_text` = SHA-256; `embed_cache_key = hash("{model}:{revision}:{max_tokens}:{snippet_hash}")` — device/batch are intentionally *not* keyed).
- [\_compat/toml.py](src/clonehunter/_compat/toml.py) — stdlib `tomllib` (3.11+) with `tomli` fallback (3.10).

## Scoring, thresholds & config layering

**Composite score** ([similarity/candidates.py](src/clonehunter/similarity/candidates.py)): `composite = (1 − lexical_weight)·embedding + lexical_weight·lexical`. A candidate is kept when `lexical ≥ lexical_min_ratio` **and** `composite ≥` the per-kind threshold (FUNC→`func`, WIN→`win`, else→`exp`).

**Config defaults** ([core/config.py](src/clonehunter/core/config.py)): `engine="semantic"`; thresholds `func=0.92, win=0.90, exp=0.90, min_window_hits=1, lexical_min_ratio=0.5, lexical_weight=0.3`; **windows `window_lines=40, stride=6, min_nonempty=4`** (note: the README example and the benchmark both use `window_lines=12`, not the code default); expansion `enabled=False, depth=1, max_chars=4000`; index `name="brute", top_k=25, faiss_nlist=128, faiss_nprobe=8`; embedder `name="codebert", model="microsoft/codebert-base", revision="main", max_length=256, batch_size=16, device="auto"`; cache `~/.cache/clonehunter`; `include_globs=["**/*.py"]`; `cluster_findings=False, cluster_min_size=2`.

**Glob layering** (`scan` only, applied *after* `load_config` in [cli/commands/scan.py](src/clonehunter/cli/commands/scan.py)): pyproject globs → `--repotype` preset globs → explicit `--include/--exclude-globs`, with the most recent CLI layer winning conflicts. With no `--repotype`, scan defaults to the **`monorepo`** preset = the union of *all* language presets, so a bare `clonehunter scan` scans every language, not just `**/*.py`.

## Design principles

Grounded in how the code actually behaves — respect these when changing it:

- **Determinism/parity is a first-class constraint.** The brute index sorts stably, `best_match` and `SelfCloneOccurrences` are order-independent, the stub embedder is deterministic. This exists because outputs are a frozen contract (below). Do not introduce nondeterminism into detection.
- **Config overrides are additive and safe.** Unset CLI flags are `None` and dropped; nested override sub-dicts are injected only for option-groups the user touched. Never let an unset flag clobber a pyproject/default value.
- **One canonicalization point per concept.** a/b pair orientation is normalized in exactly one place (`rollup._normalize_orientation`, per-pair by `function.identity` then `start_line`); snippet text is normalized once (docstring-strip + unparse). Everything downstream depends on these — don't add a second.
- **Degrade gracefully, never crash the scan.** faiss→brute, CUDA/MPS→CPU, missing tqdm→no-op progress, unparseable file→skipped, corrupt/old cache→self-heal. New external dependencies should follow suit.
- **Testable without heavy deps.** `CLONEHUNTER_EMBEDDER=stub` runs the entire detection pipeline without torch; most tests rely on it. Keep detection logic independent of the concrete embedder.

## Known limitations & gotchas

- **`diff` and `scan` do not scan the same files.** `diff` skips repotype/glob merging entirely and calls `load_config(cwd, ...)`, so it obeys only the config default `**/*.py` — whereas a bare `scan` scans all languages via the `monorepo` default. Cross-language diff needs explicit config.
- **`lexical_min_ratio` gates twice** — in candidate retrieval *and* again in rollup. `min_window_hits` is not a per-match filter; it earns a finding the `min_window_hits` *reason*, and a finding is emitted only if it has ≥1 reason.
- **Self-match filtering is two-layered:** retrieval skips a snippet matching its own hash; rollup keeps *same-function* self-clones only when line ranges are disjoint and drops *same-file cross-function* range overlaps as containment.
- **Finding *order* is not stable across multiprocessed runs** (workers use `imap_unordered`); scores/pairs are identical, only ordering drifts. The benchmark sorts before comparing — do the same in any parity check.
- **Non-python files get WIN snippets only** (FUNC/EXP derive from python functions), so cross-language detection is window-based; `stats.function_count` counts python functions only.
- **`--index faiss` silently degrades to brute** when faiss is missing, and faiss IVF is approximate → not parity-stable. Parity work uses `--index brute`.
- **Snippet text ≠ source.** Embeddings, hashes, lexical tokens, and the rendered diff all operate on the normalized (docstring-stripped, unparsed) form; `FunctionRef.code` keeps the original.
- **`interfaces.py` is partly aspirational** — the `Embedder`/`Reporter`/`Extractor`/`SnippetGenerator` ABCs are not subclassed. Match the existing duck-typed shapes rather than assuming inheritance.

## Working in this repo

This project uses **uv**. There is no CI workflow — local validation is the gate. Run all four before declaring a change done:

```bash
uv run ruff format --check .   # format (line-length 100, py310 target)
uv run ruff check .            # lint (E,F,I,UP,B,SIM,C4,RUF)
uv run pyright                 # type-check — strict mode, all reportUnknown* on
uv run pytest                  # tests (testpaths=tests)
```

- **Fast dev loop without torch:** `CLONEHUNTER_EMBEDDER=stub uv run clonehunter scan .`. The stub embedder is deterministic and covers the detection pipeline; reach for `codebert` only when embedding quality is under test.
- **Install dev + optional extras:** `uv sync` then `uv pip install -e ".[dev,faiss]"`.
- **Env vars:** `CLONEHUNTER_EMBEDDER=stub` (force stub); `CLONEHUNTER_SONAR_REPORT=<path>` (required by the `sonarqube` engine).
- **Merge convention:** PRs are **squash-merged to `master`**.

## The frozen Python baseline (Rust-rewrite parity contract)

A Rust rewrite is planned (issue #23). Before it, the Python detector's output was frozen as the parity target: tag **`v1.1.0-python-baseline`** (at `d86f05c`) and the tracked file [benchmark/baseline.json](benchmark/baseline.json). Treat this as a contract:

- [benchmark/run_benchmark.py](benchmark/run_benchmark.py) clones four pinned repos at verified SHAs (click 8.1.8, requests 2.32.3, attrs 24.3.0, rich 13.9.4) and scans each **cold then warm** with a fixed `SCAN_FLAGS` set (codebert/brute/semantic, `--repotype python`, thresholds 0.92/0.90/0.90, window 12/stride 6/min-nonempty 4). It records sorted `finding_scores` (6dp) and `finding_pairs` per repo.
- `--save-baseline` writes `baseline.json` (keys: `timestamp`, `environment`, `results`; `environment` captures clonehunter version + git SHA, lib versions, full config, hardware, resolved `torch_device`). `--compare-baseline` requires **detection counts, cache metrics, and finding pairs to match exactly, scores within 1e-4**; timing is informational (30% tolerance).
- `benchmark/{repos,output,cache}/` are gitignored — `baseline.json` is the only tracked artifact and the sole reproduction contract.

**Any change that alters detection output** (candidate generation, scoring, thresholds, rollup, expansion, parsing) must be deliberate: regenerate the baseline with `run_benchmark.py --save-baseline`, confirm the diff is fully explained by your change, and re-freeze (move the tag + refresh the release asset). Changes that only touch rendering, IO, or performance must leave `baseline.json`'s detection fields byte-identical.
