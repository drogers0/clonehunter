# CloneHunter

CloneHunter finds duplicate code across mixed-language repositories. It uses function-aware analysis for Python and windows-based analysis for other code files, with evidence so you can decide what to refactor.

Under the hood, CloneHunter combines snippet generation (function/window/call-expansion), transformer-based code embeddings (CodeBERT), vector similarity search (brute-force or FAISS), and lexical scoring before rolling matches up into findings and HTML/JSON/SARIF reports. This is intentionally not a lightweight grep-style checker: it runs a semantic retrieval pipeline with model inference and indexing to catch harder, non-trivial duplicate patterns.

![CloneHunter HTML report screenshot](https://raw.githubusercontent.com/drogers0/clonehunter/v1.0.1/assets/clonehunter-report-demo.png)

## Quickstart

Requires Python 3.10+.

### Install with uv

```bash
uv python install 3.10
uv pip install git+https://github.com/drogers0/clonehunter
```

### Install from a release tag

```bash
uv pip install git+https://github.com/drogers0/clonehunter@v1.0.1
```

### Install with venv + pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/drogers0/clonehunter
```

### From source (dev)

```bash
git clone https://github.com/drogers0/clonehunter
cd clonehunter
uv python install 3.10
uv sync
uv pip install -e .
```

### Run

```bash
clonehunter scan . --format html --out clonehunter_report.html
```

If the CLI entrypoint is not on your PATH, use:

```bash
uv run clonehunter scan . --format html --out clonehunter_report.html
```

### Notes on dependencies

* `codebert` embedder requires `torch` and `transformers`.
* `faiss` index is optional; install `faiss-cpu` for faster search.
* Use `--embedder stub` for quick local runs without ML dependencies.

---

## Basic Usage

Scan a repository (defaults to HTML and `clonehunter_report.html`; output extension follows `--format`):

```bash
uv run clonehunter scan .
```

Generate a JSON report:

```bash
uv run clonehunter scan . --format json --out report.json
```

Generate an HTML report:

```bash
uv run clonehunter scan . --format html --out report.html
```

Generate a SARIF report:

```bash
uv run clonehunter scan . --format sarif --out report.sarif
```

Diff scan (compare changed files against the full repo):

```bash
clonehunter diff --base HEAD --format json --out diff.json
```

### Language Support

* **Python files**: parsed with AST extraction and analyzed with function/window snippets.
* **Other code files**: analyzed in implicit windows-only mode by file content.
* **Cross-file-type comparisons** are allowed.
* Results can vary by language and repository shape; tune thresholds/windows for best quality.

---

## How Scoring Works

Scores are **composite**: embedding similarity blended with lexical similarity.

```
composite = (1 - lexical_weight) * embedding + lexical_weight * lexical
```

Matches are filtered by `lexical_min_ratio`, and then the composite score is compared against the relevant threshold (`func`, `win`, or `exp`).

---

## Configuration (pyproject.toml)

```toml
[tool.clonehunter]
engine = "semantic"
cluster_findings = false
cluster_min_size = 2

[tool.clonehunter.thresholds]
func = 0.92
win = 0.90
exp = 0.90
min_window_hits = 2
lexical_min_ratio = 0.5
lexical_weight = 0.3

[tool.clonehunter.windows]
window_lines = 12
stride_lines = 6
min_nonempty = 4

[tool.clonehunter.expansion]
enabled = false
depth = 1
max_chars = 4000

[tool.clonehunter.index]
name = "brute"
top_k = 25

[tool.clonehunter.cache]
path = "~/.cache/clonehunter"

[tool.clonehunter.embedder]
name = "codebert"
model_name = "microsoft/codebert-base"
revision = "main"
max_length = 256
batch_size = 16
device = "cpu"
```

By default, CLI scans apply the `monorepo` repotype preset unless overridden with `--repotype` or `--repotype none`.

---

## CLI Options (selected)

```
clonehunter scan [PATHS...] [--format json|html|sarif] [--out FILE]
  --engine semantic|sonarqube
  --embedder codebert|stub
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

clonehunter diff --base REF [--format json|html|sarif] [--out FILE]
```

`--repotype` is additive and can be mixed (for example `--repotype python --repotype react`).
If `--repotype` is omitted, CloneHunter defaults to `monorepo`.
Use `--repotype none` to disable repotype presets.
Merge order is: `pyproject.toml` globs, then `--repotype`, then explicit `--include-globs/--exclude-globs`.
When the same glob appears in both include/exclude sets, the most recent CLI layer wins.

Example mixed-language scan with custom overrides:

```bash
uv run clonehunter scan . \
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
uv run clonehunter scan . --format html --out examples/clonehunter_report.html
uv run clonehunter scan . --format json --out examples/clonehunter_report.json
uv run clonehunter scan . --format sarif --out examples/clonehunter_report.sarif
uv run clonehunter diff --base HEAD --format json --out examples/clonehunter_diff.json
uv run clonehunter diff --base HEAD --format html --out examples/clonehunter_diff_report.html
```

---

## Tuning Tips

* If you see too many false positives, increase `lexical_min_ratio` and/or `lexical_weight`.
* Increase `min_window_hits` to require stronger evidence.
* Exclude tests or generated code via `exclude_globs`.

---

## Limitations

* Semantic similarity is approximate, not guaranteed equivalence.
* Python findings are generally richer due to AST/function context.
* Non-Python findings use windows-only analysis and may require threshold/window tuning.
* Very small functions are harder to compare meaningfully.
* Domain-specific logic may require threshold tuning.

---

## Development

Run the full test suite:

```bash
python -m pytest
```

Run formatting and type checks:

```bash
ruff format .
ruff check .
pyright
```

Install dev dependencies:

```bash
uv pip install -e ".[dev,faiss]"
```
