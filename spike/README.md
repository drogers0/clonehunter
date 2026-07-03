# Phase 0 Parity Spike

**THROWAWAY RESEARCH CODE — DO NOT PORT**

This directory contains a one-shot parity spike (T1) whose sole purpose is to
produce empirical measurements for the Phase 0 GO/NO-GO decision:

- **T1a:** Numeric parity between candle (Rust) and PyTorch (Python) CodeBERT
  embeddings on the fixture corpus.
- **T1b:** tree-sitter vs Python `ast` function extraction: qualified names, line
  spans, and extracted code text.
- **T1c:** Normalization impact: tree-sitter docstring-strip+passthrough vs
  `ast.unparse` on embedding cosines and lexical similarity scores.

Results are recorded in `../23_PHASE0_SPIKE_MEMO.md`.

## Running

```bash
# 1. Generate Python reference data (one-time, CPU)
uv run python spike/generate_references.py

# 2. Run the Rust spike
cd spike && cargo run --release
```

## Files

- `generate_references.py` — Python script that produces fixture JSON files using
  the actual CloneHunter CodeBertEmbedder and parsing/normalization code.
- `fixtures/parse_targets/` — curated Python fixture files for T1b parsing tests.
- `fixtures/*.json` — generated reference data (not committed; run the script).
- `src/` — Rust spike code: T1a (`candle_embed.rs`), T1b (`treesitter_parse.rs`),
  T1c (`normalize.rs`), orchestration (`main.rs`).
- `Cargo.lock` — committed for reproducibility.
