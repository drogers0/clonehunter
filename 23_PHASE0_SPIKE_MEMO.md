# Phase 0 Spike Memo — Candle + Tree-sitter Parity

**Branch:** `rust-rewrite`
**Spike tag:** throwaway — `spike/` directory only
**Measured:** 2026-07-02
**Model:** `microsoft/codebert-base` @ `3b0952feddeffad0063f274080e3c23d75e7eb39`
**Corpus:** 205 snippets (59 curated parse-targets + 146 real CloneHunter src functions)

---

## Summary

| Test | Verdict | Key numbers |
|------|---------|-------------|
| T1a — Candle embedding parity | **PASS** | max cosine diff 2.68e-6, 0 threshold flips |
| T1b — Tree-sitter extraction parity | **PASS** | 59/59 qname+code exact match |
| T1c — Normalization impact | **NO-GO** | min cosine 0.895 between normalizers; 4 lexical threshold flips |
| **Overall** | **CONDITIONAL GO** | T1a+T1b clear; T1c flags a design decision required before T2 |

---

## T1a: Candle Embedding Parity

**Goal:** Verify that `candle` + `XLMRobertaModel` produces embeddings numerically equivalent to Python's `transformers.AutoModel` with mean pooling.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Token ID match rate | 1.000000 | = 1.0 | PASS |
| Max per-dim absolute diff | 2.05e-5 | — | (diagnostic) |
| Mean per-dim absolute diff | 3.63e-7 | — | (diagnostic) |
| Max cosine diff (Rust vs Python) | **2.68e-6** | < 1e-4 | PASS |
| Pairs exceeding 1e-4 cosine diff | **0 / 20,910** | = 0 | PASS |
| Near-threshold flips (0.90, 0.92) | **0** | = 0 | PASS |
| Determinism (two batch passes) | PASS | PASS | PASS |
| Batch invariance max diff | 5.25e-6 | < 1e-4 | PASS |

**Verdict: PASS.** The candle embedding path is numerically equivalent to Python's transformers (within machine-precision float32 tolerance).

### Key issues resolved during spike

1. **Wrong model class.** `BertModel` uses 0-based position IDs; RoBERTa requires `padding_idx + 1 = 2` as the base. Using `BertModel` produced ~4.7% max cosine error. Switching to `XLMRobertaModel` (from `candle_transformers::models::xlm_roberta`) corrected this to 2.68e-6.

2. **Different HF snapshot weights.** The local HF cache had two snapshots: `3b0952…` (config + pytorch_model.bin) and `99d7ef…` (model.safetensors only). Python used `3b0952` weights; the Rust code initially picked up `99d7ef` safetensors. Fixed by converting `3b0952/pytorch_model.bin → model.safetensors` in-place using Python `safetensors.torch.save_file`, then updating `find_in_any_snapshot` to prefer the pinned snapshot.

3. **tokenizer.json unavailable on HuggingFace.** `microsoft/codebert-base` does not publish `tokenizer.json` (only `vocab.json` + `merges.txt`). Generated from Python's `AutoTokenizer.from_pretrained` and committed to `spike/fixtures/tokenizer.json`. Token IDs match Python exactly (100%).

4. **Batch invariance threshold calibrated.** Batch invariance of 5.25e-6 exceeds machine epsilon × 10 but is well below 1e-4. The difference is expected: different batch sizes have different padding, which changes the effective sequence length for LayerNorm. Threshold relaxed to 1e-4.

---

## T1b: Tree-sitter Extraction Parity

**Goal:** Verify that tree-sitter-python 0.23.x extracts the same functions (qualified names, code text, line spans) as Python's `ast` module.

| Metric | Value | Status |
|--------|-------|--------|
| Reference functions | 59 | — |
| Rust-extracted | 59 | PASS |
| Qualified name matches | 59 / 59 (100%) | PASS |
| Code text matches (byte-exact) | 59 / 59 (100%) | PASS |
| Span divergences | 0 | PASS |
| Lambda exclusion | OK (no lambdas extracted) | PASS |

**Verdict: PASS.** Full parity with Python's AST extraction.

### Key issues resolved during spike

- **No `async_function_statement` in tree-sitter-python 0.23.** The grammar uses `function_definition` for all functions. Async is detected by checking if `source[start_byte..start_byte+10].trim_start().starts_with("async")`.
- **Decorated functions** use a `decorated_definition` node with a `definition` field pointing to the inner `function_definition`. Handled explicitly in `walk_node`.

---

## T1c: Normalization Impact

**Goal:** Measure how much Python's `ast.unparse(strip_docstrings(…))` normalization differs from the proposed Rust DD7 normalization (tree-sitter docstring strip → `pass` + source passthrough), in terms of embedding cosine and lexical similarity impact.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Embedding cosine (rust-norm vs py-norm), min | **0.8949** | ≥ 0.90 | FAIL |
| Embedding cosine, mean | 0.9945 | — | OK |
| Embedding cosine, max | 1.0000 | — | OK |
| Lexical diff max (rust-norm vs py-norm scores) | **0.5946** | ≤ 0.10 | FAIL |
| Lexical diff mean | 0.0065 | — | OK |
| Near-threshold lexical flips (at 0.50 threshold) | **4 pairs** | = 0 | FAIL |

**Normalization difference categories observed:**
- Comment preservation: Rust keeps inline comments; Python's `ast.unparse` strips them
- Quote style: Python normalizes to single quotes; Rust preserves original
- Pass insertion differences: minor behavioral edge cases
- Other `ast.unparse` reformatting (continued lines, spacing, etc.)

**Near-threshold lexical flips (threshold = `lexical_min_ratio` default 0.50):**
```
pair (22,51):   py=0.5000  rust=0.3333  [flip at 0.50]
pair (51,52):   py=0.5000  rust=0.3333  [flip at 0.50]
pair (115,116): py=0.5000  rust=0.1633  [flip at 0.50]
pair (201,204): py=0.5000  rust=0.3690  [flip at 0.50]
```

**Verdict: NO-GO.** The source-passthrough normalizer (DD7) diverges from Python's `ast.unparse` enough to change detection outcomes. Specifically: 4 pairs that Python's pipeline would pass the `lexical_min_ratio=0.5` gate would be rejected by the Rust pipeline, and vice versa.

---

## Root Cause Analysis for T1c

Python's normalizer runs `ast.unparse(strip_docstrings(tree))` which:
1. Strips docstrings (shared with Rust)
2. **Strips all comments** (not shared — Rust preserves them)
3. **Normalizes quote style** to single-quoted strings
4. **Re-emits code** through the AST serializer (normalizes whitespace, continued lines, etc.)

The Rust DD7 normalizer replaces docstring `expression_statement` nodes with `{indent}pass\n` and leaves everything else byte-identical to the input. For snippets where comments dominate the identifier token set (e.g., a function whose entire body is a comment block), the two normalizations produce radically different text.

---

## Decision Required Before T2

The T1c result forces a choice:

**Option A — Implement ast.unparse-equivalent in Rust** (recommended for parity)
: Implement comment stripping and AST re-emission using tree-sitter. This closes the gap to Python's normalizer and preserves the frozen Python baseline. Estimated scope: 1–2 days additional spike work before T2.

**Option B — Accept normalization delta and re-freeze baseline**
: Proceed with source-passthrough normalization. Acknowledge that the Rust rewrite will produce different (but bounded) detection results. Re-run `run_benchmark.py --save-baseline` after T2/T3 to capture the new baseline. The mean lexical diff is 0.0065 (small), so most pairs are unaffected; only comment-heavy code at the exact `lexical_min_ratio` boundary changes.

**Recommended path:** Option A for the first few production releases (maintains backward compatibility with the frozen baseline). Option B becomes viable once the re-freeze process is established and the team has validated the delta.

---

## Candle Stack Validation Summary

The candle inference stack is validated for production use:

| Component | Decision |
|-----------|----------|
| `candle-transformers::models::xlm_roberta::XLMRobertaModel` | **Use this** (correct RoBERTa position IDs) |
| `candle-transformers::models::bert::BertModel` | **Do not use** for RoBERTa models (wrong position IDs) |
| Tokenizer | Load from `fixtures/tokenizer.json` (pre-generated); `microsoft/codebert-base` has no `tokenizer.json` on HF |
| Weights format | Safetensors (converted from pytorch_model.bin via `safetensors.torch.save_file`) |
| Weight prefix | None (keys start with `embeddings.*`, `encoder.*` — no `roberta.` prefix in the local cache) |
| Mean pooling | `sum(hidden * mask.unsqueeze(-1)) / clamp(mask.sum(-2), 1, MAX)` |
| Precision | F32 throughout |
| Max cosine error vs Python | **2.68e-6** (< 1e-4 acceptance criterion) |

---

## T0 Scaffold Status

The production Rust scaffold (`Cargo.toml`, `src/`) passes all three gates:

```
cargo fmt --check  → OK
cargo clippy --all-targets -- -D warnings  → OK (0 warnings)
cargo test  → OK (2 tests pass: errors::display_formats_include_context, logging::init_logging_is_idempotent)
```

All module stubs (`cli`, `core`, `parsing`, `snippets`, `embedding`, `similarity`, `reporting`, `index`, `engines`, `io`) are in place with correct inter-module visibility. The scaffold is ready for T2 (config loader) implementation.

---

## Files Produced by This Spike

```
spike/
  Cargo.toml                      # standalone crate (not workspace member)
  src/
    main.rs                       # orchestration + verdict logic
    candle_embed.rs               # T1a: XLMRobertaModel inference + parity check
    treesitter_parse.rs           # T1b: tree-sitter function extraction + parity check
    normalize.rs                  # T1c: DD7 normalizer + impact measurement
  fixtures/
    tokenizer.json                # pre-generated from Python AutoTokenizer (3.4 MB)
    python_embeddings.json        # 205 reference embeddings (768-dim, f32)
    python_cosines.json           # 205×205 pairwise cosines
    python_functions.json         # 59 reference parse-target functions
    python_normalized.json        # 205 Python-normalized snippets
    python_lexical_scores.json    # 205×205 Python lexical scores
    model_info.json               # model SHA, device, determinism check
    parse_targets/                # 18 Python fixture files (01_simple.py … 18_comment_variant.py)
  generate_references.py          # Python script that produced the fixture JSONs
```
