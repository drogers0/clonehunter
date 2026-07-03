#!/usr/bin/env python3
"""Generate parity reference data from the Python CloneHunter implementation.

Run from the repo root:
    uv run python spike/generate_references.py

Produces JSON reference fixtures in spike/fixtures/:
    python_embeddings.json   — {text, embedding: [768 floats], token_ids: [int...]}
    python_cosines.json      — pairwise cosine similarity matrix
    python_functions.json    — parsed function metadata per fixture file
    python_normalized.json   — {original, normalized} per function
    python_lexical_scores.json — pairwise Jaccard lexical similarity on normalized text
    model_info.json          — {model, revision} for SHA pinning
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SPIKE_DIR = Path(__file__).parent
FIXTURES_DIR = SPIKE_DIR / "fixtures"
PARSE_TARGETS_DIR = FIXTURES_DIR / "parse_targets"

sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Resolve model revision SHA from HF Hub cache
# ---------------------------------------------------------------------------
print("Resolving model revision SHA...", flush=True)
try:
    from huggingface_hub import hf_hub_download

    _config_local = hf_hub_download(
        "microsoft/codebert-base", "config.json", revision="main"
    )
    # Path: .../snapshots/<SHA>/config.json
    _parts = Path(_config_local).parts
    _snap_idx = next(
        (i for i, p in enumerate(_parts) if p == "snapshots"), None
    )
    MODEL_SHA = _parts[_snap_idx + 1] if _snap_idx is not None else "main"
except Exception as exc:
    print(f"  Warning: could not resolve SHA via hf_hub_download: {exc}")
    MODEL_SHA = "main"

print(f"  Model revision SHA: {MODEL_SHA}", flush=True)

# ---------------------------------------------------------------------------
# Import CloneHunter modules
# ---------------------------------------------------------------------------
from clonehunter.core.types import FileRef, FunctionRef, SnippetKind, SnippetRef
from clonehunter.embedding.codebert_embedder import CodeBertConfig, CodeBertEmbedder
from clonehunter.io.fingerprints import hash_text
from clonehunter.parsing.python_ast import extract_functions
from clonehunter.similarity.lexical import lexical_similarity
from clonehunter.snippets.normalization import normalize_source

# ---------------------------------------------------------------------------
# Initialize embedder (CPU, pinned revision)
# ---------------------------------------------------------------------------
print("Loading CodeBertEmbedder on CPU...", flush=True)
_embed_config = CodeBertConfig(
    model_name="microsoft/codebert-base",
    revision=MODEL_SHA,
    max_length=256,
    batch_size=32,
    device="cpu",
)
embedder = CodeBertEmbedder(_embed_config)
tokenizer = embedder._tokenizer
print("  Embedder ready.", flush=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snippet(text: str, func_ref: FunctionRef) -> SnippetRef:
    return SnippetRef(
        kind="FUNC",
        function=func_ref,
        start_line=func_ref.start_line,
        end_line=func_ref.end_line,
        text=text,
        snippet_hash=hash_text(text),
    )


def _dummy_file_ref(path: str) -> FileRef:
    return FileRef(path=path, content_hash=hash_text(path), language="python")


@dataclass
class FuncInfo:
    file: str
    qualified_name: str
    start_line: int
    end_line: int
    is_async: bool
    code: str


# ---------------------------------------------------------------------------
# T1b: Collect function metadata from parse_targets
# ---------------------------------------------------------------------------
print("\nExtracting functions from parse_targets/...", flush=True)
python_functions: list[dict] = []

for py_file in sorted(PARSE_TARGETS_DIR.glob("*.py")):
    file_ref = _dummy_file_ref(str(py_file))
    funcs = extract_functions(file_ref)
    for f in funcs:
        first_line = f.code.split("\n")[0].lstrip()
        is_async = first_line.startswith("async ")
        python_functions.append(
            {
                "file": py_file.name,
                "qualified_name": f.qualified_name,
                "start_line": f.start_line,
                "end_line": f.end_line,
                "is_async": is_async,
                "code": f.code,
            }
        )
        print(
            f"  {py_file.name}: {f.qualified_name} [{f.start_line}–{f.end_line}]",
            flush=True,
        )

print(f"  Total: {len(python_functions)} functions", flush=True)

# ---------------------------------------------------------------------------
# Collect real CloneHunter functions (Tier 2 — at least 20)
# ---------------------------------------------------------------------------
print("\nSampling real CloneHunter functions...", flush=True)
REAL_SOURCE_FILES = [
    REPO_ROOT / "src/clonehunter/similarity/lexical.py",
    REPO_ROOT / "src/clonehunter/similarity/rollup.py",
    REPO_ROOT / "src/clonehunter/similarity/candidates.py",
    REPO_ROOT / "src/clonehunter/similarity/ranking.py",
    REPO_ROOT / "src/clonehunter/similarity/scoring.py",
    REPO_ROOT / "src/clonehunter/similarity/occurrences.py",
    REPO_ROOT / "src/clonehunter/similarity/clustering.py",
    REPO_ROOT / "src/clonehunter/embedding/codebert_embedder.py",
    REPO_ROOT / "src/clonehunter/snippets/normalization.py",
    REPO_ROOT / "src/clonehunter/snippets/generators.py",
    REPO_ROOT / "src/clonehunter/snippets/expansion.py",
    REPO_ROOT / "src/clonehunter/parsing/python_ast.py",
    REPO_ROOT / "src/clonehunter/io/fs.py",
    REPO_ROOT / "src/clonehunter/io/fingerprints.py",
    REPO_ROOT / "src/clonehunter/io/git.py",
    REPO_ROOT / "src/clonehunter/core/config_loader.py",
    REPO_ROOT / "src/clonehunter/core/pipeline.py",
    REPO_ROOT / "src/clonehunter/cli/commands/scan.py",
    REPO_ROOT / "src/clonehunter/cli/commands/diff.py",
    REPO_ROOT / "src/clonehunter/reporting/compare.py",
    REPO_ROOT / "src/clonehunter/reporting/json_reporter.py",
    REPO_ROOT / "src/clonehunter/reporting/html_reporter.py",
]

real_functions: list[dict] = []
for src_file in REAL_SOURCE_FILES:
    if not src_file.exists():
        continue
    file_ref = _dummy_file_ref(str(src_file))
    funcs = extract_functions(file_ref)
    for f in funcs:
        first_line = f.code.split("\n")[0].lstrip()
        is_async = first_line.startswith("async ")
        real_functions.append(
            {
                "file": src_file.name,
                "qualified_name": f.qualified_name,
                "start_line": f.start_line,
                "end_line": f.end_line,
                "is_async": is_async,
                "code": f.code,
            }
        )

print(f"  {len(real_functions)} real functions from CloneHunter source", flush=True)

# ---------------------------------------------------------------------------
# Build embedding corpus: all parse_target functions + real functions
# ---------------------------------------------------------------------------
all_function_codes: list[str] = []
all_function_refs: list[dict] = []

# From parse targets (skip invalid-syntax file — extractor returns [])
for fn in python_functions:
    if fn["code"]:  # should always be true since extractor returned it
        all_function_codes.append(fn["code"])
        all_function_refs.append(fn)

# From real clonehunter source
for fn in real_functions:
    if fn["code"]:
        all_function_codes.append(fn["code"])
        all_function_refs.append(fn)

print(f"\nTotal embedding corpus: {len(all_function_codes)} functions", flush=True)

# Build SnippetRef objects for the embedder
_dummy_ref = FunctionRef(
    file=_dummy_file_ref("dummy"),
    qualified_name="dummy",
    start_line=1,
    end_line=1,
    code="",
    code_hash="",
)
snippet_refs = [
    SnippetRef(
        kind="FUNC",
        function=_dummy_ref,
        start_line=1,
        end_line=1,
        text=code,
        snippet_hash=hash_text(code),
    )
    for code in all_function_codes
]

# ---------------------------------------------------------------------------
# Embed all snippets (real CodeBertEmbedder, CPU)
# ---------------------------------------------------------------------------
print("Embedding all snippets (CPU)...", flush=True)
embeddings = embedder.embed(snippet_refs)
print(f"  Embedded {len(embeddings)} snippets, dim={embeddings[0].dim}", flush=True)

# Verify 768-dim
assert embeddings[0].dim == 768, f"Expected 768-dim, got {embeddings[0].dim}"

# ---------------------------------------------------------------------------
# Extract token IDs for each text (no padding — single item tokenization)
# ---------------------------------------------------------------------------
print("Extracting token IDs...", flush=True)
token_ids_list: list[list[int]] = []
for code in all_function_codes:
    enc = tokenizer(
        [code],
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    token_ids_list.append(enc["input_ids"][0].tolist())

# ---------------------------------------------------------------------------
# Determinism check: embed all snippets a second time
# ---------------------------------------------------------------------------
print("Determinism check (second embedding pass)...", flush=True)
embeddings2 = embedder.embed(snippet_refs)
determinism_ok = True
for i, (e1, e2) in enumerate(zip(embeddings, embeddings2)):
    if list(e1.vector) != list(e2.vector):
        print(f"  FAIL: embedding {i} differs between runs!", flush=True)
        determinism_ok = False
print(f"  Determinism: {'PASS' if determinism_ok else 'FAIL'}", flush=True)

# ---------------------------------------------------------------------------
# Pairwise cosine similarity matrix
# ---------------------------------------------------------------------------
import numpy as np

print("Computing pairwise cosine similarity matrix...", flush=True)
embedding_matrix = np.array([list(e.vector) for e in embeddings], dtype=np.float32)
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms = np.where(norms > 0, norms, 1.0)
normalized_mat = embedding_matrix / norms
cosine_matrix = (normalized_mat @ normalized_mat.T).tolist()

# ---------------------------------------------------------------------------
# T1c: Normalize all function codes
# ---------------------------------------------------------------------------
print("Normalizing all function codes...", flush=True)
python_normalized: list[dict] = []
for code in all_function_codes:
    normalized = normalize_source(code)
    python_normalized.append({"original": code, "normalized": normalized})

# ---------------------------------------------------------------------------
# Pairwise Jaccard lexical similarity on normalized text
# ---------------------------------------------------------------------------
print("Computing pairwise lexical similarity...", flush=True)
normalized_texts = [n["normalized"] for n in python_normalized]
n = len(normalized_texts)
python_lexical_scores = [[0.0] * n for _ in range(n)]
for i in range(n):
    python_lexical_scores[i][i] = 1.0
    for j in range(i + 1, n):
        score = lexical_similarity(normalized_texts[i], normalized_texts[j])
        python_lexical_scores[i][j] = score
        python_lexical_scores[j][i] = score

# ---------------------------------------------------------------------------
# Build python_embeddings output
# ---------------------------------------------------------------------------
python_embeddings = [
    {
        "text": code,
        "embedding": list(e.vector),
        "token_ids": tids,
        "ref": {"file": ref["file"], "qualified_name": ref["qualified_name"]},
    }
    for code, e, tids, ref in zip(
        all_function_codes, embeddings, token_ids_list, all_function_refs
    )
]

# ---------------------------------------------------------------------------
# Compute SHA-256 hashes of fixture files for the memo
# ---------------------------------------------------------------------------
fixture_hashes: dict[str, str] = {}
for py_file in sorted(PARSE_TARGETS_DIR.glob("*.py")):
    content = py_file.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    fixture_hashes[py_file.name] = sha

# ---------------------------------------------------------------------------
# Write all outputs
# ---------------------------------------------------------------------------
FIXTURES_DIR.mkdir(exist_ok=True)

print("\nWriting output files...", flush=True)

with open(FIXTURES_DIR / "python_embeddings.json", "w") as f:
    json.dump(python_embeddings, f)
print(f"  python_embeddings.json ({len(python_embeddings)} entries)", flush=True)

with open(FIXTURES_DIR / "python_cosines.json", "w") as f:
    json.dump(cosine_matrix, f)
print(f"  python_cosines.json ({n}x{n} matrix)", flush=True)

with open(FIXTURES_DIR / "python_functions.json", "w") as f:
    json.dump(python_functions, f, indent=2)
print(f"  python_functions.json ({len(python_functions)} entries)", flush=True)

with open(FIXTURES_DIR / "python_normalized.json", "w") as f:
    json.dump(python_normalized, f)
print(f"  python_normalized.json ({len(python_normalized)} entries)", flush=True)

with open(FIXTURES_DIR / "python_lexical_scores.json", "w") as f:
    json.dump(python_lexical_scores, f)
print(f"  python_lexical_scores.json ({n}x{n} matrix)", flush=True)

model_info = {
    "model": "microsoft/codebert-base",
    "revision": MODEL_SHA,
    "device": "cpu",
    "max_length": 256,
    "embedding_dim": 768,
    "corpus_size": len(all_function_codes),
    "determinism_pass": determinism_ok,
    "fixture_hashes": fixture_hashes,
}
with open(FIXTURES_DIR / "model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print(f"  model_info.json (revision={MODEL_SHA})", flush=True)

print("\nDone! All reference data written to spike/fixtures/.", flush=True)
print(
    f"Corpus: {len(all_function_codes)} functions "
    f"({len(python_functions)} curated + {len(real_functions)} real CloneHunter code)",
    flush=True,
)
