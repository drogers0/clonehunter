from pathlib import Path

import pytest

from clonehunter.core.types import FileRef, FunctionRef, SnippetRef
from clonehunter.embedding.codebert_embedder import CodeBertConfig, CodeBertEmbedder


def test_codebert_embedder_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch  # DO NOT SKIP TEST
    import transformers  # DO NOT SKIP TEST

    torch_version = getattr(torch, "__version__", "")
    transformers_version = getattr(transformers, "__version__", "")
    assert torch_version is not None
    assert transformers_version is not None

    cache = tmp_path / "hf_cache"
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache))

    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=1, code="pass", code_hash="c"
    )
    snippet = SnippetRef(
        kind="FUNC",
        function=fn,
        start_line=1,
        end_line=1,
        text="def f():\n    return 1",
        snippet_hash="s",
    )

    embedder = CodeBertEmbedder(
        CodeBertConfig(
            model_name="hf-internal-testing/tiny-random-roberta",
            revision="main",
            max_length=16,
            batch_size=1,
            device="cpu",
        )
    )
    vectors = embedder.embed([snippet])
    assert len(vectors) == 1
    assert vectors[0].dim > 0
