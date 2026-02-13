from clonehunter.core.config import Thresholds
from clonehunter.core.types import Embedding, FileRef, FunctionRef, SnippetRef
from clonehunter.index.brute_index import BruteIndex
from clonehunter.similarity.candidates import retrieve_candidates


def test_retrieve_candidates_threshold():
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=1, code="pass", code_hash="c"
    )
    snippets = [
        SnippetRef(kind="FUNC", function=fn, start_line=1, end_line=1, text="a", snippet_hash="a"),
        SnippetRef(kind="FUNC", function=fn, start_line=1, end_line=1, text="a", snippet_hash="b"),
    ]
    embeddings = [Embedding(vector=[1.0, 0.0], dim=2), Embedding(vector=[1.0, 0.0], dim=2)]
    matches = retrieve_candidates(
        snippets,
        embeddings,
        BruteIndex,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1),
        2,
        processes=1,
    )
    assert matches
