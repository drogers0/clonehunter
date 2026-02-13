from __future__ import annotations

from collections.abc import Callable, Iterable
from multiprocessing import cpu_count, get_context
from typing import Protocol

from clonehunter.core.config import Thresholds
from clonehunter.core.types import CandidateMatch, Embedding, SnippetRef
from clonehunter.model.interfaces import VectorIndex
from clonehunter.similarity.lexical import lexical_similarity


class ProgressFn(Protocol):
    def __call__(
        self,
        iterable: Iterable[tuple[SnippetRef, Embedding]],
        desc: str,
        total: int | None = None,
    ) -> Iterable[tuple[SnippetRef, Embedding]]: ...


IndexFactory = Callable[[], VectorIndex]
WorkerArgs = tuple[
    list[SnippetRef],
    list[Embedding],
    IndexFactory,
    Thresholds,
    int,
    int,
    int,
]
WorkerResult = tuple[int, list[CandidateMatch]]


def run_multiprocessed(
    func: Callable[
        [list[SnippetRef], list[Embedding], IndexFactory, Thresholds, int, ProgressFn | None],
        list[CandidateMatch],
    ],
) -> Callable[..., list[CandidateMatch]]:
    def wrapper(
        snippets: list[SnippetRef],
        embeddings: list[Embedding],
        index_factory: IndexFactory,
        thresholds: Thresholds,
        top_k: int,
        progress: ProgressFn | None = None,
        progress_update: Callable[[int], None] | None = None,
        processes: int | None = None,
    ) -> list[CandidateMatch]:
        if processes is None:
            processes = max(1, cpu_count() - 1)
        if snippets:
            processes = min(processes, len(snippets))
        if processes <= 1 or len(snippets) == 0:
            return func(snippets, embeddings, index_factory, thresholds, top_k, progress)

        chunk_size = (len(snippets) + processes - 1) // processes
        chunks: list[tuple[int, int]] = []
        for start in range(0, len(snippets), chunk_size):
            end = min(start + chunk_size, len(snippets))
            chunks.append((start, end))

        ctx = get_context("spawn")
        args: list[WorkerArgs] = [
            (snippets, embeddings, index_factory, thresholds, top_k, start, end)
            for start, end in chunks
        ]
        results: list[CandidateMatch] = []
        with ctx.Pool(processes=processes) as pool:
            for count, part in pool.imap_unordered(_worker_retrieve, args):
                if progress_update is not None:
                    progress_update(count)
                results.extend(part)
        return results

    return wrapper


def _worker_retrieve(args: WorkerArgs) -> WorkerResult:
    snippets, embeddings, index_factory, thresholds, top_k, start, end = args
    matches = _retrieve_matches(
        snippets,
        embeddings,
        index_factory,
        thresholds,
        top_k,
        progress=None,
        start=start,
        end=end,
    )
    return (end - start), matches


@run_multiprocessed
def retrieve_candidates(
    snippets: list[SnippetRef],
    embeddings: list[Embedding],
    index_factory: IndexFactory,
    thresholds: Thresholds,
    top_k: int,
    progress: ProgressFn | None = None,
    progress_update: Callable[[int], None] | None = None,
) -> list[CandidateMatch]:
    return _retrieve_matches(
        snippets,
        embeddings,
        index_factory,
        thresholds,
        top_k,
        progress=progress,
        start=0,
        end=len(snippets),
    )


def _retrieve_matches(
    snippets: list[SnippetRef],
    embeddings: list[Embedding],
    index_factory: IndexFactory,
    thresholds: Thresholds,
    top_k: int,
    progress: ProgressFn | None,
    start: int,
    end: int,
) -> list[CandidateMatch]:
    id_to_snippet = {snip.snippet_hash: snip for snip in snippets}
    index: VectorIndex = index_factory()
    index.build(embeddings, [snip.snippet_hash for snip in snippets])

    matches: list[CandidateMatch] = []
    iterable: Iterable[tuple[SnippetRef, Embedding]] = zip(
        snippets[start:end], embeddings[start:end], strict=True
    )
    if progress is not None and start == 0 and end == len(snippets):
        iterable = progress(iterable, "Search candidates", total=len(snippets))
    for snip, emb in iterable:
        neighbors = index.query(emb, top_k)
        for neighbor_id, score in neighbors:
            if neighbor_id == snip.snippet_hash:
                continue
            other = id_to_snippet.get(neighbor_id)
            if other is None:
                continue
            lexical = lexical_similarity(snip.text, other.text)
            composite = (1.0 - thresholds.lexical_weight) * score + (
                thresholds.lexical_weight * lexical
            )
            if thresholds.lexical_min_ratio > 0 and lexical < thresholds.lexical_min_ratio:
                continue
            threshold = _threshold_for_kind(other.kind, thresholds)
            if composite >= threshold:
                matches.append(
                    CandidateMatch(
                        snippet_a=snip,
                        snippet_b=other,
                        similarity=composite,
                        evidence=(
                            f"{snip.kind}->{other.kind}"
                            f"|emb={score:.3f}|lex={lexical:.3f}|comp={composite:.3f}"
                        ),
                    )
                )
    return matches


def _threshold_for_kind(kind: str, thresholds: Thresholds) -> float:
    if kind == "FUNC":
        return thresholds.func
    if kind == "WIN":
        return thresholds.win
    return thresholds.exp
