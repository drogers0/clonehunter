from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import Protocol

from clonehunter.core.config import CloneHunterConfig
from clonehunter.core.types import Embedding, FunctionRef, ScanResult, ScanStats, SnippetRef
from clonehunter.embedding.cache import EmbeddingCache
from clonehunter.embedding.codebert_embedder import CodeBertConfig, CodeBertEmbedder
from clonehunter.embedding.stub_embedder import StubEmbedder
from clonehunter.index.brute_index import BruteIndex
from clonehunter.index.faiss_index import FaissIndex
from clonehunter.io.fingerprints import embed_cache_key
from clonehunter.io.fs import collect_files
from clonehunter.model.interfaces import VectorIndex
from clonehunter.parsing.python_ast import extract_functions
from clonehunter.parsing.text_units import extract_file_unit
from clonehunter.similarity.candidates import retrieve_candidates
from clonehunter.similarity.clustering import cluster_findings, filter_clusters
from clonehunter.similarity.rollup import rollup_findings
from clonehunter.snippets.expansion import ExpansionParams, expand_calls
from clonehunter.snippets.generators import (
    WindowParams,
    generate_function_snippets,
    generate_window_snippets,
)


class _Embedder(Protocol):
    def embed(self, snippets: list[SnippetRef]) -> list[Embedding]: ...

    @property
    def dim(self) -> int: ...


def _build_brute_index() -> VectorIndex:
    return BruteIndex()


def _build_faiss_index(nlist: int, nprobe: int) -> VectorIndex:
    try:
        return FaissIndex(nlist=nlist, nprobe=nprobe)
    except RuntimeError:
        return BruteIndex()


def _validate_config(config: CloneHunterConfig) -> None:
    if config.windows.window_lines <= 0:
        raise ValueError("window_lines must be > 0")
    if config.windows.stride_lines <= 0:
        raise ValueError("stride_lines must be > 0")
    if not 0.0 <= config.thresholds.lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    if not 0.0 <= config.thresholds.lexical_min_ratio <= 1.0:
        raise ValueError("lexical_min_ratio must be between 0 and 1")


def _embed_snippets(
    snippets: list[SnippetRef],
    embedder: _Embedder,
    cache: EmbeddingCache,
    model_name: str,
    revision: str,
    max_length: int,
    batch_size: int,
    progress: Callable[[Iterable[int], str, int | None], Iterable[int]],
) -> tuple[list[Embedding], int, int]:
    key_map = {
        snip.snippet_hash: embed_cache_key(model_name, revision, max_length, snip.snippet_hash)
        for snip in snippets
    }
    cached = cache.get_many(key_map.values())
    cache_hits = 0
    cache_misses = 0
    to_embed: list[SnippetRef] = []
    for snip in snippets:
        cache_key = key_map[snip.snippet_hash]
        if cache_key in cached:
            cache_hits += 1
        else:
            cache_misses += 1
            to_embed.append(snip)

    if to_embed:
        batches = range(0, len(to_embed), batch_size)
        new_embeddings: list[Embedding] = []
        for start_idx in progress(batches, "Embed snippets", (len(to_embed) - 1) // batch_size + 1):
            batch = to_embed[start_idx : start_idx + batch_size]
            new_embeddings.extend(embedder.embed(batch))
        for snip, emb in zip(to_embed, new_embeddings, strict=True):
            cached[key_map[snip.snippet_hash]] = emb
        cache.set_many(
            {key_map[snip.snippet_hash]: cached[key_map[snip.snippet_hash]] for snip in to_embed}
        )
    embeddings = [cached[key_map[snip.snippet_hash]] for snip in snippets]
    return embeddings, cache_hits, cache_misses


def run_pipeline(paths: list[str], config: CloneHunterConfig) -> ScanResult:
    from typing import TypeVar

    _T = TypeVar("_T")

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    bar = tqdm(total=None, desc="CloneHunter", unit="item") if tqdm else None

    def set_total(total: int) -> None:
        if bar is None:
            return
        if bar.total is None:
            bar.total = total
            bar.refresh()

    def update(n: int) -> None:
        if bar is None:
            return
        bar.update(n)

    def progress(iterable: Iterable[_T], _desc: str, total: int | None = None) -> Iterable[_T]:
        if bar is None:
            return iterable
        for item in iterable:
            update(1)
            yield item

    _validate_config(config)

    timing: dict[str, float] = {}

    start = time.perf_counter()
    files = collect_files(paths, config.include_globs, config.exclude_globs)
    timing["collect_files"] = time.perf_counter() - start

    start = time.perf_counter()
    python_functions: list[FunctionRef] = []
    window_units: list[FunctionRef] = []
    for file in progress(files, "Extract functions", total=len(files)):
        if file.language == "python":
            extracted = extract_functions(file)
            python_functions.extend(extracted)
            window_units.extend(extracted)
            continue
        window_units.extend(extract_file_unit(file))
    timing["extract_functions"] = time.perf_counter() - start

    start = time.perf_counter()
    snippets: list[SnippetRef] = []
    snippets.extend(generate_function_snippets(python_functions))
    snippets.extend(
        generate_window_snippets(
            window_units,
            WindowParams(
                window_lines=config.windows.window_lines,
                stride_lines=config.windows.stride_lines,
                min_nonempty=config.windows.min_nonempty,
            ),
        )
    )
    snippets.extend(
        expand_calls(
            python_functions,
            ExpansionParams(
                enabled=config.expansion.enabled,
                depth=config.expansion.depth,
                max_chars=config.expansion.max_chars,
            ),
        )
    )
    timing["generate_snippets"] = time.perf_counter() - start

    start = time.perf_counter()

    if config.embedder.name == "stub":
        embedder: _Embedder = StubEmbedder()
        model_name = "stub"
        revision = "0"
        max_length = 0
    else:
        embedder = CodeBertEmbedder(
            CodeBertConfig(
                model_name=config.embedder.model_name,
                revision=config.embedder.revision,
                max_length=config.embedder.max_length,
                batch_size=config.embedder.batch_size,
                device=config.embedder.device,
            )
        )
        model_name = config.embedder.model_name
        revision = config.embedder.revision
        max_length = config.embedder.max_length
    cache = EmbeddingCache(config.cache.path)
    if bar is not None:
        key_map = {
            snip.snippet_hash: embed_cache_key(model_name, revision, max_length, snip.snippet_hash)
            for snip in snippets
        }
        cached = cache.get_many(key_map.values())
        to_embed = [snip for snip in snippets if key_map[snip.snippet_hash] not in cached]
        batch_count = (len(to_embed) - 1) // config.embedder.batch_size + 1 if to_embed else 0
        set_total(len(files) + batch_count + len(snippets))
    progress_int: Callable[[Iterable[int], str, int | None], Iterable[int]] = progress
    embeddings, cache_hits, cache_misses = _embed_snippets(
        snippets=snippets,
        embedder=embedder,
        cache=cache,
        model_name=model_name,
        revision=revision,
        max_length=max_length,
        batch_size=config.embedder.batch_size,
        progress=progress_int,
    )
    timing["embed"] = time.perf_counter() - start

    start = time.perf_counter()
    if config.index.name == "faiss":
        index_factory: Callable[[], VectorIndex] = partial(
            _build_faiss_index, config.index.faiss_nlist, config.index.faiss_nprobe
        )
    else:
        index_factory = _build_brute_index
    candidates = retrieve_candidates(
        snippets=snippets,
        embeddings=embeddings,
        index_factory=index_factory,
        thresholds=config.thresholds,
        top_k=config.index.top_k,
        progress=None,
        progress_update=update if bar is not None else None,
    )
    update(len(snippets))
    findings = rollup_findings(candidates, config.thresholds)
    if config.cluster_findings:
        findings = cluster_findings(findings)
        findings = filter_clusters(findings, config.cluster_min_size)
    timing["similarity"] = time.perf_counter() - start

    stats = ScanStats(
        file_count=len(files),
        function_count=len(python_functions),
        snippet_count=len(snippets),
        candidate_count=len(candidates),
        finding_count=len(findings),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    if bar is not None:
        bar.close()
    return ScanResult(
        findings=findings,
        stats=stats,
        config_snapshot={"engine": config.engine},
        timing=timing,
    )
