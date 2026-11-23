"""Hybrid retrieval: BM25 + dense vectors, reciprocal-rank fusion, reranking and near-duplicate removal.

Embeddings are cached per chunk id (a content hash) under ``cache_dir``; re-indexing a changed document
embeds only the chunks whose text changed ("re-embedding")."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

from rag_engine.retrieval.bm25 import BM25, tokenize
from rag_engine.retrieval.chunking import Chunk
from rag_engine.retrieval.models import Embedder, LexicalReranker, Reranker, safe_name

# Domain expansions for the lexical side: a query about "dynamic memory" should also match "malloc".
_EXPANSIONS = {
    'dynamic memory': 'heap malloc calloc realloc free new delete allocation',
    'memory': 'heap allocation malloc new delete',
    'exception': 'try catch throw',
    'recursion': 'recursive',
    'rtti': 'dynamic_cast typeid runtime type',
    'complexity': 'cyclomatic mccabe',
    'function length': 'lines size body',
    'nesting': 'depth nested',
    'parameter': 'argument arguments',
    'naming': 'name names identifier convention case camelcase pascalcase snake',
    'comment': 'documentation documented header doxygen',
    'documentation': 'comment header doxygen',
    'goto': 'jump',
    'loop': 'bounded unbounded while for iteration',
}


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # blended rerank + fusion score in [0, 1] (higher is better)
    fused: float                 # reciprocal-rank-fusion score
    lexical_rank: Optional[int]
    dense_rank: Optional[int]


class EmbeddingCache:
    """chunk id -> vector, one .npz per (collection, model)."""

    def __init__(self, cache_dir: Optional[str], collection: str, model: str) -> None:
        self.path = (Path(cache_dir) / 'rag' / f'{safe_name(collection)}__{safe_name(model)}.npz') if cache_dir else None
        self.vectors: Dict[str, np.ndarray] = {}
        if self.path and self.path.exists():
            try:
                with np.load(self.path) as data:
                    self.vectors = {k: data[k] for k in data.files}
            except Exception as exc:
                logger.warning(f'Embedding cache {self.path} unreadable ({exc}); rebuilding')

    def save(self, keep: Sequence[str]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **{k: self.vectors[k] for k in keep if k in self.vectors})


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Optional[Embedder] = None,
                 reranker: Optional[Reranker] = None, cache_dir: Optional[str] = None,
                 collection: str = 'default', rerank_weight: float = 0.5) -> None:
        self.chunks = list(chunks)
        self.rerank_weight = rerank_weight
        self.embedder = embedder
        self.reranker = reranker or LexicalReranker()
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25(self._tokens)
        self._matrix: Optional[np.ndarray] = None
        self.embedded_now = 0           # chunks embedded by this build (the rest came from the cache)
        if embedder is not None and self.chunks:
            self._embed(cache_dir, collection)

    @property
    def mode(self) -> str:
        dense = 'BM25 + dense' if self._matrix is not None else 'BM25 only'
        return f'{dense}, RRF, {getattr(self.reranker, "name", "lexical")} rerank'

    def _embed(self, cache_dir: Optional[str], collection: str) -> None:
        cache = EmbeddingCache(cache_dir, collection, self.embedder.name)  # type: ignore[union-attr]
        todo = [c for c in self.chunks if c.id not in cache.vectors]
        if todo:
            try:
                vectors = self.embedder.encode([c.text for c in todo])  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning(f'Embedding failed ({exc}); lexical retrieval only')
                return
            for c, v in zip(todo, vectors):
                cache.vectors[c.id] = np.asarray(v, dtype='float32')
            self.embedded_now = len(todo)
        self._matrix = np.stack([cache.vectors[c.id] for c in self.chunks])
        cache.save([c.id for c in self.chunks])
        logger.info(f'RAG index {collection}: {len(self.chunks)} chunks, {len(todo)} newly embedded')

    @staticmethod
    def _expand(query: str) -> str:
        """Both directions: 'dynamic memory' adds 'malloc', and 'malloc' adds 'dynamic memory'."""
        q = query.lower()
        words = set(re.findall(r'[a-z_]+', q))
        extra = [v for k, v in _EXPANSIONS.items() if k in q]
        extra += [k for k, v in _EXPANSIONS.items() if words & set(v.split())]
        return query + ' ' + ' '.join(extra)

    def search(self, query: str, k: int = 5, candidates: int = 25, rrf_k: int = 60,
               min_score: Optional[float] = None) -> List[Hit]:
        if not self.chunks or not query.strip():
            return []
        lexical = self._bm25.scores(tokenize(self._expand(query)))
        lex_order = [i for i in sorted(range(len(lexical)), key=lambda i: -lexical[i]) if lexical[i] > 0][:candidates]
        dense_order: List[int] = []
        if self._matrix is not None:
            try:
                q = self.embedder.encode([query])[0]  # type: ignore[union-attr]
                sims = self._matrix @ q
                dense_order = list(np.argsort(-sims)[:candidates])
            except Exception as exc:
                logger.warning(f'Dense query failed ({exc}); lexical only')
        fused: Dict[int, float] = {}
        lex_rank = {i: r for r, i in enumerate(lex_order)}
        dense_rank = {int(i): r for r, i in enumerate(dense_order)}
        for ranks in (lex_rank, dense_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        pool = sorted(fused, key=lambda i: -fused[i])[:candidates]
        if not pool:
            return []
        # a lexical reranker needs the expanded vocabulary; a cross-encoder reads the question as asked
        rerank_query = self._expand(query) if isinstance(self.reranker, LexicalReranker) else query
        scores = self.reranker.score(rerank_query, [self.chunks[i].text for i in pool])
        # Blend the min-max-normalised reranker and fusion scores: a reranker whose scores are nearly flat on
        # this domain (web-trained cross-encoders often are) cannot overturn what BM25 and dense agree on.
        blended = dict(zip(pool, (self.rerank_weight * r + (1 - self.rerank_weight) * f for r, f in
                                  zip(_minmax(scores), _minmax([fused[i] for i in pool])))))
        ranked = sorted(pool, key=lambda i: -blended[i])
        hits: List[Hit] = []
        for i in ranked:
            s = blended[i]
            if min_score is not None and s < min_score:
                continue
            if any(self._overlap(i, h) > 0.8 for h in hits):      # drop overlapping-chunk duplicates
                continue
            hits.append(Hit(self.chunks[i], float(s), fused[i], lex_rank.get(i), dense_rank.get(i)))
            if len(hits) == k:
                break
        return hits

    def _overlap(self, i: int, hit: Hit) -> float:
        a, b = set(self._tokens[i]), set(tokenize(hit.chunk.text))
        return len(a & b) / max(1, min(len(a), len(b)))


def _minmax(values: Sequence[float]) -> List[float]:
    lo, hi = min(values), max(values)
    return [1.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo) for v in values]


def best_sentence(text: str, query: str) -> str:
    """The sentence of a chunk that matches the query best (for short quotations)."""
    parts = [p for p in re.split(r'(?<=[.!?;])\s+', text) if p.strip()]
    if not parts:
        return text
    scores = LexicalReranker().score(query, parts)
    return parts[max(range(len(parts)), key=lambda i: scores[i])]
