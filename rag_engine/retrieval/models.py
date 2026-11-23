"""Optional neural models: a sentence-transformer embedder and a cross-encoder reranker.

Both load lazily and once per process; when a model (or torch) is unavailable the caller gets None and the
retriever falls back to lexical search / lexical reranking, with a logged warning."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
from loguru import logger

from rag_engine.retrieval.bm25 import tokenize

_EMBEDDERS: Dict[str, object] = {}
_RERANKERS: Dict[str, object] = {}


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...     # (n, dim) float32, L2-normalised


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> List[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return np.asarray(vectors, dtype='float32')


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]


class LexicalReranker:
    """Fallback reranker: query-term coverage, phrase matches and proximity of the matched terms."""
    name = 'lexical'

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        q = list(dict.fromkeys(tokenize(query)))
        bigrams = {(a, b) for a, b in zip(q, q[1:])}
        out = []
        for text in texts:
            toks = tokenize(text)
            positions: Dict[str, List[int]] = {}
            for i, t in enumerate(toks):
                positions.setdefault(t, []).append(i)
            present = [t for t in q if t in positions]
            coverage = len(present) / len(q) if q else 0.0
            phrase = sum(1 for a, b in zip(toks, toks[1:]) if (a, b) in bigrams)
            span = 0.0
            if len(present) > 1:
                firsts = [positions[t][0] for t in present]
                span = len(present) / (max(firsts) - min(firsts) + 1)
            out.append(coverage + 0.3 * min(phrase, 3) + 0.3 * span)
        return out


def load_embedder(model_name: Optional[str]) -> Optional[Embedder]:
    if not model_name:
        return None
    if model_name not in _EMBEDDERS:
        try:
            _EMBEDDERS[model_name] = SentenceTransformerEmbedder(model_name)
            logger.info(f'Embedding model loaded: {model_name}')
        except Exception as exc:   # ImportError (no torch), OSError (model not downloadable offline), ...
            logger.warning(f'Embedding model {model_name} unavailable ({type(exc).__name__}: {exc}); lexical retrieval only')
            _EMBEDDERS[model_name] = None
    return _EMBEDDERS[model_name]  # type: ignore[return-value]


def load_reranker(model_name: Optional[str]) -> Reranker:
    if not model_name or model_name == 'lexical':
        return LexicalReranker()
    if model_name not in _RERANKERS:
        try:
            _RERANKERS[model_name] = CrossEncoderReranker(model_name)
            logger.info(f'Reranker loaded: {model_name}')
        except Exception as exc:
            logger.warning(f'Reranker {model_name} unavailable ({type(exc).__name__}: {exc}); lexical reranking')
            _RERANKERS[model_name] = None
    return _RERANKERS[model_name] or LexicalReranker()  # type: ignore[return-value]


def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
