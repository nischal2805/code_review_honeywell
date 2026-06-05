from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from rag_engine.models import FunctionDef


class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.3) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._threshold = threshold
        self._index: Optional[faiss.IndexFlatIP] = None
        self._functions: List[FunctionDef] = []
        self._dim = 384

    def index_functions(self, functions: List[FunctionDef]) -> None:
        if not functions:
            return
        texts = [self._function_text(fn) for fn in functions]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype='float32')
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._functions = list(functions)
        logger.debug(f"Indexed {len(functions)} functions")

    def find_related(self, query: str, k: int = 10) -> List[FunctionDef]:
        if self._index is None or not self._functions:
            return []
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype='float32')
        k_actual = min(k, len(self._functions))
        scores, indices = self._index.search(q_emb, k_actual)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) >= self._threshold:
                results.append(self._functions[idx])
        # Fallback: if nothing clears threshold, return best k matches to ensure grounding
        if not results:
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append(self._functions[idx])
        return results

    def save(self, path: str) -> None:
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(base / 'index.faiss'))
        with open(base / 'functions.pkl', 'wb') as f:
            pickle.dump(self._functions, f)

    def load(self, path: str) -> None:
        base = Path(path)
        self._index = faiss.read_index(str(base / 'index.faiss'))
        with open(base / 'functions.pkl', 'rb') as f:
            self._functions = pickle.load(f)

    @staticmethod
    def _function_text(fn: FunctionDef) -> str:
        parts = [fn.name, fn.return_type]
        parts += [f"{p.type_} {p.name}" for p in fn.parameters]
        if fn.docstring:
            parts.append(fn.docstring)
        return ' '.join(parts)
