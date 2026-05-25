from __future__ import annotations
from pathlib import Path
from typing import List
from loguru import logger

from rag_engine.core.embeddings import SemanticSearch
from rag_engine.models import FunctionDef


class IndexManager:
    def __init__(self, index_dir: str, model_name: str = 'all-MiniLM-L6-v2') -> None:
        self._dir = Path(index_dir)
        self._search = SemanticSearch(model_name=model_name)
        self._loaded = False

    def build(self, functions: List[FunctionDef]) -> None:
        self._search.index_functions(functions)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._search.save(str(self._dir))
        self._loaded = True
        logger.info(f"Index built with {len(functions)} functions at {self._dir}")

    def load(self) -> bool:
        if not (self._dir / 'index.faiss').exists():
            return False
        self._search.load(str(self._dir))
        self._loaded = True
        return True

    def find_related(self, query: str, k: int = 10) -> List[FunctionDef]:
        if not self._loaded:
            return []
        return self._search.find_related(query, k=k)
