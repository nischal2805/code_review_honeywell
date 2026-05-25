from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from loguru import logger

from rag_engine.core.parser import CodeParser
from rag_engine.knowledge_base.cache_manager import AnalysisCache
from rag_engine.models import FunctionDef, ParseResult


class CodeReader:
    def __init__(self, cache_dir: str = '.rag_cache') -> None:
        self._parser = CodeParser()
        self._cache = AnalysisCache(cache_dir=cache_dir)

    def read_directory(self, dir_path: str) -> Dict[str, ParseResult]:
        results: Dict[str, ParseResult] = {}
        base = Path(dir_path)
        files = list(base.glob('**/*.cpp')) + list(base.glob('**/*.h')) + list(base.glob('**/*.hpp'))
        for f in files:
            cached = self._cache.check_cache(str(f))
            if cached:
                results[str(f)] = cached
            else:
                try:
                    pr = self._parser.parse_file(str(f))
                    self._cache.cache_result(str(f), pr)
                    results[str(f)] = pr
                except Exception as exc:
                    logger.warning(f"Skip {f}: {exc}")
        return results

    def all_functions(self, results: Dict[str, ParseResult]) -> List[FunctionDef]:
        return [fn for r in results.values() for fn in r.functions]
