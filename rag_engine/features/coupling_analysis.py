from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from loguru import logger

from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.models import FunctionDef, LRUCoupling, ParseResult

if TYPE_CHECKING:   # annotation only; importing it would load the embedding models
    from rag_engine.core.embeddings import SemanticSearch

_WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+')   # camelCase / snake_case / ACRONYM words
# Plain English plus the verbs and nouns every C++ codebase is full of: matching on them couples an LRU to
# every read() or update() in the code.
_STOP_WORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was', 'not', 'but', 'all', 'any', 'can',
    'has', 'have', 'had', 'its', 'one', 'out', 'use', 'used', 'using', 'will', 'shall', 'should', 'must',
    'may', 'each', 'every', 'when', 'then', 'than', 'also', 'into', 'over', 'such', 'only', 'which', 'their',
    'there', 'these', 'those', 'where', 'while', 'other', 'between', 'via',
    'get', 'set', 'read', 'write', 'update', 'init', 'initialize', 'process', 'check', 'handle', 'run',
    'start', 'stop', 'value', 'values', 'data', 'new', 'make', 'create', 'compute', 'calculate', 'return',
}


def _words(text: str) -> Set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) >= 3 and w.lower() not in _STOP_WORDS}


@dataclass
class CouplingAnalysisResult:
    lru_impacts: Dict[str, LRUCoupling]
    control_coupling_map: Dict[str, List[str]]
    data_coupling_map: Dict[str, List[str]]
    global_variables: List[str]


class CouplingAnalyzer:
    def __init__(self, parse_results: Dict[str, ParseResult], graph_builder: CallGraphBuilder,
                 search: Optional[SemanticSearch], lru_docs: Dict[str, str]) -> None:
        self._results = parse_results
        self._graph = graph_builder
        self._search = search
        self._lru_docs = lru_docs
        self._all_functions: List[FunctionDef] = [fn for r in parse_results.values() for fn in r.functions]

    def analyze(self) -> CouplingAnalysisResult:
        global_vars = self._find_global_variables()
        control_map = {fn.name: list(fn.calls) for fn in self._all_functions if fn.calls}
        data_map = self._build_data_map(global_vars)
        lru_impacts: Dict[str, LRUCoupling] = {}
        for lru_name, doc_text in self._lru_docs.items():
            lru_impacts[lru_name] = self._analyze_lru(lru_name, doc_text, control_map, data_map)
        logger.info(f"Coupling analysis: {len(lru_impacts)} LRUs analyzed")
        return CouplingAnalysisResult(lru_impacts=lru_impacts, control_coupling_map=control_map,
                                      data_coupling_map=data_map, global_variables=global_vars)

    def _analyze_lru(self, lru_name: str, doc_text: str,
                     control_map: Dict[str, List[str]], data_map: Dict[str, List[str]]) -> LRUCoupling:
        related_fns = self._related_functions(doc_text)
        related_names = {fn.name for fn in related_fns}
        control_coupling = [
            f"{fn.name} ({fn.file_path}:{fn.line_number})"
            for fn in related_fns if fn.calls or fn.is_virtual
        ]
        data_coupling = [
            f"{n} -> {v}" for n in related_names for v in data_map.get(n, [])
        ]
        risk: str
        if len(control_coupling) > 5 or len(data_coupling) > 5:
            risk = 'high'
        elif len(control_coupling) > 2 or len(data_coupling) > 2:
            risk = 'medium'
        else:
            risk = 'low'
        return LRUCoupling(lru_name=lru_name, control_coupling=control_coupling,
                           data_coupling=data_coupling, risk_level=risk)  # type: ignore[arg-type]

    def _related_functions(self, doc_text: str, k: int = 15) -> List[FunctionDef]:
        if self._search is not None:
            return self._search.find_related(doc_text[:200], k=k)
        # shortcut: keyword overlap with function names and docstrings, no stemming or synonyms;
        # use the semantic search for better recall.
        doc_words = _words(doc_text)
        scored = []
        for fn in self._all_functions:
            score = len((_words(fn.name) | _words(fn.docstring or '')) & doc_words)   # distinct document terms hit
            if score:
                scored.append((-score, fn.sig_key, fn))
        return [fn for _, _, fn in sorted(scored, key=lambda s: s[:2])[:k]]

    def _build_data_map(self, global_vars: List[str]) -> Dict[str, List[str]]:
        global_set = set(global_vars)
        return {fn.name: [v for v in (fn.data_reads | fn.data_writes) if v in global_set]
                for fn in self._all_functions if fn.data_reads | fn.data_writes}

    def _find_global_variables(self) -> List[str]:
        pat = re.compile(
            r'^(?:static\s+|extern\s+|const\s+)?(?:int|float|double|bool|char|auto)\s+([A-Za-z_]\w*)\s*(?:=|;)',
            re.MULTILINE)
        found: Set[str] = set()
        for result in self._results.values():
            try:
                src = result.raw_source.decode('utf8', errors='replace')
                found.update(m.group(1) for m in pat.finditer(src))
            except Exception:
                pass
        return sorted(found)
