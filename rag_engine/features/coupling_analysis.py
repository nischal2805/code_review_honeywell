from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set
from loguru import logger

from rag_engine.core.embeddings import SemanticSearch
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.models import FunctionDef, LRUCoupling, ParseResult


@dataclass
class CouplingAnalysisResult:
    lru_impacts: Dict[str, LRUCoupling]
    control_coupling_map: Dict[str, List[str]]
    data_coupling_map: Dict[str, List[str]]
    global_variables: List[str]


class CouplingAnalyzer:
    def __init__(self, parse_results: Dict[str, ParseResult], graph_builder: CallGraphBuilder,
                 search: SemanticSearch, lru_docs: Dict[str, str]) -> None:
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
        related_fns = self._search.find_related(doc_text[:200], k=15)
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
