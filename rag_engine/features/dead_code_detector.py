from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List
from loguru import logger

from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.models import DeadCodeItem, FunctionDef, ParseResult

_CALLBACK_PAT = re.compile(r'\b(callback|handler|hook|listener|on[A-Z]|isr|interrupt|[Ii]nit)\b')
_TEST_PAT = re.compile(r'\b(test|Test|mock|Mock|stub|Stub|fixture|Fixture)\b')


@dataclass
class DeadCodeReport:
    items: List[DeadCodeItem]
    total_functions: int
    dead_count: int
    deactivated_count: int
    false_positive_notes: List[str]
    structural_coverage_impact: str


class DeadCodeDetector:
    def __init__(self, graph_builder: CallGraphBuilder, parse_results: Dict[str, ParseResult],
                 entry_points: List[str] | None = None) -> None:
        self._graph = graph_builder
        self._results = parse_results
        self._entry_points = entry_points or ['main']
        # Keyed by sig_key (e.g. "Calculator::compute(int)") to match the call
        # graph nodes produced by CallGraphBuilder.  Previously keyed by fn.name,
        # which collapsed overloads and caused incorrect reachability results.
        self._all_funcs: Dict[str, FunctionDef] = {
            fn.sig_key: fn for r in parse_results.values() for fn in r.functions
        }

    def analyze(self) -> DeadCodeReport:
        unreachable = self._graph.find_unreachable(self._entry_points)

        # Pattern matches run against the short base name only (strip param list)
        # so that "Calculator::compute()" still matches a pattern on "compute".
        def _base(sig_key: str) -> str:
            return sig_key.split('(')[0]

        callbacks = {n for n in unreachable if _CALLBACK_PAT.search(_base(n))}
        unreachable -= callbacks
        exported = {n for n in unreachable if self._all_funcs.get(n) and self._all_funcs[n].is_virtual}
        unreachable -= exported
        deactivated = {n for n in unreachable if _TEST_PAT.search(_base(n))}
        dead = unreachable - deactivated

        items: List[DeadCodeItem] = []
        for sig_key in sorted(dead):
            fn = self._all_funcs.get(sig_key)
            if fn:
                items.append(DeadCodeItem(
                    name=sig_key, file_path=fn.file_path, line_number=fn.line_number,
                    category='dead_code', do178c_disposition='Remove',
                    coverage_impact=f"~{fn.line_count} lines; {fn.cyclomatic_complexity} decision point(s)"))
        for sig_key in sorted(deactivated):
            fn = self._all_funcs.get(sig_key)
            if fn:
                items.append(DeadCodeItem(
                    name=sig_key, file_path=fn.file_path, line_number=fn.line_number,
                    category='deactivated_code', do178c_disposition='Justify as Deactivated',
                    coverage_impact=f"~{fn.line_count} lines; {fn.cyclomatic_complexity} decision point(s)"))

        fp_notes = [f"{n} excluded — likely callback/exported" for n in sorted(callbacks | exported)]
        dead_count = sum(1 for i in items if i.category == 'dead_code')
        deact_count = sum(1 for i in items if i.category == 'deactivated_code')
        logger.info(f"Dead code: {dead_count} dead, {deact_count} deactivated")
        return DeadCodeReport(
            items=items, total_functions=len(self._all_funcs),
            dead_count=dead_count, deactivated_count=deact_count,
            false_positive_notes=fp_notes,
            structural_coverage_impact=f"{len(items)} items affect ~{len(items)*3}+ uncoverable lines")
