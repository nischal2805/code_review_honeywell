from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Set
from loguru import logger

from rag_engine.models import FunctionDef, ParseResult, VirtualChange


@dataclass
class VirtualAnalysisResult:
    changes: List[VirtualChange]
    summary: Dict[str, int]
    base_virtual_count: int
    current_virtual_count: int


class VirtualAnalyzer:
    def __init__(self, base_results: Dict[str, ParseResult], current_results: Dict[str, ParseResult]) -> None:
        self._base = base_results
        self._current = current_results

    def analyze(self) -> VirtualAnalysisResult:
        base_virtuals = self._extract_virtuals(self._base)
        current_virtuals = self._extract_virtuals(self._current)
        changes = self._compare(base_virtuals, current_virtuals)
        summary = {
            'added': sum(1 for c in changes if c.change_type == 'added'),
            'removed': sum(1 for c in changes if c.change_type == 'removed'),
            'modified': sum(1 for c in changes if c.change_type == 'modified'),
            'unchanged': sum(1 for c in changes if c.change_type == 'unchanged'),
        }
        logger.info(f"Virtual analysis: {summary}")
        return VirtualAnalysisResult(
            changes=changes, summary=summary,
            base_virtual_count=len(base_virtuals),
            current_virtual_count=len(current_virtuals),
        )

    def _extract_virtuals(self, results: Dict[str, ParseResult]) -> Dict[str, FunctionDef]:
        return {fn.name: fn for r in results.values() for fn in r.functions if fn.is_virtual}

    def _compare(self, base: Dict[str, FunctionDef], current: Dict[str, FunctionDef]) -> List[VirtualChange]:
        changes: List[VirtualChange] = []
        for name in sorted(set(base) | set(current)):
            base_fn = base.get(name)
            curr_fn = current.get(name)
            if base_fn is None and curr_fn is not None:
                changes.append(VirtualChange(
                    change_type='added', function=curr_fn,
                    base_version=None, current_version=curr_fn,
                    do178c_category='Category 2',
                    reverification_scope=f'Full reverification required — new virtual function {name}',
                ))
            elif base_fn is not None and curr_fn is None:
                changes.append(VirtualChange(
                    change_type='removed', function=base_fn,
                    base_version=base_fn, current_version=None,
                    do178c_category='Category 2',
                    reverification_scope=f'Verify all callers updated — {name} removed',
                ))
            elif base_fn is not None and curr_fn is not None:
                change_type, category, scope = self._classify(base_fn, curr_fn)
                changes.append(VirtualChange(
                    change_type=change_type, function=curr_fn,
                    base_version=base_fn, current_version=curr_fn,
                    do178c_category=category, reverification_scope=scope,
                ))
        return changes

    def _classify(self, base_fn: FunctionDef, curr_fn: FunctionDef):
        if self._semantic_sig(base_fn) == self._semantic_sig(curr_fn):
            return 'unchanged', 'Category 1', 'No reverification required'
        if self._structural_sig(base_fn) == self._structural_sig(curr_fn):
            return 'modified', 'Category 1', 'Comment/formatting change only'
        return 'modified', 'Category 2', f'Signature/implementation changed — reverify callers of {curr_fn.name}'

    @staticmethod
    def _semantic_sig(fn: FunctionDef) -> str:
        parts = [fn.return_type, fn.name, ','.join(p.type_ for p in fn.parameters),
                 str(fn.cyclomatic_complexity), str(fn.line_count)]
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()

    @staticmethod
    def _structural_sig(fn: FunctionDef) -> str:
        body = re.sub(r'//[^\n]*', '', fn.body)
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
        body = re.sub(r'\s+', ' ', body).strip()
        return hashlib.sha256(f"{fn.return_type}|{fn.name}|{body}".encode()).hexdigest()
