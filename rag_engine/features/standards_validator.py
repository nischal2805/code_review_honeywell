from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List
from loguru import logger

from rag_engine.config import Config
from rag_engine.models import FunctionDef, ParseResult, Violation


@dataclass
class ComplianceReport:
    violations: List[Violation]
    compliance_score: float
    total_functions_checked: int
    violations_by_severity: Dict[str, int]
    corrections: Dict[str, str]


_FIXES = {
    'DO178-GOTO': 'Replace goto with structured control flow (break/return/state machine)',
    'DO178-DYNAMIC-MEM': 'Use static allocation or memory pools; allocate all memory at init',
    'DO178-RECURSION': 'Convert to iterative with explicit stack, or prove bounded depth',
    'DO178-EXCEPTION': 'Use error return codes; document justification if exceptions retained',
    'DO178-CC': 'Decompose into smaller single-responsibility functions',
    'DO178-FUNC-LEN': 'Extract logical sub-operations into named helpers',
    'DO178-NESTING': 'Use early returns (guard clauses) or extract nested logic',
    'DO178-PARAM-COUNT': 'Group related parameters into a struct/configuration object',
    'DO178-UNBOUNDED-LOOP': 'Add explicit loop counter or timeout condition',
    'DO178-NO-DOCSTRING': 'Add comment: purpose, inputs, outputs, DO-178C req trace',
    'DO178-NAMING': 'Rename to camelCase or snake_case per project code standard',
}


class StandardsValidator:
    def __init__(self, config: Config, parse_results: Dict[str, ParseResult]) -> None:
        self._cfg = config
        self._results = parse_results
        self._all_funcs: List[FunctionDef] = [fn for r in parse_results.values() for fn in r.functions]

    def analyze(self) -> ComplianceReport:
        violations: List[Violation] = []
        for fn in self._all_funcs:
            violations.extend(self._check_naming(fn))
            violations.extend(self._check_complexity(fn))
            violations.extend(self._check_prohibited(fn))
            violations.extend(self._check_documentation(fn))
        for result in self._results.values():
            violations.extend(self._check_file(result))

        total = len(self._all_funcs)
        weights = {'CRITICAL': 10, 'MAJOR': 5, 'MEDIUM': 2, 'MINOR': 1}
        penalty = sum(weights.get(v.severity, 1) for v in violations)
        score = max(0.0, 100.0 - (penalty / max(total * 10, 1)) * 100)
        by_sev = {s: sum(1 for v in violations if v.severity == s)
                  for s in ('CRITICAL', 'MAJOR', 'MEDIUM', 'MINOR')}
        corrections = {v.rule: _FIXES[v.rule] for v in violations if v.rule in _FIXES}
        logger.info(f"Standards: {len(violations)} violations, score {score:.1f}%")
        return ComplianceReport(violations=violations, compliance_score=score,
                                total_functions_checked=total, violations_by_severity=by_sev,
                                corrections=corrections)

    def _check_naming(self, fn: FunctionDef) -> List[Violation]:
        if fn.name and fn.name[0].isupper() and not fn.name.startswith('~'):
            return [Violation(rule='DO178-NAMING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                              element=fn.name, message=f"'{fn.name}' starts with uppercase", severity='MINOR')]
        return []

    def _check_complexity(self, fn: FunctionDef) -> List[Violation]:
        cfg = self._cfg
        out = []
        if fn.cyclomatic_complexity > cfg.cyclomatic_complexity_max:
            out.append(Violation(rule='DO178-CC', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MEDIUM',
                                 message=f"CC={fn.cyclomatic_complexity} > limit {cfg.cyclomatic_complexity_max}"))
        if fn.line_count > cfg.function_length_max:
            out.append(Violation(rule='DO178-FUNC-LEN', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Length={fn.line_count} > limit {cfg.function_length_max}"))
        if fn.nesting_depth > cfg.nesting_depth_max:
            out.append(Violation(rule='DO178-NESTING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Nesting={fn.nesting_depth} > limit {cfg.nesting_depth_max}"))
        if len(fn.parameters) > cfg.param_count_max:
            out.append(Violation(rule='DO178-PARAM-COUNT', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Params={len(fn.parameters)} > limit {cfg.param_count_max}"))
        return out

    def _check_prohibited(self, fn: FunctionDef) -> List[Violation]:
        out = []
        body = fn.body
        if re.search(r'\bgoto\b', body):
            out.append(Violation(rule='DO178-GOTO', misra_ref='MISRA-C++:6-6-1', file=fn.file_path,
                                 line=fn.line_number, element=fn.name, severity='CRITICAL',
                                 message=f"'goto' in '{fn.name}' — prohibited by DO-178C"))
        if re.search(r'\b(?:new|delete)\b', body) or re.search(r'\b(?:malloc|calloc|realloc|free)\s*\(', body):
            out.append(Violation(rule='DO178-DYNAMIC-MEM', misra_ref=None, file=fn.file_path,
                                 line=fn.line_number, element=fn.name, severity='CRITICAL',
                                 message=f"Dynamic memory in '{fn.name}' — prohibited post-init"))
        if re.search(r'\b(?:try|catch)\b', body):
            out.append(Violation(rule='DO178-EXCEPTION', misra_ref='MISRA-C++:15-0-1', file=fn.file_path,
                                 line=fn.line_number, element=fn.name, severity='MAJOR',
                                 message=f"Exception handling in '{fn.name}' — requires justification"))
        if fn.name in fn.calls:
            out.append(Violation(rule='DO178-RECURSION', misra_ref=None, file=fn.file_path,
                                 line=fn.line_number, element=fn.name, severity='MAJOR',
                                 message=f"Recursive call in '{fn.name}' — stack depth must be bounded"))
        return out

    def _check_documentation(self, fn: FunctionDef) -> List[Violation]:
        if not fn.docstring:
            return [Violation(rule='DO178-NO-DOCSTRING', misra_ref=None, file=fn.file_path,
                              line=fn.line_number, element=fn.name, severity='MINOR',
                              message=f"'{fn.name}' has no documentation comment")]
        return []

    def _check_file(self, result: ParseResult) -> List[Violation]:
        out = []
        try:
            src = result.raw_source.decode('utf8', errors='replace')
        except Exception:
            return out
        for m in re.finditer(r'\bwhile\s*\(\s*true\s*\)|\bfor\s*\(\s*;;\s*\)', src):
            line_no = src[:m.start()].count('\n') + 1
            out.append(Violation(rule='DO178-UNBOUNDED-LOOP', misra_ref=None, file=result.file_path,
                                 line=line_no, element='loop', severity='MAJOR',
                                 message='Unbounded loop — exit condition must be proven'))
        return out
