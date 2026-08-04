from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from loguru import logger

from rag_engine.config import Config
from rag_engine.knowledge_base.standards_profile import (
    deep_merge,
    load_raw_standards_profile,
    resolve_standards_profile,
)
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
        self._rules = self._load_rules()

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

    def _load_rules(self) -> Dict[str, Any]:
        profile = copy.deepcopy(self._cfg.standards_profile)
        if self._cfg.standards_file:
            try:
                profile = deep_merge(profile, load_raw_standards_profile(self._cfg.standards_file))
            except FileNotFoundError:
                logger.warning(f"Standards profile not found: {self._cfg.standards_file} — using defaults")
            except Exception as exc:
                logger.warning(f"Failed to load standards profile {self._cfg.standards_file}: {exc} — using defaults")
        return resolve_standards_profile(
            profile,
            cyclomatic_complexity_max=self._cfg.cyclomatic_complexity_max,
            function_length_max=self._cfg.function_length_max,
            nesting_depth_max=self._cfg.nesting_depth_max,
            param_count_max=self._cfg.param_count_max,
        )

    @staticmethod
    def _policy_to_violation_severity(policy: str, default: str = 'MAJOR') -> str:
        normalized = policy.upper().strip()
        if normalized == 'FORBIDDEN':
            return 'CRITICAL'
        if normalized == 'RESTRICTED':
            return 'MAJOR'
        if normalized == 'ALLOWED':
            return 'MINOR'
        return default

    def _check_naming(self, fn: FunctionDef) -> List[Violation]:
        naming = self._rules.get('naming_conventions', {})
        function_rule = naming.get('function', {}) if isinstance(naming, dict) else {}
        pattern = function_rule.get('regex', r'^[A-Z][a-zA-Z0-9]*$')
        if not re.match(pattern, fn.name):
            convention = function_rule.get('convention', 'PascalCase')
            return [Violation(rule='DO178-NAMING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                              element=fn.name, message=f"'{fn.name}' does not match {convention}", severity='MINOR')]
        return []

    def _check_complexity(self, fn: FunctionDef) -> List[Violation]:
        dal_rules = self._rules.get('dal_specific_standards', {})
        dal_cfg = dal_rules.get(self._cfg.dal_level, {}) if isinstance(dal_rules, dict) else {}
        complexity = dal_cfg.get('complexity', {}) if isinstance(dal_cfg, dict) else {}
        max_cc = int(complexity.get('cyclomatic_complexity_max', self._cfg.cyclomatic_complexity_max))
        max_len = int(complexity.get('function_length_max', self._cfg.function_length_max))
        max_nesting = int(complexity.get('nesting_depth_max', self._cfg.nesting_depth_max))
        max_params = int(complexity.get('parameter_count_max', self._cfg.param_count_max))
        out = []
        if fn.cyclomatic_complexity > max_cc:
            out.append(Violation(rule='DO178-CC', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MEDIUM',
                                 message=f"CC={fn.cyclomatic_complexity} > limit {max_cc}"))
        if fn.line_count > max_len:
            out.append(Violation(rule='DO178-FUNC-LEN', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Length={fn.line_count} > limit {max_len}"))
        if fn.nesting_depth > max_nesting:
            out.append(Violation(rule='DO178-NESTING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Nesting={fn.nesting_depth} > limit {max_nesting}"))
        if len(fn.parameters) > max_params:
            out.append(Violation(rule='DO178-PARAM-COUNT', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Params={len(fn.parameters)} > limit {max_params}"))
        return out

    def _check_prohibited(self, fn: FunctionDef) -> List[Violation]:
        out = []
        body = fn.body
        universal = self._rules.get('universal_standards', {})
        dal_rules = self._rules.get('dal_specific_standards', {})
        dal_cfg = dal_rules.get(self._cfg.dal_level, {}) if isinstance(dal_rules, dict) else {}
        prohibited = dal_cfg.get('severities', {}).get('prohibited_constructs', {}) if isinstance(dal_cfg, dict) else {}
        if re.search(r'\bgoto\b', body):
            policy = universal.get('prohibited_constructs', {}).get('goto', 'FORBIDDEN') if isinstance(universal, dict) else 'FORBIDDEN'
            out.append(Violation(rule='DO178-GOTO', misra_ref='MISRA-C++:6-6-1', file=fn.file_path,
                                 line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'CRITICAL'),
                                 message=f"'goto' in '{fn.name}' — prohibited by DO-178C"))
        if re.search(r'\b(?:new|delete)\b', body) or re.search(r'\b(?:malloc|calloc|realloc|free)\s*\(', body):
            policy = prohibited.get('dynamic_memory', 'FORBIDDEN')
            out.append(Violation(rule='DO178-DYNAMIC-MEM', misra_ref=None, file=fn.file_path,
                                 line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'CRITICAL'),
                                 message=f"Dynamic memory in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if re.search(r'\b(?:try|catch)\b', body):
            policy = prohibited.get('exceptions', 'FORBIDDEN')
            out.append(Violation(rule='DO178-EXCEPTION', misra_ref='MISRA-C++:15-0-1', file=fn.file_path,
                                 line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"Exception handling in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if fn.name in fn.calls:
            policy = prohibited.get('recursion', 'FORBIDDEN')
            out.append(Violation(rule='DO178-RECURSION', misra_ref=None, file=fn.file_path,
                                 line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"Recursive call in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if re.search(r'\b(?:dynamic_cast|typeid)\b', body):
            policy = prohibited.get('rtti', 'FORBIDDEN')
            out.append(Violation(rule='DO178-RTTI', misra_ref='MISRA-C++:5-2-2', file=fn.file_path,
                                 line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"RTTI in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        return out

    def _check_documentation(self, fn: FunctionDef) -> List[Violation]:
        if not fn.docstring:
            doc_rule = self._rules.get('universal_standards', {}).get('documentation', {})
            policy = doc_rule.get('missing_docstring', 'MINOR') if isinstance(doc_rule, dict) else 'MINOR'
            return [Violation(rule='DO178-NO-DOCSTRING', misra_ref=None, file=fn.file_path,
                              line=fn.line_number, element=fn.name,
                              severity=self._policy_to_violation_severity(policy, 'MINOR'),
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
