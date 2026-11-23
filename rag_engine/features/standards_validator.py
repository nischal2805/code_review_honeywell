from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from rag_engine.config import Config
from rag_engine.knowledge_base.standards_document import (
    ENFORCED, RULES, VIOLATION_RULES, RuleSource, StandardsDocument, load_standards_document,
)
from rag_engine.knowledge_base.standards_profile import deep_merge, resolve_standards_profile
from rag_engine.retrieval.retriever import best_sentence
from rag_engine.models import FunctionDef, ParseResult, Violation


@dataclass
class ComplianceReport:
    violations: List[Violation]
    compliance_score: float
    total_functions_checked: int
    violations_by_severity: Dict[str, int]
    corrections: Dict[str, str]
    standards_source: str = 'built-in defaults'
    active_rules: Dict[str, Any] | None = None
    # one row per rule for the configured DAL: rule, label, value, origin (document/default), citation, clause
    rule_table: List[Dict[str, Any]] = field(default_factory=list)
    unchecked_clauses: List[Tuple[str, str]] = field(default_factory=list)   # normative, not enforced


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
    'DO178-NAMING': 'Rename to the function naming convention of the project code standard',
    'DO178-CLASS-NAMING': 'Rename the class to the class naming convention of the project code standard',
}


class StandardsValidator:
    def __init__(self, config: Config, parse_results: Dict[str, ParseResult],
                 standards_doc: Optional[StandardsDocument] = None) -> None:
        self._cfg = config
        self._results = parse_results
        self._all_funcs: List[FunctionDef] = [fn for r in parse_results.values() for fn in r.functions]
        self._doc = standards_doc
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
            violations.extend(self._check_class_naming(result))
        self._cite(violations)

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
                                corrections=corrections,
                                standards_source=self._cfg.standards_file or 'built-in defaults',
                                active_rules=self._rules, rule_table=self._rule_table(),
                                unchecked_clauses=list(self._doc.unchecked) if self._doc else [])

    def _load_rules(self) -> Dict[str, Any]:
        profile = copy.deepcopy(self._cfg.standards_profile)
        if self._doc is None and self._cfg.standards_file:
            try:
                self._doc = load_standards_document(self._cfg.standards_file, cache_dir=None)
            except Exception as exc:
                logger.warning(f"Failed to load standards {self._cfg.standards_file}: {exc} — using defaults")
        if self._doc is not None:
            profile = deep_merge(profile, self._doc.profile)
        return resolve_standards_profile(
            profile,
            cyclomatic_complexity_max=self._cfg.cyclomatic_complexity_max,
            function_length_max=self._cfg.function_length_max,
            nesting_depth_max=self._cfg.nesting_depth_max,
            param_count_max=self._cfg.param_count_max,
        )

    @staticmethod
    def _applies(policy: str) -> bool:
        """A construct the standard explicitly allows is not a violation."""
        return str(policy).upper().strip() != 'ALLOWED'

    # ── Traceability to the standards document ──────────────────────────────

    def _origin(self, rule: str) -> Optional[RuleSource]:
        return self._doc.citation_for(rule, self._cfg.dal_level, self._values().get(rule)) if self._doc else None

    def _values(self) -> Dict[str, Any]:
        """rule key -> the value in force for the configured DAL."""
        dal_cfg = self._rules.get('dal_specific_standards', {}).get(self._cfg.dal_level, {})
        universal = self._rules.get('universal_standards', {})
        return {
            **{f'complexity.{k}': v for k, v in dal_cfg.get('complexity', {}).items()},
            **{f'prohibited.{k}': v for k, v in dal_cfg.get('severities', {}).get('prohibited_constructs', {}).items()},
            'prohibited.goto': universal.get('prohibited_constructs', {}).get('goto'),
            'documentation.missing_docstring': universal.get('documentation', {}).get('missing_docstring'),
            **{f'naming.{k}': v.get('convention') for k, v in self._rules.get('naming_conventions', {}).items()
               if isinstance(v, dict)},
        }

    def _cite(self, violations: List[Violation]) -> None:
        """Point every violation at the clause that sets its rule, or say it is a built-in default. When the
        document sets no value, the closest clause found by retrieval is quoted as related, never as the source."""
        cache: Dict[str, Optional[str]] = {}
        for v in violations:
            rule = VIOLATION_RULES.get(v.rule)
            if rule is None and v.rule != 'DO178-UNBOUNDED-LOOP':
                continue
            key = rule or v.rule
            if key not in cache:
                cache[key] = self._reference(rule, 'unbounded loop termination bound' if rule is None else RULES[rule][1])
            v.standard_ref = cache[key]

    def _reference(self, rule: Optional[str], query: str) -> Optional[str]:
        src = self._origin(rule) if rule else None
        if src is not None:
            return f'{src.citation}: "{src.clause}"'
        if self._doc is not None and self._doc.structured and rule and self._in_structured(rule):
            return f'{Path(self._doc.path).name} (structured profile)'
        hits = [h for h in (self._doc.search(query, k=1) if self._doc else []) if h.lexical_rank is not None]
        if hits:
            return f'related, not the rule source — {hits[0].chunk.cite()}: "{best_sentence(hits[0].chunk.text, query)}"'
        return 'built-in default (not set by the standards document)' if self._doc else None

    def _in_structured(self, rule: str) -> bool:
        group, name = rule.split('.', 1)
        p = self._doc.profile if self._doc else {}
        if group == 'naming':
            return name in p.get('naming_conventions', {})
        dal = p.get('dal_specific_standards', {}).get(self._cfg.dal_level, {})
        if group == 'complexity':
            return name in dal.get('complexity', {})
        return name in dal.get('severities', {}).get('prohibited_constructs', {}) or \
            name in p.get('universal_standards', {}).get('prohibited_constructs', {})

    def _rule_table(self) -> List[Dict[str, Any]]:
        values = self._values()
        rows = []
        for rule, (label, _) in RULES.items():
            if values.get(rule) is None:
                continue
            src = self._origin(rule)
            structured = self._doc is not None and self._doc.structured and self._in_structured(rule)
            rows.append({'rule': rule, 'label': label, 'value': values[rule],
                         'origin': 'document' if src or structured else 'default',
                         'enforced': rule in ENFORCED and (rule != 'naming.class' or self._class_naming_on()),
                         'citation': src.citation if src else (Path(self._doc.path).name if structured else ''),
                         'clause': src.clause if src else ''})
        return rows

    def _class_naming_on(self) -> bool:
        """Class names are checked when the standards document sets a class convention."""
        return self._doc is not None and (self._origin('naming.class') is not None
                                          or (self._doc.structured and self._in_structured('naming.class')))

    def _check_class_naming(self, result: ParseResult) -> List[Violation]:
        if not self._class_naming_on():
            return []
        rule = self._rules.get('naming_conventions', {}).get('class', {})
        pattern, convention = rule.get('regex'), rule.get('convention', 'PascalCase')
        out = []
        for cls in result.class_defs:
            short = cls.name.rsplit('::', 1)[-1]
            if pattern and short and not re.match(pattern, short):
                out.append(Violation(rule='DO178-CLASS-NAMING', misra_ref=None, file=cls.file_path, line=cls.line_number,
                                     element=cls.name, message=f"class '{cls.name}' does not match {convention}",
                                     severity='MINOR'))
        return out

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

    @staticmethod
    def _body_line(fn: FunctionDef, pattern: str) -> int:
        """Return the source line of the first matching construct in a function."""
        match = re.search(pattern, fn.body)
        if not match:
            return fn.line_number
        return fn.line_number + fn.body[:match.start()].count('\n')

    def _check_naming(self, fn: FunctionDef) -> List[Violation]:
        naming = self._rules.get('naming_conventions', {})
        function_rule = naming.get('function', {}) if isinstance(naming, dict) else {}
        pattern = function_rule.get('regex', r'^[A-Z][a-zA-Z0-9]*$')
        # fn.name is "Class::method" for methods; the convention applies to the short name.
        short = fn.name.rsplit('::', 1)[-1]
        is_special_member = (short.startswith(('~', 'operator'))
                             or short == (fn.class_name or '').rsplit('::', 1)[-1])
        if not is_special_member and not re.match(pattern, short):
            convention = function_rule.get('convention', 'PascalCase')
            return [Violation(rule='DO178-NAMING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                              function_line=fn.line_number,
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
                                 function_line=fn.line_number,
                                 element=fn.name, severity='MEDIUM',
                                 message=f"CC={fn.cyclomatic_complexity} > limit {max_cc}"))
        if fn.line_count > max_len:
            out.append(Violation(rule='DO178-FUNC-LEN', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 function_line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Length={fn.line_count} > limit {max_len}"))
        if fn.nesting_depth > max_nesting:
            out.append(Violation(rule='DO178-NESTING', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 function_line=fn.line_number,
                                 element=fn.name, severity='MINOR',
                                 message=f"Nesting={fn.nesting_depth} > limit {max_nesting}"))
        if len(fn.parameters) > max_params:
            out.append(Violation(rule='DO178-PARAM-COUNT', misra_ref=None, file=fn.file_path, line=fn.line_number,
                                 function_line=fn.line_number,
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
            if self._applies(policy): out.append(Violation(rule='DO178-GOTO', misra_ref='MISRA-C++:6-6-1', file=fn.file_path,
                                 line=self._body_line(fn, r'\bgoto\b'), function_line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'CRITICAL'),
                                 message=f"'goto' in '{fn.name}' — prohibited by DO-178C"))
        if re.search(r'\b(?:new|delete)\b', body) or re.search(r'\b(?:malloc|calloc|realloc|free)\s*\(', body):
            policy = prohibited.get('dynamic_memory', 'FORBIDDEN')
            if self._applies(policy): out.append(Violation(rule='DO178-DYNAMIC-MEM', misra_ref=None, file=fn.file_path,
                                 line=self._body_line(fn, r'\b(?:new|delete|malloc|calloc|realloc|free)\b'), function_line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'CRITICAL'),
                                 message=f"Dynamic memory in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if re.search(r'\b(?:try|catch)\b', body):
            policy = prohibited.get('exceptions', 'FORBIDDEN')
            if self._applies(policy): out.append(Violation(rule='DO178-EXCEPTION', misra_ref='MISRA-C++:15-0-1', file=fn.file_path,
                                 line=self._body_line(fn, r'\b(?:try|catch)\b'), function_line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"Exception handling in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if fn.sig_key in fn.calls:
            policy = prohibited.get('recursion', 'FORBIDDEN')
            if self._applies(policy): out.append(Violation(rule='DO178-RECURSION', misra_ref=None, file=fn.file_path,
                                 line=self._body_line(fn, rf'\b{re.escape(fn.name)}\s*\('), function_line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"Recursive call in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        if re.search(r'\b(?:dynamic_cast|typeid)\b', body):
            policy = prohibited.get('rtti', 'FORBIDDEN')
            if self._applies(policy): out.append(Violation(rule='DO178-RTTI', misra_ref='MISRA-C++:5-2-2', file=fn.file_path,
                                 line=self._body_line(fn, r'\b(?:dynamic_cast|typeid)\b'), function_line=fn.line_number, element=fn.name,
                                 severity=self._policy_to_violation_severity(policy, 'MAJOR'),
                                 message=f"RTTI in '{fn.name}' — {policy.lower()} for DAL {self._cfg.dal_level}"))
        return out

    def _check_documentation(self, fn: FunctionDef) -> List[Violation]:
        if not fn.docstring:
            doc_rule = self._rules.get('universal_standards', {}).get('documentation', {})
            policy = doc_rule.get('missing_docstring', 'MINOR') if isinstance(doc_rule, dict) else 'MINOR'
            return [Violation(rule='DO178-NO-DOCSTRING', misra_ref=None, file=fn.file_path,
                              line=fn.line_number, function_line=fn.line_number, element=fn.name,
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
            # the innermost function whose span holds the loop; None for a loop outside every function
            enclosing = [fn for fn in result.functions
                         if fn.line_number <= line_no <= (fn.end_line_number or fn.line_number)]
            function_line = max((fn.line_number for fn in enclosing), default=None)
            out.append(Violation(rule='DO178-UNBOUNDED-LOOP', misra_ref=None, file=result.file_path,
                                 line=line_no, function_line=function_line, element='loop', severity='MAJOR',
                                 message='Unbounded loop — exit condition must be proven'))
        return out
