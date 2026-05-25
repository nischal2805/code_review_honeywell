from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass
class Rule:
    rule_id: str
    misra_ref: Optional[str]
    category: str
    description: str
    severity: Literal['MINOR', 'MEDIUM', 'MAJOR', 'CRITICAL']
    rationale: str


MISRA_CPP_RULES: List[Rule] = [
    Rule('DO178-GOTO', 'MISRA-C++:6-6-1', 'prohibited',
         'Use of goto statement is prohibited',
         'CRITICAL', 'goto obstructs structured control flow analysis'),
    Rule('DO178-DYNAMIC-MEM', None, 'prohibited',
         'Dynamic memory allocation prohibited post-init',
         'CRITICAL', 'Non-deterministic allocation failure in safety-critical context'),
    Rule('DO178-RECURSION', None, 'restricted',
         'Recursion without bounded stack proof is prohibited',
         'MAJOR', 'Unbounded recursion causes stack overflow'),
    Rule('DO178-EXCEPTION', 'MISRA-C++:15-0-1', 'restricted',
         'Exception handling requires explicit justification',
         'MAJOR', 'Exceptions can mask errors and complicate coverage analysis'),
    Rule('DO178-UNBOUNDED-LOOP', None, 'restricted',
         'Unbounded while(true)/for(;;) loops without verified exit',
         'MAJOR', 'Cannot prove loop termination for structural coverage'),
    Rule('DO178-IMPLICIT-CONV', 'MISRA-C++:5-0-3', 'naming',
         'Implicit type conversions that may lose data',
         'MINOR', 'Silent data loss in avionics context'),
    Rule('DO178-CC', None, 'complexity',
         'Cyclomatic complexity exceeds threshold',
         'MEDIUM', 'High CC correlates with defect density and impedes MC/DC coverage'),
    Rule('DO178-FUNC-LEN', None, 'complexity',
         'Function length exceeds threshold',
         'MINOR', 'Oversized functions hinder review and structural coverage'),
    Rule('DO178-NESTING', None, 'complexity',
         'Nesting depth exceeds threshold',
         'MINOR', 'Deep nesting hinders MC/DC analysis'),
    Rule('DO178-PARAM-COUNT', None, 'complexity',
         'Parameter count exceeds threshold',
         'MINOR', 'Excess parameters increase coupling and test complexity'),
    Rule('DO178-GLOBAL-VAR', None, 'data_coupling',
         'Uncontrolled global variable (data coupling risk)',
         'MEDIUM', 'Untracked global state violates data coupling requirements'),
    Rule('DO178-NO-DOCSTRING', None, 'documentation',
         'Function has no documentation comment',
         'MINOR', 'Missing documentation prevents SQA sign-off'),
    Rule('DO178-NAMING', None, 'naming',
         'Function name does not follow code standard convention',
         'MINOR', 'Inconsistent naming impedes code review'),
]

RULES_BY_ID: Dict[str, Rule] = {r.rule_id: r for r in MISRA_CPP_RULES}


def get_rule(rule_id: str) -> Optional[Rule]:
    return RULES_BY_ID.get(rule_id)


def get_rules_by_category(category: str) -> List[Rule]:
    return [r for r in MISRA_CPP_RULES if r.category == category]
