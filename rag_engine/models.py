from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Literal


@dataclass
class Parameter:
    name: str
    type_: str
    default_value: Optional[str] = None


@dataclass
class FunctionDef:
    name: str
    file_path: str
    line_number: int
    return_type: str
    parameters: List[Parameter]
    is_virtual: bool
    is_inline: bool
    is_static: bool
    body: str
    docstring: Optional[str]
    cyclomatic_complexity: int
    line_count: int
    nesting_depth: int
    called_by: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    data_reads: Set[str] = field(default_factory=set)
    data_writes: Set[str] = field(default_factory=set)
    do178c_requirement_trace: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        return f"{self.file_path}::{self.name}"

    @property
    def signature(self) -> str:
        params = ", ".join(f"{p.type_} {p.name}" for p in self.parameters)
        return f"{self.return_type} {self.name}({params})"


@dataclass
class ParseResult:
    file_path: str
    functions: List[FunctionDef]
    classes: List[str]
    includes: List[str]
    raw_source: bytes


@dataclass
class VirtualChange:
    change_type: Literal['added', 'removed', 'modified', 'unchanged']
    function: FunctionDef
    base_version: Optional[FunctionDef]
    current_version: Optional[FunctionDef]
    do178c_category: Literal['Category 1', 'Category 2']
    reverification_scope: Optional[str] = None


@dataclass
class Violation:
    rule: str
    misra_ref: Optional[str]
    file: str
    line: int
    element: str
    message: str
    severity: Literal['MINOR', 'MEDIUM', 'MAJOR', 'CRITICAL']
    disposition: Optional[str] = None


@dataclass
class DeadCodeItem:
    name: str
    file_path: str
    line_number: int
    category: Literal['dead_code', 'deactivated_code', 'unused_export', 'dead_fragment']
    do178c_disposition: Literal['Remove', 'Justify as Deactivated', 'Investigate', 'Fix Fragment']
    coverage_impact: str


@dataclass
class LRUCoupling:
    lru_name: str
    control_coupling: List[str] = field(default_factory=list)
    data_coupling: List[str] = field(default_factory=list)
    shared_globals: List[str] = field(default_factory=list)
    timing_dependencies: List[str] = field(default_factory=list)
    risk_level: Literal['low', 'medium', 'high'] = 'low'
