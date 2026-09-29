from __future__ import annotations
import re
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

    @property
    def param_types(self) -> List[str]:
        """Normalised list of parameter types, stripped of qualifiers and spacing.

        Used to build *sig_key* and to match call-site argument types during
        overload resolution.  Pointer/reference qualifiers are preserved because
        ``void f(int*)`` and ``void f(int)`` are genuinely different overloads.
        """
        return [_normalise_type(p.type_) for p in self.parameters]

    @property
    def sig_key(self) -> str:
        """Canonical, overload-discriminating identifier for this function.

        Format:  ``<qualified_name>(<type1>, <type2>, ...)``

        Examples:
            ``Calculator::compute()``
            ``Calculator::compute(int)``
            ``Utils::log(const std::string &)``
            ``processData()``
            ``processData(int)``

        This key is used as the node identifier in the call graph and in every
        dict that maps function identity to a ``FunctionDef``.  Using it instead
        of the bare ``name`` ensures that overloaded functions are distinct nodes.
        """
        return f"{self.name}({', '.join(self.param_types)})"


def _normalise_type(raw: str) -> str:
    """Return a canonical, whitespace-collapsed representation of a C++ type.

    Removes leading/trailing whitespace and collapses internal runs of spaces
    so that ``"const  std::string &"`` and ``"const std::string&"`` both
    normalise to ``"const std::string &"``.  A single space is inserted before
    ``*`` and ``&`` when they immediately follow a non-space character so the
    representation is predictable regardless of how the parser emitted the type.
    """
    t = raw.strip()
    # Ensure pointer/ref qualifiers are separated by exactly one space.
    t = re.sub(r'\s*([*&])', r' \1', t)
    # Collapse any runs of multiple spaces.
    t = re.sub(r'  +', ' ', t)
    return t


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
