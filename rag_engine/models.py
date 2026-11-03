from __future__ import annotations
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Literal


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
    # Enclosing class for in-class definitions, or the written qualifier for out-of-line
    # ``X::m`` (X may still be a namespace here; later analysis corrects it).
    class_name: Optional[str] = None
    end_line_number: int = 0
    body_start_line: int = 0
    is_override: bool = False
    # Condition text of the nearest enclosing #if/#ifdef/#elif ("0" for ``#if 0``);
    # #ifndef is ignored so include guards do not count.
    preproc_guard: Optional[str] = None
    is_extern_c: bool = False   # defined inside an ``extern "C"`` linkage specification
    local_names: Set[str] = field(default_factory=set)   # variables declared in the body (parameters not included)
    # Further text the compiler runs as part of this function, scanned for calls like the body:
    # (first line, comment-bearing text) of a constructor's initialiser list and of the other
    # ``#if``/``#else`` alternatives of a definition that is compiled either way.
    extra_bodies: List[Tuple[int, str]] = field(default_factory=list)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state.pop('_sig_cache', None)   # rebuilt on first use
        return state

    @property
    def never_built(self) -> bool:
        """Under ``#if 0`` / ``#if false``: the compiler never sees it."""
        return self.preproc_guard in ('0', 'false')

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
        # Computed on first use: it is read in the hottest loops.  The cache is checked against
        # the name and parameter types, so it stays right if either is changed afterwards.
        stamp = (self.name, *[p.type_ for p in self.parameters])
        cached = self.__dict__.get('_sig_cache')
        if cached is None or cached[0] != stamp:
            cached = self.__dict__['_sig_cache'] = (stamp, make_sig_key(self.name, self.parameters))
        return cached[1]


def make_sig_key(name: str, parameters: List[Parameter]) -> str:
    """Build the canonical ``name(type, ...)`` key shared by functions and method declarations."""
    return f"{name}({', '.join(_normalise_type(p.type_) for p in parameters)})"


@lru_cache(maxsize=None)
def _normalise_type(raw: str) -> str:
    """Return a canonical, whitespace-collapsed representation of a C++ type.

    ``"const  std::string&"`` and ``"std::string const &"`` both normalise to
    ``"const std::string &"``: a single space goes before ``*`` and ``&``, ``T const`` becomes
    ``const T`` (also for ``T const *``), and every array bound is dropped (``int [8]`` ->
    ``int []``), so a declaration and its definition agree however they are spelled.
    """
    # shortcut: all array bounds read as one, so overloads differing only in an inner bound of a
    # multi-dimensional array parameter collide; upgrade by keeping the inner bounds.
    t = re.sub(r'\s*([*&])', r' \1', raw.strip())
    t = re.sub(r'\s*\[[^\]]*\]', ' []', t)
    t = re.sub(r'\(\s+', '(', re.sub(r'\s+\)', ')', t))
    t = re.sub(r'  +', ' ', t).strip()
    depth, cut = 0, len(t)
    for i, ch in enumerate(t):     # the base type ends at the first top-level * & [ (
        depth += (ch == '<') - (ch == '>')
        if depth == 0 and ch in '*&[(':
            cut = i
            break
    head, tail = t[:cut].rstrip(), t[cut:]
    if head.endswith(' const') and not head.startswith('const '):
        head = 'const ' + head[:-len(' const')].rstrip()
    return f"{head} {tail}".strip() if tail else head


@dataclass
class FieldDef:
    name: str
    type_: str            # as written, e.g. "TemperatureSensor", "Sensor *", "float"
    line_number: int
    is_static: bool = False


@dataclass
class MethodDecl:
    name: str             # short name, e.g. "process", "~Sensor"
    sig_key: str          # qualified, e.g. "Sensor::process()" (same normalisation as FunctionDef.sig_key)
    line_number: int
    is_virtual: bool      # 'virtual' keyword OR override/final specifier
    is_pure: bool         # declared "= 0"
    is_override: bool
    is_final: bool
    has_body: bool        # defined inline inside the class body
    parameters: List[Parameter] = field(default_factory=list)   # as declared, with default values
    is_static: bool = False


@dataclass
class ClassDef:
    name: str             # qualified with enclosing namespaces/classes, e.g. "hw::Sensor"
    file_path: str
    line_number: int
    end_line_number: int
    bases: List[str]      # base names as written minus access/virtual keywords
    methods: List[MethodDecl] = field(default_factory=list)
    fields: List[FieldDef] = field(default_factory=list)
    is_struct: bool = False
    # (line, initialiser text) of each default member initialiser (``int y = initialY();``): the
    # compiler runs them from every constructor of the class.
    field_inits: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class ParseResult:
    file_path: str
    functions: List[FunctionDef]
    classes: List[str]
    includes: List[str]
    raw_source: bytes
    class_defs: List[ClassDef] = field(default_factory=list)
    # (type as written, enclosing namespace or None) of each object defined at file or namespace scope;
    # pointers, references and function declarations are left out.
    file_scope_objects: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    # (line, initialiser text, enclosing namespace or None) of each file-scope variable with an
    # initialiser (``static int s = seedValue();``): evaluated before main() starts.
    static_inits: List[Tuple[int, str, Optional[str]]] = field(default_factory=list)


@dataclass
class CallSite:
    caller: str                 # sig_key of the calling function
    target: str                 # sig_key of the statically resolved callee; may be a declaration-only method
    file_path: str
    line: int                   # absolute 1-based line in file_path, 0 if unknown
    call_type: str              # 'direct' | 'virtual' | 'callback' | 'constructor' | 'unresolved'
    receiver: Optional[str] = None   # e.g. "sensor" for sensor.read(), "s" for s->process()


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
    name: str                 # function sig_key; the snippet of a fragment; the name of a variable
    file_path: str
    line_number: int
    category: Literal['dead_code', 'deactivated_code', 'unused_export', 'dead_fragment', 'unused_variable']
    do178c_disposition: Literal['Remove', 'Justify as Deactivated', 'Investigate', 'Fix Fragment']
    coverage_impact: str
    finding_id: str = ''      # stable across line shifts, e.g. DC-FUNC-1A2B3C4D
    end_line: int = 0
    symbol_type: Literal['function', 'method', 'fragment', 'variable', 'parameter'] = 'function'
    function_name: Optional[str] = None   # sig_key of the function the finding is (in)
    reason: str = ''
    confidence: Literal['high', 'medium', 'low'] = 'medium'
    evidence: List[str] = field(default_factory=list)
    removal_impact: List[str] = field(default_factory=list)
    review_status: str = 'open'
    reviewer_justification: Optional[str] = None
    reviewer: Optional[str] = None


@dataclass
class LRUCoupling:
    lru_name: str
    control_coupling: List[str] = field(default_factory=list)
    data_coupling: List[str] = field(default_factory=list)
    shared_globals: List[str] = field(default_factory=list)
    timing_dependencies: List[str] = field(default_factory=list)
    risk_level: Literal['low', 'medium', 'high'] = 'low'
