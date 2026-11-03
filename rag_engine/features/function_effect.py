from __future__ import annotations
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Optional, Set, Tuple

from tree_sitter import Node, Parser

from rag_engine.core.ast_builder import _declared_name, constructed_by_arguments, declared_type_names, get_text
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import _CPP_LANGUAGE
from rag_engine.models import CallSite, FunctionDef, Parameter, ParseResult


@dataclass
class VariableAccess:
    name: str
    kind: Literal['member', 'global', 'parameter', 'unknown']
    type_: str = ''
    owner: Optional[str] = None      # declaring class for members


@dataclass
class EffectCall:
    callee: str                      # sig_key
    callee_class: Optional[str]
    receiver: Optional[str]
    call_type: str
    line: int


@dataclass
class CoupledComponent:
    name: str                        # class name, or file stem for free functions
    coupling: Literal['control', 'data']
    reason: str


@dataclass
class FunctionEffect:
    function: str
    class_name: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    return_type: str
    parameters: List[Parameter]
    description: str
    reads: List[VariableAccess]
    writes: List[VariableAccess]
    calls: List[EffectCall]
    callers: List[str]
    transitive_callers: List[str]
    coupled_components: List[CoupledComponent]
    data_coupled_functions: List[str]
    affected_components: List[str]


# ── File-scope variables ────────────────────────────────────────────────────────

@dataclass
class GlobalVariable:
    name: str
    type_: str
    file_path: str
    line_number: int


# Wrappers a file-scope declaration can sit in: namespaces, extern "C" blocks, #if/#ifdef (header guards).
_SCOPE_WRAPPERS = ('namespace_definition', 'declaration_list', 'linkage_specification',
                   'preproc_if', 'preproc_ifdef', 'preproc_else', 'preproc_elif')
_STORAGE = re.compile(r'\b(?:static|extern|inline)\b')


def _declarations(node: Node) -> Iterator[Node]:
    for child in node.children:
        if child.type == 'declaration':
            yield child
        elif child.type in _SCOPE_WRAPPERS:
            yield from _declarations(child)


def _is_prototype(declarator: Node, source: bytes, type_names: Callable[[], Set[str]], fn_names: Set[str]) -> bool:
    """``int f(int);`` (also ``int *f(int);``) but not the function-pointer variable ``void (*fp)(int);``
    nor ``Uart g_uart(UART1_BASE);``, which tree-sitter reads as a prototype but is an object."""
    top = declarator
    while declarator.type in ('pointer_declarator', 'reference_declarator'):
        inner = declarator.child_by_field_name('declarator')
        if inner is None:
            return False
        declarator = inner
    name = declarator.child_by_field_name('declarator')
    if declarator.type != 'function_declarator' or name is None or name.type == 'parenthesized_declarator':
        return False
    return not (declarator is top and name.type in ('identifier', 'qualified_identifier')
                and get_text(name, source).rsplit('::', 1)[-1] not in fn_names
                and constructed_by_arguments(declarator, source, type_names))


def file_scope_variables(parse_results: Dict[str, ParseResult]) -> Dict[str, GlobalVariable]:
    """Variables declared at translation-unit or namespace level in any parsed file, keyed by name.

    Includes ``static``/``const``/``extern`` ones; excludes function prototypes, typedefs and class members.
    """
    # shortcut: keyed by bare name, so namespaces and same-named file-local statics in different
    # files are conflated; upgrade to qualified names if such collisions matter.
    found: Dict[str, GlobalVariable] = {}
    parser = Parser(_CPP_LANGUAGE)
    fn_names = {fn.name.rsplit('::', 1)[-1] for r in parse_results.values() for fn in r.functions}
    known_types = {c.rsplit('::', 1)[-1] for r in parse_results.values() for c in r.classes}
    for path in sorted(parse_results):
        source = parse_results[path].raw_source
        root = parser.parse(source).root_node
        type_names = lambda: known_types | declared_type_names(root, source)   # noqa: E731 - only a rare shape needs it
        for decl in _declarations(root):
            declarators = [d for d in decl.children_by_field_name('declarator')
                           if not _is_prototype(d, source, type_names, fn_names)]
            if not declarators:
                continue
            head = _STORAGE.sub('', source[decl.start_byte:declarators[0].start_byte].decode('utf-8', 'replace'))
            for d in declarators:
                name = _declared_name(d, source)
                if name and name not in found:
                    core = d.child_by_field_name('declarator') if d.type == 'init_declarator' else d
                    pointer = ' *' if core is not None and core.type == 'pointer_declarator' else ''
                    found[name] = GlobalVariable(name, ' '.join(head.split()) + pointer, path,
                                                 decl.start_point[0] + 1)
    return found


# ── Analyzer ────────────────────────────────────────────────────────────────────

_VarKey = Tuple[str, Optional[str], str]    # ('member', declaring class, name) or ('global', None, name)


def _component(fn: FunctionDef) -> str:
    return fn.class_name or Path(fn.file_path).stem


def _shared_key(v: VariableAccess) -> Optional[_VarKey]:
    return (v.kind, v.owner, v.name) if v.kind in ('member', 'global') else None


def _keys(accesses: List[VariableAccess]) -> Set[_VarKey]:
    return {k for k in map(_shared_key, accesses) if k}


def _describe_key(key: _VarKey) -> str:
    kind, owner, name = key
    return f"{kind} '{name}'" + (f" (declared in {owner})" if owner else '')


def _first_comment_line(doc: Optional[str]) -> str:
    for line in (doc or '').splitlines():
        line = re.sub(r'^\s*(?:/\*+|//+|\*+/?)\s?|\s*\*+/\s*$', '', line).strip()
        if line and not line.startswith('DO-178C-REQ'):
            return line
    return ''


class FunctionEffectAnalyzer:
    def __init__(self, parse_results: Dict[str, ParseResult], graph_builder: CallGraphBuilder) -> None:
        if not graph_builder.built:
            graph_builder.build()
        self._hierarchy = graph_builder.hierarchy
        self._funcs: Dict[str, FunctionDef] = {
            fn.sig_key: fn for r in parse_results.values() for fn in r.functions}
        self._globals = file_scope_variables(parse_results)
        self._fn_names = {fn.name.rsplit('::', 1)[-1] for fn in self._funcs.values()}
        self._sites: Dict[str, List[CallSite]] = {}
        for site in graph_builder.call_sites:
            self._sites.setdefault(site.caller, []).append(site)
        self._access_cache: Dict[str, Tuple[List[VariableAccess], List[VariableAccess]]] = {}
        # Code under ``#if 0`` is never compiled: it can be analysed on request but is nobody's caller,
        # data partner or affected component.
        self._live = {k: fn for k, fn in self._funcs.items() if not fn.never_built}
        self._components = {k: _component(fn) for k, fn in self._funcs.items()}
        self._variable_users: Optional[Tuple[Dict[_VarKey, Set[str]], Dict[_VarKey, Set[str]]]] = None

    def analyze(self, method: str) -> List[FunctionEffect]:
        """Effects of every function matching *method*: short name, qualified name or sig_key."""
        q = method.strip()
        matches = [fn for fn in self._funcs.values()
                   if q in (fn.sig_key, fn.name) or fn.name.endswith('::' + q) or fn.sig_key.endswith('::' + q)]
        return [self._effect(fn) for fn in sorted(matches, key=lambda f: f.sig_key)]

    # ── Per function ────────────────────────────────────────────────────────

    def _effect(self, fn: FunctionDef) -> FunctionEffect:
        reads, writes = self._accesses(fn)
        calls = self._calls(fn)
        callers = sorted(k for k in fn.called_by if k in self._live)
        transitive = sorted(self._transitive_callers(fn))
        data_funcs, data_components = self._data_coupling(fn, reads, writes)
        affected = {self._components[k] for k in transitive + data_funcs if k in self._live}
        affected.add(_component(fn))
        return FunctionEffect(
            function=fn.sig_key, class_name=fn.class_name, file_path=fn.file_path,
            start_line=fn.line_number, end_line=fn.end_line_number or fn.line_number,
            return_type=fn.return_type, parameters=list(fn.parameters),
            description=_first_comment_line(fn.docstring) or self._generated_description(reads, writes, calls),
            reads=reads, writes=writes, calls=calls, callers=callers, transitive_callers=transitive,
            coupled_components=self._control_coupling(fn, calls) + data_components,
            data_coupled_functions=data_funcs, affected_components=sorted(affected))

    def _transitive_callers(self, fn: FunctionDef) -> Set[str]:
        """Every function with a call path to *fn* (``called_by`` mirrors the call graph's predecessors)."""
        seen: Set[str] = set()
        queue = deque([fn.sig_key])
        while queue:
            for caller in self._funcs[queue.popleft()].called_by:
                if caller not in seen and caller in self._live:
                    seen.add(caller)
                    queue.append(caller)
        seen.discard(fn.sig_key)
        return seen

    # ── Variables ───────────────────────────────────────────────────────────

    def _classify(self, fn: FunctionDef, name: str) -> VariableAccess:
        found = self._hierarchy.find_field(fn.class_name, name) if fn.class_name else None
        if found:
            owner, field_def = found
            return VariableAccess(name, 'member', field_def.type_, owner)
        # shortcut: a member and a parameter of the same name are read as the member (the common
        # ``this->x = x`` setter); upgrade with per-scope name resolution.
        for p in fn.parameters:
            if re.sub(r'[\s*&]', '', p.name) == name:
                return VariableAccess(name, 'parameter', p.type_)
        if name in self._globals:
            return VariableAccess(name, 'global', self._globals[name].type_)
        return VariableAccess(name, 'unknown')

    def _accesses(self, fn: FunctionDef) -> Tuple[List[VariableAccess], List[VariableAccess]]:
        """(reads, writes) in order of first appearance in the body."""
        cached = self._access_cache.get(fn.sig_key)
        if cached is None:
            def listing(names: Set[str]) -> List[VariableAccess]:
                def first_use(n: str) -> Tuple[int, str]:
                    m = re.search(r'\b' + re.escape(n) + r'\b', fn.body)
                    return (m.start() if m else len(fn.body), n)
                found = [self._classify(fn, n) for n in sorted(names, key=first_use)]
                # A function name used as a value (callback) is a call edge, not a variable.
                return [v for v in found if not (v.kind == 'unknown' and v.name in self._fn_names)]
            cached = self._access_cache[fn.sig_key] = (listing(fn.data_reads), listing(fn.data_writes))
        return cached

    # ── Calls and control coupling ──────────────────────────────────────────

    def _callee_component(self, sig_key: str) -> Optional[str]:
        info = self._hierarchy.method(sig_key)
        if info is not None:
            return info.class_name
        callee = self._funcs.get(sig_key)
        return _component(callee) if callee else None

    def _effect_call(self, target: str, receiver: Optional[str], call_type: str, line: int) -> EffectCall:
        info = self._hierarchy.method(target)
        return EffectCall(target, info.class_name if info else None, receiver, call_type, line)

    def _calls(self, fn: FunctionDef) -> List[EffectCall]:
        sites = sorted(self._sites.get(fn.sig_key, []), key=lambda s: s.line)
        calls = [self._effect_call(s.target, s.receiver, s.call_type, s.line) for s in sites]
        # A virtual call runs the override of every instantiated class; the graph has those edges
        # but no call sites, so attach them to the line of the virtual call.
        for s in sites:
            if s.call_type == 'virtual':
                calls += [self._effect_call(o, s.receiver, 'virtual', s.line)
                          for o in self._hierarchy.overrides_of(s.target) if o in fn.calls]
        unique = {(c.callee, c.line, c.receiver): c for c in calls}
        return sorted(unique.values(), key=lambda c: c.line)

    def _receiver_text(self, fn: FunctionDef, receiver: str) -> str:
        v = self._classify(fn, receiver)
        return f"{v.kind} '{receiver}'" if v.kind != 'unknown' else f"'{receiver}'"

    def _control_coupling(self, fn: FunctionDef, calls: List[EffectCall]) -> List[CoupledComponent]:
        own = _component(fn)
        reasons: Dict[str, List[str]] = {}
        classes: List[str] = []
        for c in calls:
            comp = self._callee_component(c.callee)
            if comp and comp != own:
                via = f" via {self._receiver_text(fn, c.receiver)}" if c.receiver else ''
                reasons.setdefault(comp, []).append(f"calls {c.callee}{via}")
                if c.callee_class:
                    classes.append(c.callee_class)
        for cls in classes:
            for base in self._hierarchy.ancestors(cls):
                if base != own:
                    reasons.setdefault(base, []).append(f"base class of {cls}")
        return [CoupledComponent(name, 'control', '; '.join(dict.fromkeys(why))) for name, why in reasons.items()]

    # ── Data coupling ───────────────────────────────────────────────────────

    def _users_by_variable(self) -> Tuple[Dict[_VarKey, Set[str]], Dict[_VarKey, Set[str]]]:
        """(readers, writers): variable key -> sig_keys of the live functions that read / write it; built once."""
        if self._variable_users is None:
            readers: Dict[_VarKey, Set[str]] = {}
            writers: Dict[_VarKey, Set[str]] = {}
            for key, other in self._live.items():
                reads, writes = (_keys(a) for a in self._accesses(other))
                for k in reads:
                    readers.setdefault(k, set()).add(key)
                for k in writes:
                    writers.setdefault(k, set()).add(key)
            self._variable_users = (readers, writers)
        return self._variable_users

    def _data_coupling(self, fn: FunctionDef, reads: List[VariableAccess],
                       writes: List[VariableAccess]) -> Tuple[List[str], List[CoupledComponent]]:
        """Functions that read what *fn* writes or write what it reads, and the components that touch
        the same variables.  Members are shared only along the class hierarchy (same class, bases,
        derived): sibling classes have separate objects."""
        r_keys, w_keys = _keys(reads), _keys(writes)
        related = ({fn.class_name} | set(self._hierarchy.ancestors(fn.class_name))
                   | set(self._hierarchy.descendants(fn.class_name))) if fn.class_name else set()
        own = _component(fn)
        readers, writers = self._users_by_variable()

        def visible(key: _VarKey, other: FunctionDef) -> bool:
            return other is not fn and (key[0] == 'global' or other.class_name in related)

        coupled: Set[str] = set()
        for keys, partners in ((w_keys, readers), (r_keys, writers)):
            for key in keys:
                coupled |= {o for o in partners.get(key, ()) if visible(key, self._funcs[o])}
        shared_with: Dict[str, Dict[_VarKey, List[str]]] = {}
        for key in r_keys | w_keys:
            for o in readers.get(key, set()) | writers.get(key, set()):
                other = self._funcs[o]
                if self._components[o] != own and visible(key, other):
                    shared_with.setdefault(self._components[o], {}).setdefault(key, []).append(o)
        components = [
            CoupledComponent(name, 'data', '; '.join(
                f"shares {_describe_key(key)} with {', '.join(sorted(funcs))}"
                for key, funcs in sorted(by_var.items(), key=str)))
            for name, by_var in sorted(shared_with.items())]
        return sorted(coupled), components

    # ── Description ─────────────────────────────────────────────────────────

    @staticmethod
    def _generated_description(reads: List[VariableAccess], writes: List[VariableAccess],
                               calls: List[EffectCall]) -> str:
        written = {v.name for v in writes}
        parts = []
        if writes:
            parts.append('writes ' + ', '.join(v.name for v in writes))
        if any(v.name not in written for v in reads):
            parts.append('reads ' + ', '.join(v.name for v in reads if v.name not in written))
        if calls:
            parts.append('calls ' + ', '.join(dict.fromkeys(c.callee for c in calls)))
        return (parts[0][0].upper() + '; '.join(parts)[1:] + '.') if parts else 'No variable access or calls.'


# ── Text block ──────────────────────────────────────────────────────────────────

def _var_text(v: VariableAccess) -> str:
    detail = ', '.join(x for x in (f"member of {v.owner}" if v.kind == 'member' and v.owner else v.kind, v.type_) if x)
    return f"{v.name} ({detail})"


def _call_text(c: EffectCall) -> str:
    via = f" via {c.receiver}" if c.receiver else ''
    kind = '' if c.call_type == 'direct' else f", {c.call_type}"
    return f"{c.callee}{via} (line {c.line}{kind})" if c.line else f"{c.callee}{via} ({c.call_type})"


def format_effect(effect: FunctionEffect) -> str:
    """The R2 text block: Method / Reads / Writes / Calls / Called by / Coupled components / Affected if changed."""
    def section(title: str, items: List[str]) -> List[str]:
        return [f"{title}:"] + [f"  - {i}" for i in items] if items else [f"{title}: none"]

    params = ', '.join(f"{p.type_} {p.name}".strip() for p in effect.parameters) or 'none'
    lines = [f"Method: {effect.function}",
             f"  Location: {effect.file_path}:{effect.start_line}-{effect.end_line}",
             f"  Returns: {effect.return_type or 'unknown'}    Parameters: {params}",
             f"  Does: {effect.description}"]
    lines += section('Reads', [_var_text(v) for v in effect.reads])
    lines += section('Writes', [_var_text(v) for v in effect.writes])
    lines += section('Calls', [_call_text(c) for c in effect.calls])
    lines += section('Called by', effect.callers)
    indirect = [c for c in effect.transitive_callers if c not in effect.callers]
    if indirect:
        lines.append(f"  Also, indirectly: {', '.join(indirect)}")
    lines += section('Coupled components', [f"{c.name} ({c.coupling}): {c.reason}" for c in effect.coupled_components])
    lines += section('Data-coupled functions', effect.data_coupled_functions)
    lines.append(f"Affected if changed: {', '.join(effect.affected_components) or 'none'}")
    return '\n'.join(lines)


FunctionEffectAnalyzer.format_effect = staticmethod(format_effect)
