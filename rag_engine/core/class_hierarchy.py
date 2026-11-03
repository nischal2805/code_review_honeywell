from __future__ import annotations
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rag_engine.models import ClassDef, FieldDef, Parameter, ParseResult

_TYPE_NOISE = re.compile(r'\b(?:const|volatile|mutable|struct|class|enum|typename)\b|[*&]')


def bare_type(type_: str) -> str:
    """``'const Sensor *'`` -> ``'Sensor'``: a type name without qualifiers, pointers or references."""
    return ' '.join(_TYPE_NOISE.sub(' ', type_).split())


def _param_types(sig_key: str) -> List[str]:
    """Parameter types of a sig_key such as ``'Foo::f(int, std::map<int, int>)'``."""
    end = sig_key.rindex(')')
    depth = 0
    start = end
    for i in range(end, -1, -1):          # the last ')' closes the parameter list; find its '('
        if sig_key[i] == ')':
            depth += 1
        elif sig_key[i] == '(':
            depth -= 1
            if depth == 0:
                start = i
                break
    inner = sig_key[start + 1:end]
    types: List[str] = []
    depth = 0
    begin = 0
    for i, ch in enumerate(inner):
        if ch in '(<[{':
            depth += 1
        elif ch in ')>]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            types.append(inner[begin:i].strip())
            begin = i + 1
    if inner.strip():
        types.append(inner[begin:].strip())
    return types


@dataclass
class MethodInfo:
    """A method as declared in its class body (or, if the body never declared it, as defined)."""
    sig_key: str
    class_name: str
    short_name: str
    param_types: List[str]
    parameters: List[Parameter]   # as declared (default values included); empty if unknown
    file_path: str
    line_number: int
    is_virtual: bool      # effective: keyword/override/final, or overrides a virtual method of an ancestor
    is_pure: bool
    is_override: bool     # effective, as is_virtual
    has_body: bool        # defined inside the class body
    is_static: bool = False


def _match_key(info: MethodInfo) -> Optional[Tuple]:
    """What makes a method override another: name and parameter types; any destructor overrides
    an ancestor's virtual destructor; constructors override nothing."""
    if info.short_name == info.class_name.rsplit('::', 1)[-1]:
        return None
    if info.short_name.startswith('~'):
        return ('~',)
    return (info.short_name, tuple(info.param_types))


class ClassHierarchy:
    """Inheritance graph and method tables merged over all parsed files.

    Constructing it also completes the parser's per-function data (the post-pass is idempotent):
    ``FunctionDef.class_name`` becomes None when its qualifier is not a known class (e.g. a
    namespace), and ``is_virtual``/``is_override`` are set for out-of-line definitions of virtual
    methods and for methods overriding a virtual method of an ancestor without saying so.
    """

    def __init__(self, parse_results: Dict[str, ParseResult]) -> None:
        self.classes: Dict[str, ClassDef] = {}
        self.methods: Dict[str, MethodInfo] = {}
        self._bases: Dict[str, List[str]] = {}
        self._derived: Dict[str, List[str]] = {}
        self._class_methods: Dict[str, List[str]] = {}           # class -> sig_keys in declaration order
        self._by_name: Dict[str, Dict[str, List[str]]] = {}      # class -> short name -> sig_keys
        self._by_key: Dict[str, Dict[Tuple, str]] = {}           # class -> override key -> sig_key
        self._roots: Set[str] = set()
        self._merge_classes(parse_results)
        self._link_bases()
        self._index_methods(parse_results)
        self._classify_virtuals()
        self._annotate_functions(parse_results)

    # ── Construction ────────────────────────────────────────────────────────

    def _merge_classes(self, parse_results: Dict[str, ParseResult]) -> None:
        for result in parse_results.values():
            for cdef in result.class_defs:
                have = self.classes.get(cdef.name)
                if have is None:
                    self.classes[cdef.name] = ClassDef(
                        name=cdef.name, file_path=cdef.file_path, line_number=cdef.line_number,
                        end_line_number=cdef.end_line_number, bases=list(cdef.bases),
                        methods=list(cdef.methods), fields=list(cdef.fields), is_struct=cdef.is_struct)
                    continue
                have.bases += [b for b in cdef.bases if b not in have.bases]
                known = {m.sig_key for m in have.methods}
                have.methods += [m for m in cdef.methods if m.sig_key not in known]
                names = {f.name for f in have.fields}
                have.fields += [f for f in cdef.fields if f.name not in names]

    def _link_bases(self) -> None:
        for name, cdef in self.classes.items():
            scope = name.rpartition('::')[0] or None
            bases: List[str] = []
            for written in cdef.bases:
                base = self.resolve_class(written, scope)
                # Bases that are not parsed (std::exception, vendor classes) stay out of the graph.
                if base and base != name and base not in bases:
                    bases.append(base)
            self._bases[name] = bases
        for name, bases in self._bases.items():
            for base in bases:
                self._derived.setdefault(base, []).append(name)
        for kids in self._derived.values():
            kids.sort()

    def _index_methods(self, parse_results: Dict[str, ParseResult]) -> None:
        for name in sorted(self.classes):
            cdef = self.classes[name]
            for decl in cdef.methods:
                self._add_method(MethodInfo(
                    sig_key=decl.sig_key, class_name=name, short_name=decl.name,
                    param_types=_param_types(decl.sig_key), parameters=list(decl.parameters),
                    file_path=cdef.file_path,
                    line_number=decl.line_number, is_virtual=decl.is_virtual, is_pure=decl.is_pure,
                    is_override=decl.is_override, has_body=decl.has_body, is_static=decl.is_static))
        # Definitions the class body did not declare (member templates, parser misses).
        for result in parse_results.values():
            for fn in result.functions:
                if fn.class_name in self.classes and fn.sig_key not in self.methods:
                    self._add_method(MethodInfo(
                        sig_key=fn.sig_key, class_name=fn.class_name, short_name=fn.name.rsplit('::', 1)[-1],
                        param_types=list(fn.param_types), parameters=list(fn.parameters),
                        file_path=fn.file_path, line_number=fn.line_number,
                        is_virtual=fn.is_virtual, is_pure=False, is_override=fn.is_override, has_body=True,
                        is_static=fn.is_static))

    def _add_method(self, info: MethodInfo) -> None:
        self.methods[info.sig_key] = info
        self._class_methods.setdefault(info.class_name, []).append(info.sig_key)
        self._by_name.setdefault(info.class_name, {}).setdefault(info.short_name, []).append(info.sig_key)

    def _classify_virtuals(self) -> None:
        """Propagate virtuality down the hierarchy (bases first)."""
        virtual_keys: Dict[str, Set[Tuple]] = {}
        for cls in self._bases_first():
            ancestors = self.ancestors(cls)
            mine = virtual_keys.setdefault(cls, set())
            for sig in self._class_methods.get(cls, []):
                info = self.methods[sig]
                key = _match_key(info)
                if key is None:
                    continue
                self._by_key.setdefault(cls, {})[key] = sig
                overrides = any(key in virtual_keys.get(a, ()) for a in ancestors)
                info.is_virtual = info.is_virtual or overrides
                info.is_override = info.is_override or overrides
                if info.is_virtual:
                    mine.add(key)
                    if not overrides:
                        self._roots.add(sig)

    def _bases_first(self) -> List[str]:
        order: List[str] = []
        seen: Set[str] = set()

        def visit(cls: str) -> None:
            if cls in seen:
                return
            seen.add(cls)
            for base in self._bases.get(cls, []):
                visit(base)
            order.append(cls)

        for cls in sorted(self.classes):
            visit(cls)
        return order

    def _annotate_functions(self, parse_results: Dict[str, ParseResult]) -> None:
        for result in parse_results.values():
            for fn in result.functions:
                if fn.class_name is not None and fn.class_name not in self.classes:
                    fn.class_name = None
                info = self.methods.get(fn.sig_key) if fn.class_name else None
                if info is not None:
                    fn.is_virtual = fn.is_virtual or info.is_virtual
                    fn.is_override = fn.is_override or info.is_override

    # ── Class relations ─────────────────────────────────────────────────────

    def resolve_class(self, name: str, scope: Optional[str] = None) -> Optional[str]:
        """Qualified name of the class *name* denotes inside *scope* (a class or namespace), or None.

        Looks outwards from *scope* like C++ name lookup does.
        """
        name = re.sub(r'<.*>', '', name).strip()
        parts = scope.split('::') if scope else []
        if name.startswith('::'):
            name, parts = name[2:], []
        if not name:
            return None
        for i in range(len(parts), -1, -1):
            candidate = '::'.join(parts[:i] + [name])
            if candidate in self.classes:
                return candidate
        # shortcut: a unique class with this short name is assumed (covers using-directives and
        # aliases); upgrade with real using-declaration handling if names collide.
        matches = [c for c in self.classes if c.endswith('::' + name)]
        return matches[0] if len(matches) == 1 else None

    def bases(self, c: str) -> List[str]:
        return list(self._bases.get(c, []))

    def derived(self, c: str) -> List[str]:
        """Direct subclasses, sorted by name."""
        return list(self._derived.get(c, []))

    def descendants(self, c: str) -> List[str]:
        """All subclasses, breadth first."""
        found: List[str] = []
        seen = {c}
        queue = deque([c])
        while queue:
            for kid in self._derived.get(queue.popleft(), []):
                if kid not in seen:
                    seen.add(kid)
                    found.append(kid)
                    queue.append(kid)
        return found

    def ancestors(self, c: str) -> List[str]:
        """All base classes, nearest first."""
        found: List[str] = []
        seen = {c}
        queue = deque([c])
        while queue:
            for base in self._bases.get(queue.popleft(), []):
                if base not in seen:
                    seen.add(base)
                    found.append(base)
                    queue.append(base)
        return found

    def inheritance_tree(self, root: str) -> str:
        lines = [root]

        def walk(cls: str, prefix: str, path: Set[str]) -> None:
            kids = [k for k in self.derived(cls) if k not in path]
            for i, kid in enumerate(kids):
                last = i == len(kids) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{kid}")
                walk(kid, prefix + ('    ' if last else '│   '), path | {kid})

        walk(root, '', {root})
        return '\n'.join(lines)

    # ── Members ─────────────────────────────────────────────────────────────

    def find_field(self, c: str, name: str) -> Optional[Tuple[str, FieldDef]]:
        """``(declaring class, field)`` for *name* in *c* or the nearest ancestor declaring it."""
        for cls in [c] + self.ancestors(c):
            cdef = self.classes.get(cls)
            for f in (cdef.fields if cdef else []):
                if f.name == name:
                    return cls, f
        return None

    def field_type(self, c: str, name: str) -> Optional[str]:
        """Type of field *name* in *c* or an ancestor, without ``*``/``&``/``const``; None if no such field."""
        found = self.find_field(c, name)
        if found is None:
            return None
        owner, f = found
        bare = bare_type(f.type_)
        return (self.resolve_class(bare, owner) or bare) or None

    def method(self, sig_key: str) -> Optional[MethodInfo]:
        return self.methods.get(sig_key)

    def find_methods(self, c: str, short_name: str) -> List[str]:
        """Overloads of *short_name* in the nearest class (c, then ancestors) that declares the name."""
        for cls in [c] + self.ancestors(c):
            sigs = self._by_name.get(cls, {}).get(short_name)
            if sigs:
                return list(sigs)
        return []

    def find_method(self, c: str, short_name: str, param_types: Optional[List[str]] = None) -> Optional[str]:
        """sig_key declared on *c* or on the nearest ancestor that declares it (static lookup).

        With *param_types* only an overload with exactly these parameter types matches.
        """
        for cls in [c] + self.ancestors(c):
            for sig in self._by_name.get(cls, {}).get(short_name, []):
                if param_types is None or self.methods[sig].param_types == param_types:
                    return sig
        return None

    def is_virtual_method(self, sig_key: str) -> bool:
        info = self.methods.get(sig_key)
        return info is not None and info.is_virtual

    def overrides_of(self, sig_key: str) -> List[str]:
        """sig_keys in descendants that override the virtual method *sig_key*."""
        info = self.methods.get(sig_key)
        key = _match_key(info) if info is not None and info.is_virtual else None
        if info is None or key is None:
            return []
        return [self._by_key[d][key] for d in self.descendants(info.class_name) if key in self._by_key.get(d, {})]

    def root_virtual_methods(self) -> List[str]:
        """Virtual methods introduced by their class: no ancestor declares the same virtual method.

        Ordered by class name, then declaration order.
        """
        return [sig for cls in sorted(self.classes) for sig in self._class_methods.get(cls, []) if sig in self._roots]
