from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple
import networkx as nx
from loguru import logger

from rag_engine.core.class_hierarchy import ClassHierarchy, MethodInfo, bare_type
from rag_engine.models import CallSite, FunctionDef, Parameter, ParseResult

CallGraph = nx.DiGraph

# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove C/C++ comments and string-literal content from *text*.

    Handles:
    - Single-line comments ``// …``  — replaced with spaces up to the newline.
    - Block comments ``/* … */``     — replaced with spaces (newlines kept).
    - String literals ``"…"``        — interior replaced with spaces so that
      tokens embedded in strings (e.g. ``"process(int)\\n"``) are not mistaken
      for live call sites.  The surrounding quotes are preserved so that
      string-initialised variables are still parseable by other passes.
    - Character literals ``'x'``     — interior replaced with a space.

    Newlines are always preserved so that line-count logic is not disturbed.
    """
    result: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        # ── Block comment ────────────────────────────────────────────────
        if text[i:i+2] == '/*':
            j = text.find('*/', i + 2)
            if j == -1:
                result.append(re.sub(r'[^\n]', ' ', text[i:]))
                break
            result.append(re.sub(r'[^\n]', ' ', text[i:j+2]))
            i = j + 2

        # ── Single-line comment ──────────────────────────────────────────
        elif text[i:i+2] == '//':
            j = text.find('\n', i + 2)
            if j == -1:
                result.append(' ' * (n - i))
                break
            result.append(' ' * (j - i))
            i = j   # '\n' handled in the next iteration

        # ── String literal ───────────────────────────────────────────────
        elif text[i] == '"':
            # Consume the opening quote, then blank the interior up to the
            # closing unescaped quote.
            result.append('"')
            i += 1
            while i < n:
                ch = text[i]
                if ch == '\\':
                    # Escaped character — blank both the backslash and the
                    # next char (which might otherwise look like a token).
                    result.append('  ')
                    i += 2
                elif ch == '"':
                    result.append('"')
                    i += 1
                    break
                elif ch == '\n':
                    # Unterminated string literal (shouldn't happen in valid
                    # C++ but handle gracefully).
                    result.append('\n')
                    i += 1
                    break
                else:
                    result.append(' ')
                    i += 1

        # ── Character literal ────────────────────────────────────────────
        elif text[i] == "'":
            result.append("'")
            i += 1
            while i < n:
                ch = text[i]
                if ch == '\\':
                    result.append('  ')
                    i += 2
                elif ch == "'":
                    result.append("'")
                    i += 1
                    break
                elif ch == '\n':
                    result.append('\n')
                    i += 1
                    break
                else:
                    result.append(' ')
                    i += 1

        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Argument splitting and type inference from literals
# ---------------------------------------------------------------------------

def _split_args(arg_text: str) -> List[str]:
    """Split *arg_text* on top-level commas, respecting nested brackets.

    Returns a list of individual argument expression strings (stripped).
    Returns an empty list for an empty/whitespace-only *arg_text*.

    Examples:
        ""            → []
        "10"          → ["10"]
        "10, 20"      → ["10", "20"]
        "f(1, 2), x"  → ["f(1, 2)", "x"]
    """
    text = arg_text.strip()
    if not text:
        return []
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in ('(', '[', '{', '<'):
            depth += 1
        elif ch in (')', ']', '}', '>'):
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


# Mapping from inferred type category to the set of C++ parameter type strings
# that should match.  Keys are the values returned by _infer_arg_type().
# The matching is done via substring so that "const int &" still matches "int".
_TYPE_COMPAT: Dict[str, List[str]] = {
    'int':          ['int'],
    'float':        ['float'],
    'double':       ['double', 'float'],   # double literal can bind to double or float
    'long double':  ['long double', 'double', 'float'],
    'bool':         ['bool', 'int'],
    'char':         ['char', 'int'],
    'const char *': ['const char', 'char', 'std::string', 'string'],
    'nullptr_t':    ['nullptr', 'void *'],
}


def _infer_arg_type(expr: str) -> Optional[str]:
    """Infer the C++ type category of a single argument expression.

    Only handles literal expressions that can be classified without a full
    type-system.  Returns ``None`` for identifiers, function calls, and other
    expressions whose type cannot be determined syntactically.

    Recognised categories:
        ``'int'``          – integer literals (decimal, hex, octal, binary)
        ``'float'``        – float literals ending with ``f`` or ``F``
        ``'double'``       – floating-point literals without suffix
        ``'long double'``  – floating-point literals ending with ``l`` or ``L``
        ``'bool'``         – ``true`` / ``false``
        ``'char'``         – character literals ``'x'``
        ``'const char *'`` – string literals ``"…"``
        ``'nullptr_t'``    – ``nullptr`` / ``NULL``
    """
    s = expr.strip()
    if not s:
        return None
    # Integer literal: decimal, hex, octal, binary; optional sign; optional suffix
    if re.match(r'^[+-]?(0[xXbBoO][\da-fA-F_]+|\d[\d_]*)[uUlL]*$', s):
        return 'int'
    # Float literal (f/F suffix required)
    if re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[fF]$', s):
        return 'float'
    # Long double literal (l/L suffix)
    if re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[lL]$', s):
        return 'long double'
    # Double literal: decimal point or exponent, no suffix
    if re.match(r'^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$', s):
        return 'double'
    # String literal: optional encoding prefix, then "…"
    if re.match(r'^(L|u8|u|U)?"', s):
        return 'const char *'
    # Char literal: optional encoding prefix, then '…'
    if re.match(r"^(L|u|U)?'", s):
        return 'char'
    if s in ('true', 'false'):
        return 'bool'
    if s in ('nullptr', 'NULL'):
        return 'nullptr_t'
    return None


def _types_match(inferred: str, param_type: str) -> bool:
    """Return True if *inferred* type category is compatible with *param_type*.

    Uses substring matching so that ``"const int &"`` matches ``"int"``.
    """
    compat = _TYPE_COMPAT.get(inferred, [inferred])
    pt = param_type.lower()
    return any(c.lower() in pt for c in compat)


# ---------------------------------------------------------------------------
# Signatures and per-call-site matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Sig:
    """The part of a callee's parameter list that call-site matching looks at."""
    types: Tuple[str, ...]    # fixed parameter types ("..." is not one of them)
    required: int             # parameters without a default argument
    variadic: bool = False

    @classmethod
    def of(cls, param_types: List[str], *declarations: List[Parameter]) -> _Sig:
        """*declarations* are the parameter lists of the definition and/or the in-class declaration
        of one function; a default argument given by any of them counts."""
        fixed = [i for i, t in enumerate(param_types) if t != '...']
        required = sum(all(i >= len(ps) or ps[i].default_value is None for ps in declarations) for i in fixed)
        return cls(tuple(param_types[i] for i in fixed), required, len(fixed) != len(param_types))

    def accepts(self, arity: int) -> bool:
        return self.required <= arity and (self.variadic or arity <= len(self.types))


def _call_args(body: str, open_idx: int) -> Optional[List[str]]:
    """Argument expressions of the call whose ``(`` is at *open_idx*; None if it is never closed."""
    depth = 0
    for i in range(open_idx, len(body)):
        ch = body[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return _split_args(body[open_idx + 1:i])
    return None


def _iter_call_sites(body: str, call_token: str) -> Iterator[Tuple[int, List[str]]]:
    """Yield ``(offset, argument expressions)`` for every call of *call_token* in *body*.

    ``call_token`` is a bare name (``"process"``) or a qualified name (``"Utils::log"``).  It must
    not be the tail of a longer expression (``obj.f(``, ``p->f(``, ``ns::f(``, ``xf(``), so ``f``
    matches plain calls only.
    """
    for m in re.finditer(r'(?<![\w.:])(?<!->)' + re.escape(call_token) + r'\s*\(', body):
        args = _call_args(body, m.end() - 1)
        if args is not None:
            yield m.start(), args


def _site_matches(candidates: Set[str], arg_exprs: List[str], sigs: Dict[str, _Sig]) -> Set[str]:
    """The *candidates* a single call with arguments *arg_exprs* can resolve to.

    1. **Arity**: keep candidates that accept this many arguments (default arguments and a
       trailing ``...`` widen the range).  If none does, keep all candidates (conservative).
    2. **Types**: infer C++ type categories from literal arguments, score each remaining
       candidate by how many positional parameter types are compatible and keep the best.
       Skipped when no argument has an inferable type or when it cannot discriminate.
    """
    arity = len(arg_exprs)
    arity_matched = {c for c in candidates if c in sigs and sigs[c].accepts(arity)}
    pool = arity_matched if arity_matched else set(candidates)

    inferred = [_infer_arg_type(e) for e in arg_exprs]
    if any(t is not None for t in inferred):
        scores: Dict[str, int] = {}
        for cand in pool:
            sig = sigs.get(cand)
            if sig is None:
                continue
            score = 0
            for pos, inf_type in enumerate(inferred):
                if inf_type is None or pos >= len(sig.types):
                    continue   # unknown argument or extra variadic argument — neutral
                score += 1 if _types_match(inf_type, sig.types[pos]) else -1
            scores[cand] = score
        if scores:
            best = max(scores.values())
            typed_filtered = {c for c, s in scores.items() if s == best}
            # If every candidate scored equally there is no discrimination power: keep the pool.
            if len(typed_filtered) < len(pool):
                pool = typed_filtered
    return pool


# ---------------------------------------------------------------------------
# Resolved calls of one function
# ---------------------------------------------------------------------------

# Strongest evidence first: the call_type kept on an edge reached in several ways.
_CALL_RANK = {'direct': 0, 'virtual': 1, 'constructor': 2, 'callback': 3, 'unresolved': 4}


class _Calls:
    """Edges and call sites resolved for one caller."""

    def __init__(self, fn: FunctionDef, body: str, nodes: Dict[str, FunctionDef]) -> None:
        self.fn = fn
        self.caller = fn.sig_key   # a property that re-normalises types on every access
        self.body = body      # comment-stripped; offsets are relative to it
        self._nodes = nodes
        self.edges: Dict[str, Tuple[str, Set[int]]] = {}
        self.sites: List[CallSite] = []
        self._seen: Set[Tuple[str, int, str, Optional[str]]] = set()

    def line_of(self, offset: Optional[int]) -> int:
        """Absolute line of a body offset, 0 when it cannot be determined."""
        if offset is None or not self.fn.body_start_line:
            return 0
        return self.fn.body_start_line + self.body.count('\n', 0, offset)

    def edge(self, target: str, call_type: str, offset: Optional[int]) -> None:
        """Graph edge only (a target without a function body has no node and gets none)."""
        self._put_edge(target, call_type, self.line_of(offset))

    def _put_edge(self, target: str, call_type: str, line: int) -> None:
        if target not in self._nodes:
            return
        kind, lines = self.edges.get(target, (call_type, set()))
        if _CALL_RANK[call_type] < _CALL_RANK[kind]:
            kind = call_type
        if line:
            lines.add(line)
        self.edges[target] = (kind, lines)

    def call(self, target: str, call_type: str, offset: Optional[int], receiver: Optional[str] = None,
             site_type: Optional[str] = None, allow_self: bool = False) -> None:
        """Edge plus call site.  *site_type* overrides the type recorded on the call site.

        A call resolving to the caller itself is kept only when *allow_self*: conservative
        fan-out must not fabricate recursion.
        """
        if target == self.caller and not allow_self:
            return
        line = self.line_of(offset)
        self._put_edge(target, call_type, line)
        site_type = site_type or call_type
        if (target, line, site_type, receiver) not in self._seen:
            self._seen.add((target, line, site_type, receiver))
            self.sites.append(CallSite(self.caller, target, self.fn.file_path, line, site_type, receiver))


# ---------------------------------------------------------------------------
# Main call-graph builder
# ---------------------------------------------------------------------------

_QUALIFIED_CALL = re.compile(r'\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\(')
# Receiver may be indexed: sensors[i]->read()
_MEMBER_CALL = re.compile(r'\b([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*(->|\.)([A-Za-z_]\w*)\s*\(')
_BARE_CALL = re.compile(r'(?<![\w.:])(?<!->)([A-Za-z_]\w*)\s*\(')
# ClassName var;  /  ClassName* ptr = …;
_LOCAL_DECL = re.compile(r'\b([A-Z][A-Za-z_]\w*)\s*([\*&]?)\s+([a-z_]\w*)\s*(?:=|;|\()')
_CTOR_LOCAL = re.compile(r'\b([A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*;')
_CTOR_NEW = re.compile(r'\bnew\s+([A-Za-z_]\w*)\s*[\(;]')
_ARG_CALL = re.compile(r'\b[A-Za-z_]\w*\s*\(([^)]*)\)')
# T x;  T x(...);  T x = ...;  T x{...};  T x[n];  -- a non-pointer object
_LOCAL_OBJECT = re.compile(r'(?<![\w:.>])((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*[;=({\[]')
_NEW_OBJECT = re.compile(r'\bnew\s+((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)')


class CallGraphBuilder:
    def __init__(self, parse_results: Dict[str, ParseResult]) -> None:
        self._results = parse_results
        self._graph: CallGraph = nx.DiGraph()
        # Keyed by sig_key (e.g. "Calculator::compute(int)") — not by bare name.
        self._all_funcs: Dict[str, FunctionDef] = {}
        self._sigs: Dict[str, _Sig] = {}
        self._by_short: Dict[str, Set[str]] = {}
        self._by_qualified: Dict[str, Set[str]] = {}
        self.hierarchy = ClassHierarchy({})
        # Every resolved call, including ones whose target has no graph node (pure virtuals).
        self.call_sites: List[CallSite] = []
        self.instantiated_classes: Set[str] = set()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def built(self) -> bool:
        return self._graph.number_of_nodes() > 0

    def build(self) -> CallGraph:
        self.hierarchy = ClassHierarchy(self._results)
        self.call_sites = []

        # Phase 1: register every function as a graph node, keyed by sig_key.
        for result in self._results.values():
            for fn in result.functions:
                self._all_funcs[fn.sig_key] = fn
                self._graph.add_node(fn.sig_key, function=fn)
        self._index_functions()

        # Strip comments before any regex scanning so that commented-out calls
        # are not treated as live call sites.
        work = [(fn, _strip_comments(fn.body)) for result in self._results.values() for fn in result.functions]
        self.instantiated_classes = self._find_instantiated(work)

        # Phase 2: resolve call edges for every function.
        for fn, body in work:
            calls = self._extract_calls(fn, body)
            fn.calls.update(calls.edges)
            for callee_key, (call_type, lines) in calls.edges.items():
                self._graph.add_edge(calls.caller, callee_key, call_type=call_type, lines=sorted(lines))
                self._all_funcs[callee_key].called_by.add(calls.caller)
            self.call_sites.extend(calls.sites)

        logger.debug(
            f"Call graph: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges, {len(self.call_sites)} call sites"
        )
        return self._graph

    def find_reachable_from(self, function_name: str) -> Set[str]:
        """Return all sig_keys reachable from *function_name*.

        Accepts both a full sig_key (``"main()"``) and a bare name (``"main"``).
        """
        node = self._resolve_entry_point(function_name)
        if node is None:
            return set()
        return nx.descendants(self._graph, node)

    def find_unreachable(self, entry_points: List[str]) -> Set[str]:
        reachable: Set[str] = set()
        for ep in entry_points:
            node = self._resolve_entry_point(ep)
            if node:
                reachable.add(node)
                reachable |= self.find_reachable_from(ep)
        return set(self._graph.nodes) - reachable

    def get_function(self, name: str) -> FunctionDef | None:
        # Accept both sig_key and bare name for convenience.
        if name in self._all_funcs:
            return self._all_funcs[name]
        for sig_key, fn in self._all_funcs.items():
            if fn.name == name:
                return fn
        return None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _resolve_entry_point(self, name: str) -> Optional[str]:
        """Map an entry-point name (e.g. ``"main"``) to its sig_key graph node."""
        if name in self._graph:
            return name
        matches = [k for k, fn in self._all_funcs.items() if fn.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            zero_arg = [k for k in matches if not self._all_funcs[k].parameters]
            if zero_arg:
                return zero_arg[0]
            logger.warning(
                f"Entry point '{name}' is ambiguous ({len(matches)} overloads). "
                f"Using first match: {matches[0]}"
            )
            return matches[0]
        return None

    def _build_short_name_index(self, known_sig_keys: Set[str]) -> Dict[str, Set[str]]:
        """Map short base name → {sig_key, …}.

        Example:
            ``"Calculator::compute(int)"`` → key ``"compute"``
            ``"Utils::log()"``             → key ``"log"``
            ``"compute()"``                → key ``"compute"``
        """
        index: Dict[str, Set[str]] = {}
        for sig_key in known_sig_keys:
            short = sig_key.split('(')[0].rsplit('::', 1)[-1]
            index.setdefault(short, set()).add(sig_key)
        return index

    def _index_functions(self) -> None:
        """Lookup tables over the registered functions (and declared-only methods for signatures)."""
        self._by_short = self._build_short_name_index(set(self._all_funcs))
        # Qualified name (no param list) → {sig_key, …},
        # e.g. "Utils::log" → {"Utils::log()", "Utils::log(const std::string &)"}
        self._by_qualified = {}
        for sig_key in self._all_funcs:
            self._by_qualified.setdefault(sig_key.split('(')[0], set()).add(sig_key)
        # Default arguments are usually written on the in-class declaration, not on the definition.
        # shortcut: free-function prototypes are not parsed, so defaults given only there read as
        # required; upgrade by extracting prototypes in the parser.
        self._sigs = {}
        for sig_key, fn in self._all_funcs.items():
            declared = self.hierarchy.methods.get(sig_key)
            self._sigs[sig_key] = _Sig.of(fn.param_types, fn.parameters, *([declared.parameters] if declared else []))
        for sig_key, info in self.hierarchy.methods.items():
            if sig_key not in self._sigs:
                self._sigs[sig_key] = _Sig.of(info.param_types, info.parameters)

    @staticmethod
    def _scope(fn: FunctionDef) -> Optional[str]:
        """Class or namespace whose names are visible unqualified inside *fn*."""
        return fn.class_name or (fn.name.rpartition('::')[0] or None)

    def _match_site(self, candidates: Set[str], body: str, open_idx: int) -> List[str]:
        """Candidates a call whose ``(`` is at *open_idx* can resolve to, sorted."""
        args = _call_args(body, open_idx)
        # A call that is never closed cannot be matched by its arguments: keep every candidate.
        return sorted(set(candidates) if args is None else _site_matches(candidates, args, self._sigs))

    def _find_instantiated(self, work: List[Tuple[FunctionDef, str]]) -> Set[str]:
        """Classes that are constructed somewhere: local objects, ``new T``, object-typed fields."""
        found: Set[str] = set()
        # shortcut: field types count as instantiated even if the owner never is; upgrade with whole-program construction analysis.
        # shortcut: file-scope objects, make_unique/make_shared and container elements are not seen.
        for cdef in self.hierarchy.classes.values():
            for f in cdef.fields:
                if '*' not in f.type_ and '&' not in f.type_:
                    cls = self.hierarchy.resolve_class(bare_type(f.type_), cdef.name)
                    if cls:
                        found.add(cls)
        for fn, body in work:
            scope = self._scope(fn)
            for pattern in (_LOCAL_OBJECT, _NEW_OBJECT):
                for m in pattern.finditer(body):
                    cls = self.hierarchy.resolve_class(m.group(1), scope)
                    if cls:
                        found.add(cls)
        return found

    # ── Call extraction ─────────────────────────────────────────────────────

    def _extract_calls(self, fn: FunctionDef, body: str) -> _Calls:
        out = _Calls(fn, body, self._all_funcs)
        if not fn.body:
            return out
        self._qualified_calls(fn, body, out)
        self._member_calls(fn, body, out)
        self._bare_calls(fn, body, out)
        self._constructor_calls(body, out)
        self._callback_calls(body, out)
        self._self_calls(fn, body, out)
        return out

    def _qualified_calls(self, fn: FunctionDef, body: str, out: _Calls) -> None:
        """Fully-qualified calls, e.g. ``Utils::log(...)``.  A qualified call is never virtual."""
        for m in _QUALIFIED_CALL.finditer(body):
            qcall = m.group(1)
            qualifier, _, short = qcall.rpartition('::')
            candidates, kind = self._by_qualified.get(qcall), 'direct'
            if not candidates:
                # Base::m() where m is declared further up (or is pure): look it up in the class.
                cls = self.hierarchy.resolve_class(qualifier, self._scope(fn))
                inherited = self.hierarchy.find_methods(cls, short) if cls else []
                if inherited:
                    candidates = set(inherited)
                else:
                    # Name not stored as written (partial qualification, library call):
                    # short-name expansion with type-aware filtering.
                    candidates, kind = self._by_short.get(short, set()), 'unresolved'
            for sig_key in self._match_site(candidates, body, m.end() - 1):
                out.call(sig_key, kind, m.start())

    def _member_calls(self, fn: FunctionDef, body: str, out: _Calls) -> None:
        """``obj.method(...)`` and ``ptr->method(...)``, typed by parameters, locals and fields."""
        var_type = self._local_types(fn, body)
        scope = self._scope(fn)
        for m in _MEMBER_CALL.finditer(body):
            receiver, op, method = m.groups()
            offset, open_idx = m.start(), m.end() - 1
            if receiver == 'this':
                type_name, shown, dispatch = fn.class_name, None, True
            else:
                type_name, indirect = self._receiver_type(fn, var_type, receiver)
                shown, dispatch = receiver, op == '->' or indirect
            cls = self.hierarchy.resolve_class(type_name, scope) if type_name else None
            if cls:
                targets = self.hierarchy.find_methods(cls, method)
                if targets:
                    self._member_call(out, cls, targets, body, open_idx, offset, shown, dispatch)
                    continue

            candidates = self._by_short.get(method, set())
            if type_name:
                typed = {c for c in candidates if c.split('(')[0].startswith(type_name + '::')}
                if typed:
                    for sig_key in self._match_site(typed, body, open_idx):
                        out.call(sig_key, 'direct', offset, shown, allow_self=True)
                    continue
            # Receiver type unknown: conservative fan-out over every same-named function.
            for sig_key in self._match_site(candidates, body, open_idx):
                out.call(sig_key, 'unresolved', offset, shown)

    @staticmethod
    def _local_types(fn: FunctionDef, body: str) -> Dict[str, Tuple[str, bool]]:
        """Variable name → (type name, is pointer or reference) for parameters and local declarations."""
        var_type: Dict[str, Tuple[str, bool]] = {}
        # Strip qualifiers; the parser may attach * / & to either the type or the name field.
        for p in fn.parameters:
            bare_type_name = re.sub(r'[\*&]', '', p.type_).strip().split()[-1] if p.type_.strip() else ''
            bare_name = re.sub(r'[\*&\s]', '', p.name)
            if bare_name and bare_type_name:
                var_type[bare_name] = (bare_type_name, bool(re.search(r'[*&]', p.type_ + p.name)))
        for type_name, qualifier, var_name in _LOCAL_DECL.findall(body):
            var_type[var_name] = (type_name, bool(qualifier))
        return var_type

    def _receiver_type(self, fn: FunctionDef, var_type: Dict[str, Tuple[str, bool]],
                       receiver: str) -> Tuple[Optional[str], bool]:
        """Static type name of *receiver* (parameter, local, then field) and whether it is a pointer or reference."""
        if receiver in var_type:
            return var_type[receiver]
        cls = fn.class_name
        found = self.hierarchy.find_field(cls, receiver) if cls else None
        if cls is None or found is None:
            return None, False
        return self.hierarchy.field_type(cls, receiver), bool(re.search(r'[*&]', found[1].type_))

    def _bare_calls(self, fn: FunctionDef, body: str, out: _Calls) -> None:
        """Plain ``name(...)`` calls: methods of the enclosing class first (an implicit this-call),
        then free functions; with neither, conservative fan-out over same-named members."""
        cls = fn.class_name
        for m in _BARE_CALL.finditer(body):
            name, offset, open_idx = m.group(1), m.start(), m.end() - 1
            if cls:
                targets = self.hierarchy.find_methods(cls, name)
                if targets:
                    self._member_call(out, cls, targets, body, open_idx, offset, None)
                    continue
            candidates = self._by_short.get(name, set())
            free = {c for c in candidates if '::' not in c.split('(')[0]}
            # If a free function exists for this name use only free-function candidates;
            # otherwise the callee genuinely is unknown.
            kind, pool = ('direct', free) if free else ('unresolved', candidates)
            for sig_key in self._match_site(pool, body, open_idx):
                out.call(sig_key, kind, offset)

    def _member_call(self, out: _Calls, cls: str, targets: List[str], body: str, open_idx: int,
                     offset: int, receiver: Optional[str], dispatch: bool = True) -> None:
        """A call statically resolved to a method of *cls* (or one of its ancestors).

        Only a pointer, a reference or ``this`` can hold a derived object, so *dispatch* is False
        for a value receiver: it binds to the static target.
        """
        for target in self._match_site(set(targets), body, open_idx):
            info = self.hierarchy.method(target)
            virtual = dispatch and info is not None and info.is_virtual
            out.call(target, 'direct', offset, receiver, site_type='virtual' if virtual else 'direct',
                     allow_self=True)
            if virtual and info:
                self._dispatch(out, cls, info, offset)

    def _dispatch(self, out: _Calls, cls: str, info: MethodInfo, offset: int) -> None:
        """Virtual edges to the final overrider of *info* in every instantiated class that is
        *cls* or derives from it; overrides in classes never instantiated get no edge."""
        for x in [cls] + self.hierarchy.descendants(cls):
            if x in self.instantiated_classes:
                impl = self.hierarchy.find_method(x, info.short_name, info.param_types)
                if impl:
                    out.edge(impl, 'virtual', offset)

    def _constructor_calls(self, body: str, out: _Calls) -> None:
        """``T x;`` and ``new T(...)``."""
        for pattern in (_CTOR_LOCAL, _CTOR_NEW):
            for m in pattern.finditer(body):
                ctor_class = m.group(1)
                for sig_key in sorted(self._by_qualified.get(f"{ctor_class}::{ctor_class}", ())):
                    out.call(sig_key, 'constructor', m.start())

    def _callback_calls(self, body: str, out: _Calls) -> None:
        """Free-function names passed as arguments (callbacks / function pointers)."""
        for call in _ARG_CALL.finditer(body):
            for arg in re.finditer(r'\b([A-Za-z_]\w*)\b(?!\s*\()', call.group(1)):   # not a nested call
                for sig_key in sorted(self._by_short.get(arg.group(1), ())):
                    if '::' not in sig_key.split('(')[0]:
                        out.call(sig_key, 'callback', call.start(1) + arg.start())

    def _self_calls(self, fn: FunctionDef, body: str, out: _Calls) -> None:
        """Direct self-recursion by plain or qualified name.

        The passes above never emit a self-edge for these forms so conservative fan-out
        cannot fabricate recursion; only a call that resolves to this very overload counts.
        """
        parts = fn.name.split('::')
        # An unqualified m() inside a method is an implicit this-call, handled in _bare_calls.
        tokens = ['::'.join(parts[i:]) for i in range(len(parts) - (1 if fn.class_name else 0))]
        overloads = self._by_qualified.get(fn.name, {fn.sig_key})
        for token in tokens:
            for offset, args in _iter_call_sites(body, token):
                if fn.sig_key in _site_matches(overloads, args, self._sigs):
                    out.call(fn.sig_key, 'direct', offset, allow_self=True)
