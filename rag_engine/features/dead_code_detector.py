from __future__ import annotations
import bisect
import fnmatch
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
import networkx as nx
import yaml
from loguru import logger
from tree_sitter import Node

from rag_engine.config import DEFAULT_ENTRY_POINT_PATTERNS
from rag_engine.core.graph_builder import CallGraphBuilder, _strip_comments
from rag_engine.features.control_flow_analyzer import declares_name, find_dead_fragments, find_unused_variables, parse_root
from rag_engine.models import DeadCodeItem, FunctionDef, ParseResult

_NAME_TOKENS = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')   # camelCase / snake_case / ACRONYM words
_CALLBACK_TOKENS = {'callback', 'handler', 'hook', 'listener', 'isr', 'irq', 'interrupt', 'init'}
# A name or scope segment that starts with test/mock/stub/fixture (mockSensor, TestHelper::run); not selfTest.
_TEST_PAT = re.compile(r'(?:^|::)(?:[Tt]est|[Mm]ock|[Ss]tub|[Ff]ixture)(?![a-z])')
_OPERATOR_PAT = re.compile(r'operator(?!\w)')
_DEFINE_LINE = re.compile(r'^[ \t]*#[ \t]*define\b(?:[^\n]*\\\n)*[^\n]*', re.M)
_DIRECTIVE = re.compile(r'^[ \t]*#[ \t]*(ifdef|ifndef|if|elif|else|endif)\b([^\n]*)', re.M)
_IDENT = re.compile(r'\b[A-Za-z_]\w*')
_ADDRESS_OF = re.compile(r'&\s*(?:(?:\w+\s*)?::\s*)*$')   # "&", "&C::", "&Ns::C::", "&::" right before a name
_SCOPE = re.compile(r'(?:(?:\w+\s*)?::\s*)+$')               # "C::", "Ns::C::", "::" right before a name
_ASSIGNED_OR_LISTED = re.compile(r'(?:[=,{(]|\breturn)$')
_UNARY_KEYWORD = re.compile(r'\b(?:return|case|throw|co_return|co_yield)$')
_CATEGORIES = ('dead_code', 'deactivated_code', 'unused_export', 'dead_fragment', 'unused_variable')
_HEADER_SUFFIXES = ('.h', '.hpp')
_PROTOTYPE = re.compile(r'^[ \t]*(?!(?:return|else|throw|delete|case|goto)\b)[A-Za-z_][\w:<>,*&\s]*?[\s*&]'
                        r'(operator[^\s(]+|[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?;', re.M)   # name of a function prototype
_MAX_EVIDENCE = 3
DISPOSITION_STATUSES = ('open', 'confirmed_dead', 'false_positive', 'deactivated')
CLOSED_STATUSES = DISPOSITION_STATUSES[1:]      # a reviewer's outcome; 'open' means no decision yet


def _is_callback_name(qualified: str) -> bool:
    """Short name starts with ``on`` + capital (onTimerTick) or has a callback word as a whole token
    (sensorHandler, uartIsr, timer_callback, init); ``chandler`` and ``onionPeel`` do not match."""
    short = qualified.rsplit('::', 1)[-1]
    return bool(re.match(r'on[A-Z]', short)) or any(t.lower() in _CALLBACK_TOKENS for t in _NAME_TOKENS.findall(short))


class _Ref(NamedTuple):
    """One place a wanted name occurs without being called."""
    path: str
    offset: int       # character offset in the comment-stripped text (= byte offset in the source)
    kind: str         # 'define' | 'template' | 'address' | 'value'
    qualified: bool   # written behind a scope: C::name


def _in_spans(spans: List[Tuple[int, int]], pos: int) -> bool:
    i = bisect.bisect_right(spans, (pos, float('inf'))) - 1
    return i >= 0 and spans[i][0] <= pos < spans[i][1]


def _never_built_spans(text: str) -> List[Tuple[int, int]]:
    """Spans of ``#if 0`` / ``#if false`` branches (the ``#else`` and ``#elif`` branches are live)."""
    # shortcut: only the literals 0 and false; the #else branch of an always-true #if is not treated as dead.
    spans: List[Tuple[int, int]] = []
    frames: List[bool] = []          # per open conditional: is the current branch constant-false
    start = 0
    for m in _DIRECTIVE.finditer(text):
        was_dead = any(frames)
        word, rest = m.group(1), m.group(2).strip()
        if word in ('if', 'ifdef', 'ifndef'):
            frames.append(word == 'if' and rest in ('0', 'false'))
        elif word == 'elif' and frames:
            frames[-1] = rest in ('0', 'false')
        elif word == 'else' and frames:
            frames[-1] = False
        elif word == 'endif' and frames:
            frames.pop()
        if any(frames) and not was_dead:
            start = m.end()
        elif was_dead and not any(frames):
            spans.append((start, m.start()))
    if any(frames):
        spans.append((start, len(text)))
    return spans


def _reference_kind(before: str, after: str) -> Optional[Tuple[str, bool]]:
    """``(kind, qualified)`` for a name that is neither called nor followed by ``<``: ``'address'`` for a
    unary ``&name`` / ``&C::name``, ``'value'`` for a name in value position (after ``= , { (`` or
    ``return``, before ``, } ; )``); None for any other use."""
    amp = _ADDRESS_OF.search(before)
    if amp:
        left = before[:amp.start()].rstrip()
        prev = left[-1:]
        # Unary after an operator, a keyword (return &f) or a cast ((void*)&f); a word, ], > , & or * before
        # it means a binary &, && or a reference declarator (Config& c, Frame&& f).
        if not (prev.isalnum() or prev in ('_', ']', '>', '&', '*')) or _UNARY_KEYWORD.search(left):
            return 'address', bool(re.search(r'\w\s*::', amp.group()))
    core = before.rstrip()
    scope = _SCOPE.search(core)
    if scope:
        core = core[:scope.start()].rstrip()
    if _ASSIGNED_OR_LISTED.search(core) and after[:1] in (',', '}', ';', ')'):
        return 'value', scope is not None
    return None


class _Verdict(NamedTuple):
    category: str
    disposition: str
    confidence: str
    reason: str
    kind: str                            # finding ID kind
    evidence: Optional[List[str]] = None   # replaces the call-graph evidence when set


@dataclass
class DeadCodeReport:
    items: List[DeadCodeItem]
    total_functions: int
    dead_count: int
    deactivated_count: int
    false_positive_notes: List[str]
    structural_coverage_impact: str
    warnings: List[str] = field(default_factory=list)
    entry_points_used: List[str] = field(default_factory=list)
    counts_by_category: Dict[str, int] = field(default_factory=dict)
    unresolved_calls: List[str] = field(default_factory=list)   # conservative call edges: "<caller> → <target> (<file>:<line>)"


class DeadCodeDetector:
    def __init__(self, graph_builder: CallGraphBuilder, parse_results: Dict[str, ParseResult],
                 entry_points: List[str] | None = None, entry_point_patterns: List[str] | None = None,
                 dispositions_file: str | None = None) -> None:
        self._graph = graph_builder
        self._results = parse_results
        self._entry_points = entry_points or ['main']
        self._patterns = list(DEFAULT_ENTRY_POINT_PATTERNS if entry_point_patterns is None else entry_point_patterns)
        self._dispositions_file = dispositions_file
        # Keyed by sig_key to match the call graph nodes (the bare name collapses overloads).
        self._all_funcs: Dict[str, FunctionDef] = {
            fn.sig_key: fn for r in parse_results.values() for fn in r.functions
        }
        self._wanted: Set[str] = set()               # short names whose references are worth indexing
        self._ref_index: Optional[Dict[str, List[_Ref]]] = None
        self._line_starts: Dict[str, List[int]] = {}   # offset of each line start, per file
        self._scopes: Dict[str, Tuple[List[int], list, List[int]]] = {}   # per file: function spans, nesting
        self._trees: Dict[str, Node] = {}
        self._prototypes: Optional[Dict[str, List[str]]] = None   # function prototyped in a header -> those headers
        try:
            self._root = os.path.commonpath([str(Path(p).parent) for p in parse_results]) if parse_results else ''
        except ValueError:                           # relative and absolute paths, or different drives
            self._root = ''

    def analyze(self) -> DeadCodeReport:
        gb = self._graph
        if not gb.built:
            gb.build()
        entries, warnings = gb.resolve_entry_points(self._entry_points, self._patterns)
        warnings += self._pattern_warnings()
        reachable = gb.reachable_from(entries | gb.implicit_entry_points())
        unreachable = sorted(set(self._all_funcs) - reachable)
        entry_names = sorted(entries)
        self._wanted = {self._all_funcs[k].name.rsplit('::', 1)[-1] for k in unreachable}
        self._ref_index = None

        taken: Counter = Counter()
        items = [self._function_item(self._all_funcs[key], entry_names, taken) for key in unreachable]
        self._investigate_callees_of_possibly_live(items)
        items += self._flow_items(taken, warnings)
        items.sort(key=lambda i: (i.file_path, i.line_number, i.finding_id))

        total = len(self._all_funcs)
        if total and gb.graph.number_of_edges() == 0:
            warnings.append("The call graph has no call edges; every function that is not an entry point is unreachable")
        if total and len(unreachable) * 2 > total:
            warnings.append(f"{len(unreachable)} of {total} functions ({len(unreachable) * 100 // total}%) are "
                            "unreachable; check entry_points configuration")
        self._apply_dispositions(items, warnings)

        counts = Counter(i.category for i in items)
        notes = [f"{i.name} is not proposed for removal: {i.reason}" for i in items
                 if i.symbol_type in ('function', 'method') and i.category != 'dead_code' and i.confidence != 'high']
        unresolved = sorted({f"{s.caller} → {s.target} ({s.file_path}:{s.line})"
                             for s in gb.call_sites if s.call_type == 'unresolved'})
        logger.info(f"Dead code: {len(items)} findings ({dict(counts)})")
        return DeadCodeReport(
            items=items, total_functions=total,
            dead_count=counts['dead_code'], deactivated_count=counts['deactivated_code'],
            false_positive_notes=notes, structural_coverage_impact=self._coverage_impact(items),
            warnings=warnings, entry_points_used=entry_names,
            counts_by_category={c: counts[c] for c in _CATEGORIES}, unresolved_calls=unresolved)

    def _pattern_warnings(self) -> List[str]:
        """A configured pattern that matches nothing (a typo) and patterns that match most of the code
        (``*`` from a mistyped list), which would hide every dead function."""
        # Same rule as CallGraphBuilder.resolve_entry_points: a pattern without `::` matches the short name of
        # a free function, one with `::` the qualified name; code under `#if 0` is never an entry point.
        live = [(key, fn.name, fn.name.rsplit('::', 1)[-1], fn.class_name is None)
                for key, fn in self._all_funcs.items() if not fn.never_built]
        warnings: List[str] = []
        matched: Set[str] = set()
        for pattern in self._patterns:
            qualified = '::' in pattern
            hits = {key for key, name, short, free in live
                    if (qualified and fnmatch.fnmatchcase(name, pattern)) or (not qualified and free and fnmatch.fnmatchcase(short, pattern))}
            matched |= hits
            # the built-in patterns are speculative (most codebases have no `*_ISR`): only the user's own are checked
            if not hits and pattern not in DEFAULT_ENTRY_POINT_PATTERNS:
                warnings.append(f"entry_point_pattern '{pattern}' matches nothing")
        total = len(self._all_funcs)
        if total and len(matched) * 2 > total:
            warnings.append(f"entry_point_patterns match {len(matched)} of {total} functions — check the configuration")
        return warnings

    # ── Functions ───────────────────────────────────────────────────────────

    def _classify(self, fn: FunctionDef, entry_names: List[str]) -> _Verdict:
        """The first matching rule wins."""
        guard = fn.preproc_guard
        if fn.never_built:
            return _Verdict('deactivated_code', 'Justify as Deactivated', 'high',
                            f"excluded from the build by `#if {guard}`", 'CFG')
        if guard:
            return _Verdict('deactivated_code', 'Justify as Deactivated', 'medium', f"only built when `{guard}`", 'CFG')
        kind = 'METH' if fn.class_name else 'FUNC'
        if _TEST_PAT.search(fn.name):
            return _Verdict('deactivated_code', 'Justify as Deactivated', 'low', 'test-support code', kind)
        if fn.is_virtual:
            if fn.class_name and not self._instantiated(fn.class_name):
                what = 'override' if fn.is_override else 'virtual method'
                return _Verdict('unused_export', 'Investigate', 'medium', f"{what} in class never instantiated", kind)
            return _Verdict('unused_export', 'Investigate', 'medium', 'virtual method never called or dispatched', kind)
        if fn.is_extern_c:
            return _Verdict('unused_export', 'Investigate', 'low',
                            'defined in an `extern "C"` block: may be called from outside the analyzed build', kind)
        short = fn.name.rsplit('::', 1)[-1]
        if _OPERATOR_PAT.match(short):
            return _Verdict('unused_export', 'Investigate', 'low',
                            'operator function: used through expression syntax, not through a call the graph can see', kind)
        if _is_callback_name(fn.name):
            return _Verdict('unused_export', 'Investigate', 'low',
                            'name suggests callback/ISR; add to entry_points if invoked externally', kind)
        references = self._references(fn, short)
        if references:
            return _Verdict('unused_export', 'Investigate', 'low',
                            'referenced by name outside the call graph (function pointer, table, macro or template use)',
                            kind, references)
        return _Verdict('dead_code', 'Remove', 'high',
                        f"no call path from entry points [{', '.join(entry_names) or 'none resolved'}]", kind)

    def _reached_from(self, seeds: List[str]) -> Dict[str, List[str]]:
        """function -> up to three of *seeds* that reach it."""
        graph = self._graph.graph
        labels: Dict[str, List[str]] = {}
        for seed in seeds:
            seen = {seed}
            stack = list(graph.successors(seed))
            while stack:
                node = stack.pop()
                if node in seen or len(labels.get(node, ())) >= _MAX_EVIDENCE:   # saturated: its callees are too
                    continue
                seen.add(node)
                labels.setdefault(node, []).append(seed)
                stack.extend(graph.successors(node))
        return labels

    def _investigate_callees_of_possibly_live(self, items: List[DeadCodeItem]) -> None:
        """A function under Investigate may be live, and so may everything only it calls: those functions
        move from dead_code to Investigate/low.  Everything only called from code built under a
        ``#ifdef`` / ``#if X`` guard moves to Justify as Deactivated/medium (other configurations build it).
        The call path is the evidence.  Callees of ``#if 0`` code stay dead: no configuration builds it."""
        graph = self._graph.graph
        # shortcut: a callee shared by guarded and possibly-live code is Investigate (the more cautious verdict).
        live = self._reached_from(sorted(i.name for i in items if i.category == 'unused_export'))
        guarded = self._reached_from(sorted(i.name for i in items if i.category == 'deactivated_code'
                                            and self._all_funcs[i.name].preproc_guard
                                            and not self._all_funcs[i.name].never_built))
        for item in items:
            if item.category != 'dead_code':
                continue
            if item.name in live:
                seeds = live[item.name]
                item.category, item.do178c_disposition, item.confidence = 'unused_export', 'Investigate', 'low'
                item.reason = f"only reachable from possibly-live {', '.join(seeds)}"
            elif item.name in guarded:
                seeds = guarded[item.name]
                item.category, item.do178c_disposition, item.confidence = 'deactivated_code', 'Justify as Deactivated', 'medium'
                item.reason = ("only reachable from code built when "
                               + ', '.join(f"{self._all_funcs[s].preproc_guard} ({s})" for s in seeds))
            else:
                continue
            item.evidence = [' → '.join(nx.shortest_path(graph, seeds[0], item.name))]

    def _instantiated(self, cls: str) -> bool:
        """The class or one of its subclasses is constructed somewhere."""
        gb = self._graph
        return cls in gb.instantiated_classes or any(d in gb.instantiated_classes for d in gb.hierarchy.descendants(cls))

    def _references(self, fn: FunctionDef, short: str) -> List[str]:
        """Up to three ``file:line`` places that name *fn* without calling it.

        Always counted: a ``#define`` line, explicit template arguments (``f<int>``).  A unary ``&f`` /
        ``&C::f`` counts for free functions and static members, and as ``&C::f`` for other members.  A bare
        value (table, initializer, assignment, argument) counts only for free functions and static members,
        and not where it declares a variable, parameter, field or enumerator of that name: a non-static
        member function cannot be named as a bare value.
        """
        # shortcut: matched by name only, so a variable of the same name used as a value still counts for
        # a free function; this errs towards Investigate. Upgrade with symbol resolution if it hides too much.
        if self._ref_index is None:
            self._build_reference_index()
        assert self._ref_index is not None
        bare_ok = not fn.class_name or self._is_static(fn)
        found: List[str] = []
        for ref in self._ref_index.get(short, ()):
            counts = (ref.kind in ('define', 'template')
                      or (ref.kind == 'address' and (ref.qualified or bare_ok))
                      or (ref.kind == 'value' and bare_ok and not self._declares(ref, short)))
            if not counts:
                continue
            line = bisect.bisect_right(self._line_starts[ref.path], ref.offset)
            if ref.kind != 'define' and self._is_local(ref.path, line, short):
                continue
            place = f"{self._display_path(ref.path)}:{line}"
            if place not in found:
                found.append(place)
            if len(found) == _MAX_EVIDENCE:
                break
        return found

    def _is_local(self, path: str, line: int, name: str) -> bool:
        """*name* is a parameter or local of the function that contains *line*, or of one around it
        (so the occurrence is that variable, not the function)."""
        if path not in self._scopes:
            spans = sorted(((f.line_number, max(f.end_line_number, f.line_number),
                             f.local_names | {re.sub(r'[*&\s]', '', p.name) for p in f.parameters})
                            for f in self._results[path].functions), key=lambda s: (s[0], -s[1]))
            parents: List[int] = []
            stack: List[int] = []
            for i, (start, _, _) in enumerate(spans):
                while stack and spans[stack[-1]][1] < start:
                    stack.pop()
                parents.append(stack[-1] if stack else -1)
                stack.append(i)
            self._scopes[path] = ([s[0] for s in spans], spans, parents)
        starts, spans, parents = self._scopes[path]
        i = bisect.bisect_right(starts, line) - 1
        while i >= 0 and spans[i][1] < line:    # the nearest function that starts before is already over
            i = parents[i]
        while i >= 0:
            if name in spans[i][2]:
                return True
            i = parents[i]
        return False

    def _is_static(self, fn: FunctionDef) -> bool:
        info = self._graph.hierarchy.method(fn.sig_key)
        return fn.is_static or (info is not None and info.is_static)

    def _declares(self, ref: _Ref, name: str) -> bool:
        if ref.path not in self._trees:
            self._trees[ref.path] = parse_root(self._results[ref.path])
        return declares_name(self._trees[ref.path], ref.offset, ref.offset + len(name))

    def _build_reference_index(self) -> None:
        """One pass over the comment-stripped text of every file, recording where each wanted name (the
        short names of the unreachable functions) occurs without being called and outside ``#if 0``."""
        self._line_starts = {}
        index: Dict[str, List[_Ref]] = {}
        for path, result in sorted(self._results.items()):
            text = _strip_comments(result.raw_source.decode('latin-1'))   # one character per byte
            starts, pos = [0], 0
            for line in text.split('\n')[:-1]:
                pos += len(line) + 1
                starts.append(pos)
            self._line_starts[path] = starts
            defines = [m.span() for m in _DEFINE_LINE.finditer(text)]
            dead = _never_built_spans(text)
            for m in _IDENT.finditer(text):
                name, pos = m.group(), m.start()
                if name not in self._wanted or (pos and text[pos - 1] == '~') or _in_spans(dead, pos):
                    continue
                if _in_spans(defines, pos):
                    found: Optional[Tuple[str, bool]] = ('define', False)
                else:
                    after = text[m.end():m.end() + 24].lstrip()
                    if after.startswith('('):
                        continue
                    found = (('template', False) if after.startswith('<')
                             else _reference_kind(text[max(0, pos - 80):pos], after))
                if found:
                    index.setdefault(name, []).append(_Ref(path, pos, found[0], found[1]))
        self._ref_index = index

    def _display_path(self, path: str) -> str:
        """*path* relative to the common directory of the analysed files, so same-named files differ."""
        if self._root:
            try:
                return Path(os.path.relpath(path, self._root)).as_posix()
            except ValueError:
                pass
        return path

    def _function_item(self, fn: FunctionDef, entry_names: List[str], taken: Counter) -> DeadCodeItem:
        verdict = self._classify(fn, entry_names)
        key = fn.sig_key
        end = max(fn.end_line_number, fn.line_number)
        return DeadCodeItem(
            name=key, file_path=fn.file_path, line_number=fn.line_number,
            category=verdict.category, do178c_disposition=verdict.disposition,  # type: ignore[arg-type]
            coverage_impact=f"~{end - fn.line_number + 1} line(s); {fn.cyclomatic_complexity} decision point(s)",
            finding_id=self._finding_id(verdict.kind, fn.file_path, key, '', taken), end_line=end,
            symbol_type='method' if fn.class_name else 'function', function_name=key, reason=verdict.reason,
            confidence=verdict.confidence,  # type: ignore[arg-type]
            evidence=verdict.evidence or self._evidence(key),
            removal_impact=self._removal_impact(fn) or ['none identified'])

    def _evidence(self, key: str) -> List[str]:
        graph = self._graph.graph
        callers = sorted(graph.predecessors(key))
        if not callers:
            return ['callers: none']
        types = sorted({graph.edges[c, key]['call_type'] for c in callers})
        return [f"only called by unreachable: {', '.join(callers)}", f"edge types: {', '.join(types)}"]

    def _removal_impact(self, fn: FunctionDef) -> List[str]:
        graph = self._graph.graph
        key = fn.sig_key
        impact = [f"{callee} would lose its only caller" for callee in sorted(graph.successors(key))
                  if callee != key and set(graph.predecessors(callee)) <= {key}]
        params = {re.sub(r'[*&\s]', '', p.name) for p in fn.parameters}
        writes = sorted(fn.data_writes - params)
        if writes:
            impact.append(f"writes members/globals: {', '.join(writes)}")
        if fn.do178c_requirement_trace:
            impact.append(f"update trace matrix ({fn.do178c_requirement_trace})")
        header = self._declaring_header(fn)
        if header:
            impact.append(f"remove declaration in {Path(header).name}")
        return impact

    def _declaring_header(self, fn: FunctionDef) -> Optional[str]:
        """File that declares *fn* apart from the one defining it, if any."""
        info = self._graph.hierarchy.method(fn.sig_key) if fn.class_name else None
        if info is not None:
            return info.file_path if info.file_path != fn.file_path else None
        # shortcut: a free function's prototype is found by name in the other headers (overloads are
        # not told apart); upgrade by parsing prototypes in the parser.
        short = fn.name.rsplit('::', 1)[-1]
        return next((path for path in self._prototype_headers(short) if path != fn.file_path), None)

    def _prototype_headers(self, short: str) -> List[str]:
        """Headers (sorted) that declare a free function *short*: one pass over every header, on first use."""
        if self._prototypes is None:
            self._prototypes = {}
            for path, result in sorted(self._results.items()):
                if path.endswith(_HEADER_SUFFIXES):
                    for name in {m.group(1) for m in _PROTOTYPE.finditer(result.raw_source.decode('utf-8', errors='replace'))}:
                        self._prototypes.setdefault(name, []).append(path)
        return self._prototypes.get(short, [])

    # ── Fragments and variables ─────────────────────────────────────────────

    def _flow_items(self, taken: Counter, warnings: List[str]) -> List[DeadCodeItem]:
        """Dead fragments and unused variables, file by file; a file that cannot be analysed is a warning."""
        items: List[DeadCodeItem] = []
        for path in sorted(self._results):
            try:
                fragments = find_dead_fragments(self._results[path])
                variables = find_unused_variables(self._results[path])
            except Exception as exc:
                warnings.append(f"Control-flow analysis of {path} failed ({exc}); no fragments or unused variables reported for it")
                continue
            for frag in fragments:
                evidence = (f"follows unconditional `{frag.jump}` at line {frag.jump_line}" if frag.jump
                            else f"condition `{frag.condition}` is constant false")
                impact = (f"contains calls to {', '.join(frag.calls)} (never executed); "
                          "removing the fragment removes these call sites" if frag.calls
                          else 'no behaviour change expected: statements are unreachable')
                items.append(DeadCodeItem(
                    name=frag.snippet, file_path=frag.file_path, line_number=frag.start_line,
                    category='dead_fragment', do178c_disposition='Fix Fragment',
                    coverage_impact=f"~{frag.end_line - frag.start_line + 1} uncoverable line(s)",
                    finding_id=self._finding_id('FRAG', frag.file_path, frag.function, frag.snippet, taken),
                    end_line=frag.end_line, symbol_type='fragment', function_name=frag.function,
                    reason=frag.reason, confidence='high', evidence=[evidence], removal_impact=[impact]))
            for var in variables:
                local = var.kind == 'local'
                if local:
                    evidence = f"declared at line {var.line}; no later reference"
                    impact = ([f"initializer has side effects ({', '.join(var.side_effects)}) — "
                               "keep the expression when removing the variable"] if var.side_effects else ['none'])
                else:
                    evidence = f"declared at line {var.line}; no reference in the function"
                    impact = [self._signature_impact(var.function)]
                    fn = self._all_funcs.get(var.function)
                    if fn is not None and fn.is_virtual:
                        impact.append('part of a virtual signature: overrides must keep the parameter')
                items.append(DeadCodeItem(
                    name=var.name, file_path=var.file_path, line_number=var.line,
                    category='unused_variable', do178c_disposition='Remove' if local else 'Investigate',
                    coverage_impact='1 declaration line; no executable statement',
                    finding_id=self._finding_id('VAR', var.file_path, var.function, var.name, taken),
                    end_line=var.line, symbol_type='variable' if local else 'parameter',
                    function_name=var.function,
                    reason=(f"local variable `{var.name}` is never used after its declaration" if local else
                            f"parameter `{var.name}` is never used in the function body"),
                    confidence='medium' if local else 'low', evidence=[evidence], removal_impact=impact))
        return items

    def _signature_impact(self, function: str) -> str:
        graph = self._graph.graph
        callers = sorted(c for c in graph.predecessors(function) if c != function) if function in graph else []
        return f"signature change affects callers: {', '.join(callers) or 'none known'}"

    # ── IDs, dispositions, summary ──────────────────────────────────────────

    @staticmethod
    def _finding_id(kind: str, file_path: str, symbol: str, detail: str, taken: Counter) -> str:
        """``DC-<kind>-<8 hex>``: no line numbers go into the hash, so the ID survives edits above it.
        Findings that would share an ID (identical fragments in one function) are told apart by their order."""
        text = f"{Path(file_path).name}|{symbol}|{kind}|{detail}"
        seen = taken[text]
        taken[text] += 1
        if seen:
            text += f"#{seen}"
        return f"DC-{kind}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8].upper()}"

    def _apply_dispositions(self, items: List[DeadCodeItem], warnings: List[str]) -> None:
        """``findings: {<id>: {status, justification, reviewer}}``; an entry needs a status (one of
        DISPOSITION_STATUSES) and a justification, and is ignored with a warning otherwise."""
        path = self._dispositions_file
        if not path:
            return
        try:
            data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        except FileNotFoundError:
            warnings.append(f"Dispositions file not found: {path}")
            return
        except (OSError, yaml.YAMLError) as exc:
            warnings.append(f"Dispositions file {path} could not be read: {exc}")
            return
        entries = data.get('findings') if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            warnings.append(f"Dispositions file {path} has no 'findings' mapping")
            return
        by_id = {i.finding_id: i for i in items}
        for finding_id, entry in entries.items():
            item = by_id.get(finding_id)
            if item is None:
                warnings.append(f"stale disposition {finding_id}: no finding has this ID")
                continue
            entry = entry if isinstance(entry, dict) else {}
            status = str(entry.get('status') or '').strip()
            justification = str(entry.get('justification') or '').strip()
            if not justification or not status:
                warnings.append(f"Disposition {finding_id} ignored: "
                                f"no {'justification' if not justification else 'status'}")
                continue
            if status not in DISPOSITION_STATUSES:
                warnings.append(f"disposition {finding_id}: unknown status '{status}' "
                                f"(allowed: {', '.join(DISPOSITION_STATUSES)})")
                continue
            item.review_status, item.reviewer_justification = status, justification
            item.reviewer = str(entry['reviewer']) if entry.get('reviewer') else None

    @staticmethod
    def _coverage_impact(items: List[DeadCodeItem]) -> str:
        """Union of the line spans of function and fragment findings, so nested spans count once."""
        lines: Dict[str, Set[int]] = {}
        for i in items:
            if i.symbol_type in ('function', 'method', 'fragment'):
                lines.setdefault(i.file_path, set()).update(range(i.line_number, i.end_line + 1))
        return f"{len(items)} findings; ~{sum(len(v) for v in lines.values())} uncoverable source lines"
