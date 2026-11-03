from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Literal, Optional, Set, Tuple

import tree_sitter_cpp as tscpp
from loguru import logger
from tree_sitter import Language, Node, Parser

from rag_engine.core.ast_builder import find_nodes, get_text
from rag_engine.core.class_hierarchy import MethodInfo, bare_type
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.features.sensor_template import SensorTemplate
from rag_engine.models import FunctionDef, ParseResult

_READER_NAME = re.compile(r'(?i)read|sample|acquire')                        # fullmatch on a short name
_GETTER_NAME = re.compile(r'(?i)get\w*')
_VALIDATOR_NAME = re.compile(r'(?i)check|valid|range|stale|debounce|filter|limit')
_NAME_TOKEN = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')
_STALE_TOKENS = {'ms', 'msec', 'millis', 'us', 'usec', 'sec', 'secs', 'time', 'timestamp', 'stamp', 'elapsed',
                 'timeout', 'stale', 'tick', 'ticks', 'age', 'now'}
_DEBOUNCE_TOKENS = {'debounce', 'prev', 'previous', 'last', 'delta', 'diff', 'stable'}
_ABS_FUNCTIONS = {'abs', 'fabs', 'fabsf'}
_RELATIONAL = {'<', '<=', '>', '>='}
# Scopes whose declarations are still file scope (include guards wrap whole headers).
_FILE_SCOPES = {'namespace_definition', 'declaration_list', 'linkage_specification',
                'preproc_if', 'preproc_ifdef', 'preproc_elif', 'preproc_else'}
_COMMENT = re.compile(r'//.*|/\*.*?\*/')


@dataclass
class CheckResult:
    # not_required: explicitly disabled in the template; not_configured: the template gives no value to check against
    status: Literal['implemented', 'mismatch', 'missing', 'not_required', 'not_configured']
    expected: str                 # e.g. "range [-54, 54]", "threshold 500 ms"
    found: List[float] = field(default_factory=list)       # constants of the matching comparisons
    evidence: List[str] = field(default_factory=list)      # "<file>:<line> <sig_key>: <comparison text>"


@dataclass
class SensorFinding:
    template: SensorTemplate
    class_name: Optional[str]
    status: Literal['found', 'not_found']
    file_path: str = ''
    line_number: int = 0
    instantiated: bool = False
    hierarchy: List[str] = field(default_factory=list)     # parameter group -> I/O group -> class -> data
    data_members: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)       # "<sig_key> (<file>:<line>)"
    processors: List[str] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)
    checks: Dict[str, CheckResult] = field(default_factory=dict)    # corrupt / stale / debounce


@dataclass
class SensorAnalysisResult:
    findings: List[SensorFinding]
    unmapped_sensor_classes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Comparison:
    line: int
    text: str
    kind: Literal['range', 'stale', 'debounce']
    values: Tuple[float, ...]            # the numeric constants among the two operands
    operands: FrozenSet[str]             # identifiers of the non-constant operand
    calls: FrozenSet[str]                # short names of functions called in the non-constant operand
    in_for_condition: bool


@dataclass
class _Facts:
    comparisons: List[_Comparison]
    assigned: Dict[str, Set[str]]        # variable -> short names of functions called in the values assigned to it


def _fmt(x: float) -> str:
    return f'{x:.10g}'


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def _tokens(ident: str) -> Set[str]:
    """``'lastUpdateMs'`` / ``'TEMP_STALE_MS'`` -> lower-case words, split on underscores and camel case."""
    return {t.lower() for t in _NAME_TOKEN.findall(ident)}


def _key(info: MethodInfo) -> Tuple:
    return info.short_name, tuple(info.param_types)


def _literal(text: str) -> Optional[float]:
    """Value of a C++ numeric literal; suffixes (f/F/u/U/l/L) and digit separators are dropped."""
    t = text.strip().replace("'", '')
    try:
        if t[:2] in ('0x', '0X'):
            return float(int(t.rstrip('uUlL'), 16))
        if t[:2] in ('0b', '0B'):
            return float(int(t.rstrip('uUlL')[2:], 2))
        t = t.rstrip('fFuUlL')
        value = float(int(t, 8)) if re.fullmatch(r'0[0-7]+', t) else float(t)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _value(node: Optional[Node], src: bytes, consts: Dict[str, float]) -> Optional[float]:
    """Constant value of an operand: a literal, a signed/parenthesised one, or a known named constant."""
    if node is None:
        return None
    kind = node.type
    if kind == 'number_literal':
        return _literal(get_text(node, src))
    if kind == 'parenthesized_expression':
        return _value(node.named_children[0] if node.named_children else None, src, consts)
    if kind == 'unary_expression':
        operator = node.child_by_field_name('operator')
        inner = _value(node.child_by_field_name('argument'), src, consts)
        sign = get_text(operator, src) if operator is not None else ''
        return None if inner is None or sign not in ('-', '+') else (-inner if sign == '-' else inner)
    if kind in ('identifier', 'qualified_identifier'):
        return consts.get(get_text(node, src).rsplit('::', 1)[-1])
    return None


def _file_scope(node: Node) -> Iterator[Node]:
    for child in node.children:
        if child.type in _FILE_SCOPES:
            yield from _file_scope(child)
        else:
            yield child


def _identifiers(node: Node, src: bytes) -> List[str]:
    return [get_text(n, src) for n in find_nodes(node, 'identifier', 'field_identifier')]


def _callee_name(call: Node, src: bytes) -> str:
    """Short name of the function a call expression calls (``s.read()`` -> ``read``), '' if not a plain name."""
    callee = call.child_by_field_name('function')
    m = re.search(r'(\w+)\s*$', get_text(callee, src)) if callee is not None else None
    return m.group(1) if m else ''


def _called_names(node: Optional[Node], src: bytes) -> Set[str]:
    calls = find_nodes(node, 'call_expression') if node is not None else []
    return {n for n in (_callee_name(c, src) for c in calls) if n}


def _has_abs_of_difference(node: Node, src: bytes) -> bool:
    for call in find_nodes(node, 'call_expression'):
        args = call.child_by_field_name('arguments')
        if args is None or _callee_name(call, src) not in _ABS_FUNCTIONS:
            continue
        for expr in find_nodes(args, 'binary_expression'):
            operator = expr.child_by_field_name('operator')
            if operator is not None and get_text(operator, src) == '-':
                return True
    return False


def _in_for_condition(node: Node) -> bool:
    parent = node.parent
    while parent is not None:
        if parent.type == 'for_statement':
            cond = parent.child_by_field_name('condition')
            if cond is not None and cond.start_byte <= node.start_byte and node.end_byte <= cond.end_byte:
                return True
        parent = parent.parent
    return False


def _classify(node: Node, src: bytes) -> Literal['range', 'stale', 'debounce']:
    # shortcut: identifier-pattern classification, upgrade to data-flow tracking of the sensor value.
    tokens = set().union(*(_tokens(i) for i in _identifiers(node, src)))
    if tokens & _STALE_TOKENS:
        return 'stale'
    if tokens & _DEBOUNCE_TOKENS or _has_abs_of_difference(node, src):
        return 'debounce'
    return 'range'


class SensorAnalyzer:
    """Matches sensor templates to classes and checks the code for corrupt / stale / debounce handling."""

    def __init__(self, parse_results: Dict[str, ParseResult], graph_builder: CallGraphBuilder,
                 templates: List[SensorTemplate]) -> None:
        self._results = parse_results
        self._gb = graph_builder
        self._templates = list(templates)
        self._parser = Parser(Language(tscpp.language()))
        self._funcs: Dict[str, FunctionDef] = {}
        self._methods: Dict[str, List[MethodInfo]] = {}      # class -> its methods
        self._consts: Dict[str, float] = {}
        self._facts: Dict[str, _Facts] = {}                  # sig_key -> comparisons etc., filled lazily
        self._families: List[Set[str]] = []                  # of the matched sensors

    def analyze(self) -> SensorAnalysisResult:
        if not self._gb.built:
            self._gb.build()
        self._funcs = {fn.sig_key: fn for r in self._results.values() for fn in r.functions}
        self._methods = {}
        for info in self._gb.hierarchy.methods.values():
            self._methods.setdefault(info.class_name, []).append(info)
        self._consts = self._collect_constants()
        self._families = []
        findings = [self._analyze_template(t) for t in self._templates]
        unmapped = self._unmapped(findings)
        logger.info(f"Sensor analysis: {sum(f.status == 'found' for f in findings)}/{len(findings)} "
                    f"templates matched a class, {len(unmapped)} unmapped sensor classes")
        return SensorAnalysisResult(findings, unmapped)

    # ── Template -> class ───────────────────────────────────────────────────

    def _analyze_template(self, tpl: SensorTemplate) -> SensorFinding:
        h = self._gb.hierarchy
        groups = [g for g in (tpl.parameter_group, tpl.io_group) if g]
        cls = h.resolve_class(tpl.class_name or tpl.name)
        if cls is None:
            return SensorFinding(template=tpl, class_name=None, status='not_found', hierarchy=groups)
        cdef = h.classes[cls]
        members: List[str] = []
        for c in [cls] + h.ancestors(cls):
            members += [f.name for f in h.classes[c].fields if f.name not in members]
        finding = SensorFinding(
            template=tpl, class_name=cls, status='found', file_path=cdef.file_path, line_number=cdef.line_number,
            instantiated=cls in self._gb.instantiated_classes,
            hierarchy=groups + [cls] + ([f"data: {', '.join(members)}"] if members else []),
            data_members=members)
        self._data_flow(finding)
        return finding

    # ── Data flow: where the value is obtained, processed, validated, consumed ──

    def _visible_methods(self, cls: str) -> Tuple[List[str], Set[str]]:
        """``(visible, hidden)``: defined methods of *cls*, own first then inherited ones it does not
        override, and the ancestors' methods its own (or a nearer ancestor's) overrides replace."""
        seen: Set[Tuple] = set()
        visible: List[str] = []
        hidden: Set[str] = set()
        for c in [cls] + self._gb.hierarchy.ancestors(cls):
            for info in self._methods.get(c, []):
                if info.short_name.startswith('~') or info.short_name == c.rsplit('::', 1)[-1]:
                    continue
                if _key(info) in seen:
                    hidden.add(info.sig_key)
                    continue
                seen.add(_key(info))
                if info.sig_key in self._funcs:
                    visible.append(info.sig_key)
        return visible, hidden

    def _sources(self, tpl: SensorTemplate, visible: List[str]) -> List[str]:
        sources = [k for k in visible if _READER_NAME.fullmatch(self._short(k))]
        if sources:
            return sources
        # A getter only stands in for a reader when it returns the sensor's data type.
        return [k for k in visible if _GETTER_NAME.fullmatch(self._short(k))
                and bare_type(self._funcs[k].return_type) == tpl.data_type]

    def _family(self, cls: str, sources: List[str]) -> Set[str]:
        """The topmost class(es) declaring the source methods, and all their descendants."""
        h = self._gb.hierarchy
        keys = {_key(h.methods[s]) for s in sources if s in h.methods}
        declarers = [c for c in [cls] + h.ancestors(cls)
                     if any(_key(i) in keys for i in self._methods.get(c, []))] or [cls]
        tops = [d for d in declarers if not set(h.ancestors(d)) & set(declarers)]
        return {c for t in tops for c in [t] + h.descendants(t)}

    def _entry(self, sig_key: str) -> str:
        fn = self._funcs[sig_key]
        return f'{sig_key} ({fn.file_path}:{fn.line_number})'

    def _data_flow(self, finding: SensorFinding) -> None:
        h = self._gb.hierarchy
        cls = finding.class_name or ''
        visible, hidden = self._visible_methods(cls)
        sources = self._sources(finding.template, visible)
        family = self._family(cls, sources)
        self._families.append(family)
        mine = {cls, *h.ancestors(cls)}
        # Methods of sibling (or derived) sensors and replaced ancestor methods are not this sensor's code.
        excluded = hidden | {k for k, info in h.methods.items() if info.class_name in family - mine}

        processors = self._unique(c for s in sources for c in sorted(self._funcs[s].called_by)
                                  if c in self._funcs and c not in sources and c not in excluded)
        callees = self._unique(c for p in processors for c in sorted(self._funcs[p].calls)
                               if c in self._funcs and c not in excluded)
        validators = [k for k in self._unique(callees + visible)
                      if k not in sources and _VALIDATOR_NAME.search(self._short(k))]
        family_methods = {k for k, info in h.methods.items() if info.class_name in family}
        consumers = [k for k in callees if k not in sources and k not in validators and k not in family_methods]

        finding.sources = [self._entry(k) for k in sources]
        finding.processors = [self._entry(k) for k in processors]
        finding.validators = [self._entry(k) for k in validators]
        finding.consumers = [self._entry(k) for k in consumers]
        scope = self._unique(visible + processors + validators)
        finding.checks = self._checks(finding, scope, {self._short(s) for s in sources})

    @staticmethod
    def _unique(items) -> List[str]:
        return list(dict.fromkeys(items))

    def _short(self, sig_key: str) -> str:
        return self._funcs[sig_key].name.rsplit('::', 1)[-1]

    # ── Checks ──────────────────────────────────────────────────────────────

    def _checks(self, finding: SensorFinding, scope: List[str], source_names: Set[str]) -> Dict[str, CheckResult]:
        tpl = finding.template
        hits = self._hits(scope, set(finding.data_members), source_names)
        bounds = [b for b in (tpl.valid_min, tpl.valid_max) if b is not None]
        low = _fmt(tpl.valid_min) if tpl.valid_min is not None else '-inf'
        high = _fmt(tpl.valid_max) if tpl.valid_max is not None else 'inf'
        stale, debounce = tpl.stale_threshold_ms, tpl.debounce_threshold

        if not tpl.corrupt_range_check:
            corrupt = CheckResult('not_required', 'range check disabled')
        elif not bounds:
            corrupt = CheckResult('not_configured', 'no valid_range configured')
        else:
            corrupt = self._check(hits['range'], bounds, f'range [{low}, {high}]')
        if stale is None:
            stale_check = CheckResult('not_configured', 'no stale threshold configured')
        else:
            stale_check = self._check(hits['stale'], [stale], f'threshold {_fmt(stale)} ms')
        if not tpl.debounce_enabled:
            debounce_check = CheckResult('not_required', 'debounce disabled')
        elif debounce is None:
            debounce_check = CheckResult('not_configured', 'no debounce threshold configured')
        else:
            debounce_check = self._check(hits['debounce'], [debounce], f'threshold {_fmt(debounce)}')
        return {'corrupt': corrupt, 'stale': stale_check, 'debounce': debounce_check}

    def _hits(self, scope: List[str], data: Set[str],
              source_names: Set[str]) -> Dict[str, List[Tuple[str, _Comparison]]]:
        """Comparisons per kind.  A range comparison counts only when its non-constant operand is
        sensor data: a data member, a parameter, a variable assigned from a source call, or a source
        call itself; a ``for`` condition never counts."""
        hits: Dict[str, List[Tuple[str, _Comparison]]] = {'range': [], 'stale': [], 'debounce': []}
        for sig_key in scope:
            facts = self._facts_of(sig_key)
            sensor_data = (data | {p.name for p in self._funcs[sig_key].parameters if p.name}
                           | {v for v, called in facts.assigned.items() if called & source_names})
            for cmp in facts.comparisons:
                if cmp.kind == 'range' and (
                        cmp.in_for_condition or not (cmp.operands & sensor_data or cmp.calls & source_names)):
                    continue
                hits[cmp.kind].append((sig_key, cmp))
        return hits

    def _check(self, hits: List[Tuple[str, _Comparison]], required: List[float], expected: str) -> CheckResult:
        found = sorted({v for _, c in hits for v in c.values})
        evidence = [f'{self._funcs[k].file_path}:{c.line} {k}: {c.text}' for k, c in hits]
        if not hits:
            status = 'missing'
        elif all(any(_close(f, r) for f in found) for r in required):
            status = 'implemented'
        else:
            status = 'mismatch'
        return CheckResult(status, expected, found, evidence)

    def _facts_of(self, sig_key: str) -> _Facts:
        if sig_key not in self._facts:
            self._facts[sig_key] = self._extract_facts(self._funcs[sig_key])
        return self._facts[sig_key]

    def _extract_facts(self, fn: FunctionDef) -> _Facts:
        """Relational comparisons in the body where an operand is a numeric constant, and which
        variables are assigned from which function calls."""
        if not fn.body:
            return _Facts([], {})
        src = b'void __f()' + fn.body.encode('utf-8')    # the body's '{' lands on row 0
        first_line = fn.body_start_line or fn.line_number
        tree = self._parser.parse(src)
        found: List[_Comparison] = []
        for node in find_nodes(tree.root_node, 'binary_expression'):
            operator = node.child_by_field_name('operator')
            if operator is None or get_text(operator, src) not in _RELATIONAL:
                continue
            sides = [node.child_by_field_name(side) for side in ('left', 'right')]
            values = [_value(s, src, self._consts) for s in sides]
            if all(v is None for v in values):
                continue
            variable = [s for s, v in zip(sides, values) if s is not None and v is None]
            found.append(_Comparison(
                line=first_line + node.start_point[0], text=' '.join(get_text(node, src).split()),
                kind=_classify(node, src), values=tuple(v for v in values if v is not None),
                operands=frozenset(i for s in variable for i in _identifiers(s, src)),
                calls=frozenset(n for s in variable for n in _called_names(s, src)),
                in_for_condition=_in_for_condition(node)))

        assigned: Dict[str, Set[str]] = {}

        def record(name: str, value: Optional[Node]) -> None:
            called = _called_names(value, src)
            if name and called:
                assigned.setdefault(name, set()).update(called)

        for node in find_nodes(tree.root_node, 'init_declarator'):
            declarator = node.child_by_field_name('declarator')
            names = _identifiers(declarator, src) if declarator is not None else []
            record(names[0] if names else '', node.child_by_field_name('value'))
        for node in find_nodes(tree.root_node, 'assignment_expression'):
            left = node.child_by_field_name('left')
            if left is not None and left.type in ('identifier', 'field_expression'):
                field_node = left.child_by_field_name('field') if left.type == 'field_expression' else left
                record(get_text(field_node, src) if field_node is not None else '', node.child_by_field_name('right'))
        return _Facts(list(dict.fromkeys(found)), assigned)

    # ── Named constants ─────────────────────────────────────────────────────

    def _collect_constants(self) -> Dict[str, float]:
        """File-scope ``const T NAME = <literal>;`` and ``#define NAME <literal>`` of every parsed file."""
        # shortcut: constants are keyed by bare name across all files, a repeated name keeps its last value;
        # upgrade to per-translation-unit scoping if two files define different constants of one name.
        consts: Dict[str, float] = {}
        for result in self._results.values():
            src = result.raw_source
            tree = self._parser.parse(src)
            for node in _file_scope(tree.root_node):
                if node.type == 'preproc_def':
                    name, text = node.child_by_field_name('name'), node.child_by_field_name('value')
                    if name is not None and text is not None:
                        value = self._define_value(_COMMENT.sub('', get_text(text, src)), consts)
                        if value is not None:
                            consts[get_text(name, src)] = value
                elif node.type == 'declaration' and any(
                        c.type == 'type_qualifier' and get_text(c, src) in ('const', 'constexpr')
                        for c in node.children):
                    for decl in node.children_by_field_name('declarator'):
                        ident = decl.child_by_field_name('declarator') if decl.type == 'init_declarator' else None
                        if ident is not None and ident.type == 'identifier':
                            value = _value(decl.child_by_field_name('value'), src, consts)
                            if value is not None:
                                consts[get_text(ident, src)] = value
        return consts

    def _define_value(self, text: str, consts: Dict[str, float]) -> Optional[float]:
        """Value of a macro body, read as the initializer of a variable."""
        src = b'int __v = ' + text.encode('utf-8') + b';'
        tree = self._parser.parse(src)
        decls = find_nodes(tree.root_node, 'init_declarator')
        return _value(decls[0].child_by_field_name('value'), src, consts) if decls else None

    # ── Unmapped sensor classes ─────────────────────────────────────────────

    def _is_abstract(self, cls: str) -> bool:
        """True if *cls* has a pure virtual method that nothing at or below it implements."""
        h = self._gb.hierarchy
        for c in [cls] + h.ancestors(cls):
            for info in self._methods.get(c, []):
                nearest = h.find_method(cls, info.short_name, info.param_types) if info.is_pure else None
                if nearest and h.methods[nearest].is_pure:
                    return True
        return False

    def _unmapped(self, findings: List[SensorFinding]) -> List[str]:
        classes = self._gb.hierarchy.classes
        mapped = {f.class_name for f in findings if f.class_name}
        candidates = set().union(*self._families) | {c for c in classes if c.rsplit('::', 1)[-1].endswith('Sensor')}
        return sorted(c for c in candidates - mapped if not self._is_abstract(c))
