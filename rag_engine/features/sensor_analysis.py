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
from rag_engine.features.sensor_validation import CHECKS, CheckRequirement, check_requirements
from rag_engine.models import FunctionDef, ParseResult

_READER_NAME = re.compile(r'(?i)read|sample|acquire')                        # fullmatch on a short name
_GETTER_NAME = re.compile(r'(?i)get\w*')
_VALIDATOR_NAME = re.compile(r'(?i)check|valid|range|stale|debounce|filter|limit')
_NAME_TOKEN = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')
_STALE_TOKENS = {'ms', 'msec', 'millis', 'us', 'usec', 'sec', 'secs', 'time', 'timestamp', 'stamp', 'elapsed',
                 'timeout', 'stale', 'tick', 'ticks', 'age', 'now'}
_DEBOUNCE_TOKENS = {'debounce', 'prev', 'previous', 'last', 'delta', 'diff', 'stable'}
_ABS_FUNCTIONS = {'abs', 'fabs', 'fabsf'}
# return types of a reader that make its class a sensor candidate
_NUMERIC = re.compile(r'(?:(?:unsigned|signed|long|short)\s+)*(?:float|double|int|long|short)|u?int(?:8|16|32|64)_t')
_RELATIONAL = {'<', '<=', '>', '>='}
# Scopes whose declarations are still file scope (include guards wrap whole headers).
_FILE_SCOPES = {'namespace_definition', 'declaration_list', 'linkage_specification',
                'preproc_if', 'preproc_ifdef', 'preproc_elif', 'preproc_else'}
_COMMENT = re.compile(r'//.*|/\*.*?\*/')


@dataclass
class CheckResult:
    """YAML expectation vs. C++ implementation for one check of one sensor.

    status (the verification result):
      implemented     the code compares the sensor data against exactly the configured limits
      mismatch        it compares, but not against (all of) the configured limits
      missing         the data path was traced and holds no such comparison
      inconclusive    static analysis cannot decide (limits not constant, no traceable data path, unresolved calls)
      disabled        switched off in the YAML: absence is not a failure
      not_configured  enabled but the YAML gives no value to check the code against"""
    status: Literal['implemented', 'mismatch', 'missing', 'inconclusive', 'disabled', 'not_configured']
    expected: str                 # e.g. "range [-54, 54]", "threshold 500 ms"
    found: List[float] = field(default_factory=list)       # constants of the matching comparisons
    evidence: List[str] = field(default_factory=list)      # "<file>:<line> <sig_key>: <comparison text>"
    configured: str = ''          # the YAML side: enabled / disabled / not_configured
    functions: List[str] = field(default_factory=list)     # functions holding the comparisons, "<sig_key> (<file>:<line>)"
    confidence: str = ''          # high / medium / low; '' when nothing was verified
    note: str = ''                # why this status, in a sentence
    recommendation: str = ''


@dataclass
class SensorFinding:
    template: SensorTemplate
    class_name: Optional[str]
    status: Literal['found', 'not_found', 'ambiguous']
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
    constructors: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)   # ambiguous: the competing classes or templates
    note: str = ''                                        # why not_found / ambiguous


@dataclass
class SensorAnalysisResult:
    findings: List[SensorFinding]
    unmapped_sensor_classes: List[str] = field(default_factory=list)   # discovered, no YAML configuration
    discovered_sensor_classes: List[str] = field(default_factory=list) # every sensor class found in the code
    limitations: List[str] = field(default_factory=list)

    @property
    def unused_configs(self) -> List[str]:
        """YAML configurations not (uniquely) mapped to a C++ class."""
        return [f.template.name for f in self.findings if f.status != 'found']

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            for c in f.checks.values():
                counts[c.status] = counts.get(c.status, 0) + 1
        return counts


_Call = Tuple[str, int, Optional[str]]   # (short callee name, absolute line, receiver) as CallSite names them


@dataclass(frozen=True)
class _Comparison:
    line: int
    text: str
    kind: Literal['range', 'stale', 'debounce']
    values: Tuple[float, ...]            # the numeric constants among the two operands
    operands: FrozenSet[str]             # identifiers of the non-constant operand
    calls: FrozenSet[_Call]              # functions called in the non-constant operand
    in_for_condition: bool


@dataclass(frozen=True)
class _ParamComparison:
    """A relational comparison in a function body that involves the function's parameters, e.g.
    ``v >= lo`` in ``inRange(float v, float lo, float hi)``: its constants come from each call site."""
    kind: Literal['range', 'stale', 'debounce']
    text: str
    sides: Tuple[FrozenSet[str], FrozenSet[str]]   # parameter names on the left and right operand
    has_constant: bool = False                     # an operand is already a constant: the callee counts itself


@dataclass
class _Facts:
    comparisons: List[_Comparison]
    assigned: Dict[str, Set[_Call]]      # variable -> functions called in the values assigned to it


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


def _call_of(call: Node, src: bytes, first_line: int) -> Optional[_Call]:
    """``s.read()`` -> ``('read', <line of s>, 's')``: the keys under which the call graph records the call site."""
    callee, name = call.child_by_field_name('function'), _callee_name(call, src)
    if callee is None or not name:
        return None
    start, receiver = callee, None
    if callee.type == 'field_expression':
        obj = callee.child_by_field_name('argument')
        while obj is not None and obj.type == 'subscript_expression':
            obj = obj.child_by_field_name('argument')
        if obj is not None and obj.type == 'field_expression':
            obj = obj.child_by_field_name('field')
        if obj is not None and obj.type in ('identifier', 'field_identifier'):
            start, receiver = obj, get_text(obj, src)
    return name, first_line + start.start_point[0], receiver


def _calls_in(node: Optional[Node], src: bytes, first_line: int) -> Set[_Call]:
    calls = find_nodes(node, 'call_expression') if node is not None else []
    return {c for c in (_call_of(n, src, first_line) for n in calls) if c}


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
        self._param_cmps: Dict[str, List[_ParamComparison]] = {}   # sig_key -> parametric comparisons, lazily
        self._families: List[Set[str]] = []                  # of the matched sensors
        self._sites: Dict[str, Dict[Tuple[int, Optional[str]], Set[str]]] = {}   # caller -> (line, receiver) -> targets

    def analyze(self) -> SensorAnalysisResult:
        if not self._gb.built:
            self._gb.build()
        self._funcs = {fn.sig_key: fn for r in self._results.values() for fn in r.functions}
        self._methods = {}
        for info in self._gb.hierarchy.methods.values():
            self._methods.setdefault(info.class_name, []).append(info)
        self._consts = self._collect_constants()
        self._families = []
        self._sites = {}
        for site in self._gb.call_sites:
            if site.call_type != 'unresolved':      # a fan-out over same-named functions resolves nothing
                self._sites.setdefault(site.caller, {}).setdefault((site.line, site.receiver), set()).add(site.target)
        mapping = {t.name: self._map(t) for t in self._templates}
        claims: Dict[str, List[str]] = {}
        for t in self._templates:
            cls = mapping[t.name][0]
            if cls:
                claims.setdefault(cls, []).append(t.name)
        findings = []
        for t in self._templates:
            cls, candidates = mapping[t.name]
            groups = [g for g in (t.parameter_group, t.io_group) if g]
            if cls and len(claims[cls]) > 1:
                others = [n for n in claims[cls] if n != t.name]
                findings.append(SensorFinding(
                    template=t, class_name=cls, status='ambiguous', hierarchy=groups, candidates=others,
                    note=f"class {cls} is also claimed by {', '.join(others)}; no configuration was chosen"))
            elif candidates:
                findings.append(SensorFinding(
                    template=t, class_name=None, status='ambiguous', hierarchy=groups, candidates=candidates,
                    note=f"'{t.class_name or t.name}' matches several classes: {', '.join(candidates)}; "
                         "set class_name to the qualified name"))
            elif cls is None:
                findings.append(SensorFinding(
                    template=t, class_name=None, status='not_found', hierarchy=groups,
                    note=f"no class named '{t.class_name or t.name}' in the parsed code"))
            else:
                findings.append(self._analyze_template(t, cls))
        unmapped = self._unmapped(findings)
        discovered = sorted({f.class_name for f in findings if f.class_name and f.status == 'found'} | set(unmapped))
        logger.info(f"Sensor analysis: {sum(f.status == 'found' for f in findings)}/{len(findings)} "
                    f"templates matched a class, {len(unmapped)} unmapped sensor classes")
        return SensorAnalysisResult(findings, unmapped, discovered, self._limitations(findings))

    def _map(self, tpl: SensorTemplate) -> Tuple[Optional[str], List[str]]:
        """(class the template names, competing classes when the name is ambiguous)."""
        h = self._gb.hierarchy
        wanted = tpl.class_name or tpl.name
        cls = h.resolve_class(wanted)
        if cls is not None:
            return cls, []
        candidates = sorted(c for c in h.classes if c == wanted or c.endswith('::' + wanted))
        return None, candidates if len(candidates) > 1 else []

    def _limitations(self, findings: List[SensorFinding]) -> List[str]:
        out = ['Static analysis only: a check counts when a comparison on the traced sensor data path uses '
               'the configured limit; timing, scheduling and hardware behaviour at run time are not verified.',
               'Shared validator functions are followed one call level deep; debounce confirmation counts '
               '(debounce.count) and frozen-value detection are not verified in the code.']
        for f in findings:
            if f.status == 'ambiguous':
                out.append(f'{f.template.name}: mapping unresolved — {f.note}.')
            elif f.status == 'found' and not f.sources:
                out.append(f'{f.template.name}: no acquisition function (read/sample/acquire, or a getter of '
                           f'type {f.template.data_type}) was identified, so its data path could not be traced.')
            for name, c in f.checks.items():
                if c.status == 'inconclusive':
                    out.append(f'{f.template.name}.{name}: {c.note}')
        return out

    # ── Template -> class ───────────────────────────────────────────────────

    def _analyze_template(self, tpl: SensorTemplate, cls: str) -> SensorFinding:
        h = self._gb.hierarchy
        groups = [g for g in (tpl.parameter_group, tpl.io_group) if g]
        cdef = h.classes[cls]
        members: List[str] = []
        for c in [cls] + h.ancestors(cls):
            members += [f.name for f in h.classes[c].fields if f.name not in members]
        finding = SensorFinding(
            template=tpl, class_name=cls, status='found', file_path=cdef.file_path, line_number=cdef.line_number,
            instantiated=cls in self._gb.instantiated_classes,
            hierarchy=groups + [cls] + ([f"data: {', '.join(members)}"] if members else []),
            data_members=members)
        short = cls.rsplit('::', 1)[-1]
        finding.constructors = [self._entry(k) if k in self._funcs else k for k, info in h.methods.items()
                                if info.class_name == cls and info.short_name == short]
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
        finding.checks = self._checks(finding, scope, sources)

    @staticmethod
    def _unique(items) -> List[str]:
        return list(dict.fromkeys(items))

    def _short(self, sig_key: str) -> str:
        return self._funcs[sig_key].name.rsplit('::', 1)[-1]

    # ── Checks ──────────────────────────────────────────────────────────────

    def _checks(self, finding: SensorFinding, scope: List[str], sources: List[str]) -> Dict[str, CheckResult]:
        """Compare each check the YAML asks for (generic engine) with what the code on the data path does."""
        hits = self._hits(finding, scope, sources)
        unresolved = [f'{s.caller} -> {s.target} ({s.file_path}:{s.line})' for s in self._gb.call_sites
                      if s.caller in scope and s.call_type == 'unresolved']
        kind = {'corrupt': 'range', 'stale': 'stale', 'debounce': 'debounce'}
        out: Dict[str, CheckResult] = {}
        for check in CHECKS:
            req = check_requirements(finding.template)[check]
            if req.state == 'disabled':
                out[check] = CheckResult('disabled', req.expected, configured='disabled',
                                         note='Disabled in the YAML configuration; its absence is not a failure.')
            elif req.state == 'not_configured':
                out[check] = CheckResult('not_configured', req.expected, configured='not_configured',
                                         note='The YAML enables this check but gives no limit to verify the code against.',
                                         recommendation=f'Add the approved {check} parameters to the YAML entry '
                                                        f'of {finding.template.name}, or disable the check.')
            else:
                out[check] = self._check(finding, req, hits[kind[check]], sources, unresolved)
        return out

    def _reaching(self, cls: str, sources: List[str]) -> Set[str]:
        """Call-site targets that run one of *sources* for *cls*: the sources themselves and the virtual
        methods of *cls* or its ancestors whose final overrider in *cls* is a source (a call through a base)."""
        h = self._gb.hierarchy
        mine = {cls, *h.ancestors(cls)}
        return set(sources) | {k for k, info in h.methods.items() if info.is_virtual and info.class_name in mine
                               and h.find_method(cls, info.short_name, info.param_types) in sources}

    def _hits(self, finding: SensorFinding, scope: List[str],
              sources: List[str]) -> Dict[str, List[Tuple[str, _Comparison, bool]]]:
        """Comparisons per kind that concern *this* sensor.  Sensor data is a variable assigned from a call
        site that resolves to one of the sensor's sources, a source call written in the comparison, or the
        receiver of such a call (``temp`` in ``temp.lastMs``).  A range comparison also counts on a parameter
        and, in the sensor's own methods, on a data member.  In the sensor's own methods every stale and
        debounce comparison counts; elsewhere it needs sensor data.  A ``for`` condition never counts.
        The flag says whether the link to the sensor data is data flow (True) or one of those name shortcuts."""
        h = self._gb.hierarchy
        cls = finding.class_name or ''
        mine = {cls, *h.ancestors(cls)}
        reach = self._reaching(cls, sources)
        data = set(finding.data_members)
        hits: Dict[str, List[Tuple[str, _Comparison, bool]]] = {'range': [], 'stale': [], 'debounce': []}
        for sig_key in scope:
            facts = self._facts_of(sig_key)
            sites = self._sites.get(sig_key, {})
            own = sig_key in h.methods and h.methods[sig_key].class_name in mine

            def from_source(call: _Call) -> bool:
                name, line, receiver = call
                return any(t in reach and t.split('(')[0].rsplit('::', 1)[-1] == name
                           for t in sites.get((line, receiver), ()))

            receivers = {r for (_, r), targets in sites.items() if r and targets & reach}
            sensor_data = receivers | {v for v, calls in facts.assigned.items() if any(map(from_source, calls))}
            for cmp in facts.comparisons:
                if cmp.in_for_condition:
                    continue
                refers = strong = bool(cmp.operands & sensor_data) or any(map(from_source, cmp.calls))
                if cmp.kind == 'range':
                    # shortcut: a range operand is a parameter or member by name, not by data flow from the caller.
                    refers = refers or bool(cmp.operands & ({p.name for p in self._funcs[sig_key].parameters if p.name}
                                                            | (data if own else set())))
                elif own:
                    refers = True
                # Without a constant a comparison only counts on a data-flow link: a name shortcut plus a
                # run-time limit says nothing about this sensor.
                if refers and (cmp.values or strong):
                    hits[cmp.kind].append((sig_key, cmp, strong))
        return hits

    def _check(self, finding: SensorFinding, req: CheckRequirement, hits: List[Tuple[str, _Comparison, bool]],
               sources: List[str], unresolved: List[str]) -> CheckResult:
        name = finding.template.name
        valued = [(k, c, strong) for k, c, strong in hits if c.values]
        found = sorted({v for _, c, _ in valued for v in c.values})
        evidence = [f'{self._funcs[k].file_path}:{c.line} {k}: {c.text}' for k, c, _ in hits]
        functions = [self._entry(k) for k in self._unique(k for k, _, _ in hits)]
        result = CheckResult('missing', req.expected, found, evidence, configured='enabled', functions=functions)
        label = {'corrupt': 'range', 'stale': 'stale-data', 'debounce': 'debounce'}[req.check]
        if not hits:
            if not sources:
                result.status, result.confidence = 'inconclusive', 'low'
                result.note = ('No acquisition function was identified for the class, so the sensor data path '
                               'could not be traced.')
                result.recommendation = 'Review manually, or give the class a read()/sample()/acquire() method.'
            elif unresolved:
                result.status, result.confidence = 'inconclusive', 'low'
                result.note = (f'No {label} comparison on the traced data path, but {len(unresolved)} call(s) on it '
                               f'could not be resolved statically ({unresolved[0]}).')
                result.recommendation = 'Review the unresolved call(s) manually.'
            else:
                result.confidence = 'medium'
                result.note = f'No {label} comparison of the sensor data was found on its traced data path.'
                result.recommendation = (f'Add a {label} check ({req.expected}) between the acquisition '
                                         f'({sources[0].split(" (")[0]}) and the consumers of {name}; a shared '
                                         f'generic validator called with this sensor\'s limits is sufficient.')
            return result
        if not valued:
            result.status, result.confidence = 'inconclusive', 'low'
            result.note = (f'{label.capitalize()} comparison(s) on the sensor data were found, but their limits are '
                           'not compile-time constants, so they cannot be compared with the YAML values.')
            result.recommendation = 'Use named constants for the limits, or confirm the run-time values manually.'
            return result
        result.confidence = 'high' if any(strong for _, _, strong in valued) else 'medium'
        absent = [v for v in req.values if not any(_close(f, v) for f in found)]
        if not absent:
            result.status = 'implemented'
            result.note = f'The code compares the sensor data against {req.expected}.'
        else:
            result.status = 'mismatch'
            result.note = (f"Configured {', '.join(_fmt(v) for v in absent)} not used by the code "
                           f"(code limits: {', '.join(_fmt(v) for v in found)}).")
            result.recommendation = ('Align the code limits with the approved configuration, or correct the YAML '
                                     'after review.')
        return result

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
            # Kept even without a constant: a comparison of the sensor data against a run-time limit makes
            # the check inconclusive (cannot be verified), not missing.
            variable = [s for s, v in zip(sides, values) if s is not None and v is None]
            found.append(_Comparison(
                line=first_line + node.start_point[0], text=' '.join(get_text(node, src).split()),
                kind=_classify(node, src), values=tuple(v for v in values if v is not None),
                operands=frozenset(i for s in variable for i in _identifiers(s, src)),
                calls=frozenset(c for s in variable for c in _calls_in(s, src, first_line)),
                in_for_condition=_in_for_condition(node)))

        found += self._generic_validator_calls(fn, tree.root_node, src, first_line)
        assigned: Dict[str, Set[_Call]] = {}

        def record(name: str, value: Optional[Node]) -> None:
            called = _calls_in(value, src, first_line)
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

    # ── Generic validators: one function, constants supplied per sensor at the call site ──

    def _parametric_comparisons(self, sig_key: str) -> List[_ParamComparison]:
        """Comparisons in *sig_key*'s body with a parameter as an operand; the function name decides the
        kind when it says one (``isStale`` -> stale), otherwise the comparison's own wording does."""
        if sig_key in self._param_cmps:
            return self._param_cmps[sig_key]
        fn = self._funcs[sig_key]
        params = {p.name for p in fn.parameters if p.name}
        out: List[_ParamComparison] = []
        if params and fn.body:
            src = b'void __f()' + fn.body.encode('utf-8')
            name_tokens = _tokens(fn.name.rsplit('::', 1)[-1])
            named = ('stale' if name_tokens & _STALE_TOKENS else
                     'debounce' if name_tokens & _DEBOUNCE_TOKENS else None)
            for node in find_nodes(self._parser.parse(src).root_node, 'binary_expression'):
                operator = node.child_by_field_name('operator')
                if operator is None or get_text(operator, src) not in _RELATIONAL:
                    continue
                operands = (node.child_by_field_name('left'), node.child_by_field_name('right'))
                sides = tuple(frozenset(_identifiers(n, src)) & params if n is not None else frozenset()
                              for n in operands)
                if sides[0] or sides[1]:
                    constant = any(_value(n, src, self._consts) is not None for n in operands)
                    out.append(_ParamComparison(named or _classify(node, src), ' '.join(get_text(node, src).split()),
                                                sides, constant))   # type: ignore[arg-type]
        self._param_cmps[sig_key] = out
        return out

    def _generic_validator_calls(self, fn: FunctionDef, root: Node, src: bytes, first_line: int) -> List[_Comparison]:
        """Calls in *fn* to a function that compares its parameters, with a constant argument bound to one of
        them: ``inRange(raw, AIRSPEED_MIN, AIRSPEED_MAX)`` is the comparisons ``raw >= 0`` and ``raw <= 450``
        at the call site.  One validator written once then serves every sensor that passes its own limits."""
        # shortcut: one call level; a validator that forwards its limits to another validator is not followed.
        sites = self._sites.get(fn.sig_key, {})
        out: List[_Comparison] = []
        for call in find_nodes(root, 'call_expression'):
            key = _call_of(call, src, first_line)
            args_node = call.child_by_field_name('arguments')
            if key is None or args_node is None:
                continue
            name, line, receiver = key
            targets = [t for t in sorted(sites.get((line, receiver), ())) if t in self._funcs
                       and self._funcs[t].name.rsplit('::', 1)[-1] == name]
            args = [a for a in args_node.named_children if a.type != 'comment']
            for target in targets[:1]:
                params = [p.name for p in self._funcs[target].parameters]
                bound = {p: a for p, a in zip(params, args) if p}
                for pc in self._parametric_comparisons(target):
                    values: List[float] = []
                    operands: Set[str] = set()
                    calls: Set[_Call] = set()
                    for side in pc.sides:
                        for param in side:
                            arg = bound.get(param)
                            if arg is None:
                                continue
                            value = _value(arg, src, self._consts)
                            if value is not None:
                                values.append(value)
                            else:
                                operands.update(_identifiers(arg, src))
                                calls |= _calls_in(arg, src, first_line)
                    if values or ((operands or calls) and not pc.has_constant):   # no constant: inconclusive evidence
                        out.append(_Comparison(
                            line=first_line + call.start_point[0],
                            text=f"{' '.join(get_text(call, src).split())} -> {pc.text}",
                            kind=pc.kind, values=tuple(values), operands=frozenset(operands),
                            calls=frozenset(calls), in_for_condition=_in_for_condition(call)))
        return out

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
                if info.short_name.startswith('~'):      # a pure destructor still gets a body
                    continue
                nearest = h.find_method(cls, info.short_name, info.param_types) if info.is_pure else None
                if nearest and h.methods[nearest].is_pure:
                    return True
        return False

    def _unmapped(self, findings: List[SensorFinding]) -> List[str]:
        classes = self._gb.hierarchy.classes
        mapped = {f.class_name for f in findings if f.class_name}
        named = {c for c in classes if c.rsplit('::', 1)[-1].endswith('Sensor')}
        derived = {d for c in classes if 'sensor' in c.rsplit('::', 1)[-1].lower() for d in self._gb.hierarchy.descendants(c)}
        readers = {self._gb.hierarchy.methods[k].class_name for k, fn in self._funcs.items()
                   if k in self._gb.hierarchy.methods and _READER_NAME.fullmatch(fn.name.rsplit('::', 1)[-1])
                   and _NUMERIC.fullmatch(bare_type(fn.return_type) or '')}
        candidates = set().union(*self._families) | named | derived | readers
        return sorted(c for c in candidates - mapped if not self._is_abstract(c))
