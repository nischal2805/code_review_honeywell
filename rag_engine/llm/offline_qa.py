"""Offline Q&A: answers questions from the structured analysis results, without an LLM.

The deterministic facts come first; when a local LLM is available it only explains them.
"""
from __future__ import annotations
import difflib
import re
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Set, Tuple

from rag_engine.features.function_effect import FunctionEffectAnalyzer, format_effect
from rag_engine.retrieval.retriever import best_sentence

if TYPE_CHECKING:
    from rag_engine.features.sensor_analysis import SensorFinding
    from rag_engine.features.virtual_analysis import VirtualAnalysisResult
    from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
    from rag_engine.pipeline import AnalysisResults

_NAME = r'([A-Za-z_~][\w:~]*)'
_THE = r'(?:the\s+|class\s+|method\s+|function\s+|sensor\s+)*'
_FLAGS = re.IGNORECASE   # the captured name keeps the question's case

_VIRTUALS_BY = [
    re.compile(rf'\bvirtual\b.*?\b(?:implemented|overridden|overrid\w+|defined|declared)\s+by\s+{_THE}{_NAME}', _FLAGS),
    re.compile(rf'\bvirtual\b.*?\bdoes\s+{_THE}{_NAME}\s+(?:implement|override|define)', _FLAGS),
]
_EFFECT_OF = [
    re.compile(rf'\b(?:affected|impacted|broken)\b.*?\bif\s+{_THE}{_NAME}(?:\s*\(\s*\))?', _FLAGS),
    re.compile(rf'\b(?:effects?|impact|coupling|side[- ]effects?)\s+of\s+{_THE}{_NAME}', _FLAGS),
]
_CHECKS_OF = re.compile(
    rf'\b(?:checks?|validations?|validating|validated|validation\s+checks?)\b.*?\b(?:to|for|on|of|by)\s+{_THE}{_NAME}', _FLAGS)
_HIERARCHY_OF = re.compile(
    rf'\b(?:inheritance|hierarchy|derived|subclass\w*|base\s+class\w*)\b.*?\b(?:of|for|from|by)\s+{_THE}{_NAME}', _FLAGS)

_HELP = ("I can answer questions about your analysis results.\nTry:\n"
         "  'how many critical violations'\n"
         "  'which functions need reverification'\n"
         "  'what dead code should be removed'\n"
         "  'which LRU has highest risk'\n"
         "  'what is the compliance score'\n"
         "  'what are the virtual functions implemented by <Class>'\n"
         "  'what components are affected if <method> changes'\n"
         "  'what validation checks are applied to <Sensor>'\n"
         "  'is the stale check implemented for <Sensor>' / 'why is <Sensor> inconclusive'\n"
         "  'which sensors have missing checks' / 'what is the data flow of <Sensor>'\n"
         "  'what is the inheritance of <Class>'\n"
         "  'what is the maximum function length' / 'is recursion allowed' / 'which rules come from the document'\n"
         "  'what does the standard say about <topic>'")


_NO_FACTS_LABEL = "Explanation (local LLM — no specific static-analysis facts matched; summary facts used)"


def function_names(results: AnalysisResults) -> Set[str]:
    """Short names of every parsed function: what a method name in a question or command can match."""
    return {fn.name.rsplit('::', 1)[-1] for r in results.parse_results.values() for fn in r.functions}


def not_found_message(kind: str, name: str, candidates: Iterable[str]) -> str:
    candidates = sorted(set(candidates))
    close = difflib.get_close_matches(name, candidates, n=5, cutoff=0.5)
    close += [c for c in candidates if name.lower() in c.lower() and c not in close][:5 - len(close)]
    if close:
        return f"{kind} '{name}' not found. Close names: {', '.join(close)}"
    shown = ', '.join(candidates[:10]) + (', ...' if len(candidates) > 10 else '')
    return f"{kind} '{name}' not found. Known: {shown or 'none'}"


class _NotFound(str):
    """A not-found message. *likely_typo*: some known name is nearly the one asked for."""
    likely_typo = False


def _not_found(kind: str, name: str, candidates: Iterable[str]) -> _NotFound:
    candidates = set(candidates)
    message = _NotFound(not_found_message(kind, name, candidates))
    message.likely_typo = bool(difflib.get_close_matches(name, candidates, n=1, cutoff=0.8))
    return message


def _first_match(patterns: Iterable[re.Pattern], question: str) -> Optional[str]:
    for pattern in patterns:
        m = pattern.search(question)
        if m:
            return m.group(1)
    return None


def _find_class(results: AnalysisResults, name: str) -> Optional[str]:
    h = results.graph.hierarchy
    found = h.resolve_class(name)
    if found:
        return found
    lowered = {c.lower(): c for c in h.classes}
    return lowered.get(name.lower()) or next(
        (c for low, c in lowered.items() if low.endswith('::' + name.lower())), None)


# ── Handlers: each returns an answer, or None when the question is not about its topic ──

def _virtuals_of_class(q: str, results: AnalysisResults) -> Optional[str]:
    name = _first_match(_VIRTUALS_BY, q)
    if name is None or results.graph is None or results.virtual is None:
        return None
    cls = _find_class(results, name)
    if cls is None:
        return _not_found('Class', name, results.graph.hierarchy.classes)
    h = results.graph.hierarchy
    base_of = {o.function: (v, o) for v in results.virtual.inventory for o in v.overridden_by}
    roots = {v.function: v for v in results.virtual.inventory}
    lines = []
    for info in h.methods.values():
        if info.class_name != cls or not info.is_virtual:
            continue
        if info.sig_key in base_of:
            base, over = base_of[info.sig_key]
            lines.append(f"  {info.sig_key} — overrides {base.function} "
                         f"(reachable: {'yes' if over.reachable else 'no'})")
        elif info.sig_key in roots:
            overs = ', '.join(o.function for o in roots[info.sig_key].overridden_by) or 'none'
            lines.append(f"  {info.sig_key} — introduces {'pure ' if info.is_pure else ''}virtual function; "
                         f"overridden by: {overs}")
        else:
            lines.append(f"  {info.sig_key} — virtual")
    if not lines:
        return f"{cls} declares or overrides no virtual functions."
    return f"Virtual functions implemented by {cls}:\n" + '\n'.join(lines)


def _effect_of_method(q: str, results: AnalysisResults) -> Optional[str]:
    name = _first_match(_EFFECT_OF, q)
    if name is None or results.graph is None or not results.parse_results:
        return None
    effects = FunctionEffectAnalyzer(results.parse_results, results.graph).analyze(name)
    if not effects:
        return _not_found('Function', name, function_names(results))
    return '\n\n'.join(format_effect(e) for e in effects)


def _check_line(name: str, c) -> str:
    found = f"; code limits {', '.join(f'{v:g}' for v in c.found)}" if c.found else ''
    conf = f", confidence {c.confidence}" if c.confidence else ''
    return f"  {name}: {c.status}{conf} (YAML: {c.expected}{found})"


def sensor_check_text(finding: SensorFinding) -> str:
    """Mapping, configuration, check results and data flow of one sensor (R3), as plain text."""
    tpl = finding.template
    if finding.status != 'found':
        return (f"Sensor {tpl.name}: mapping {finding.status.replace('_', ' ')} — {finding.note or 'no code class found'}."
                " No check could be verified.")
    lines = [f"Sensor {tpl.name} (category {tpl.category or 'not given'}; class {finding.class_name}, "
             f"{finding.file_path}:{finding.line_number})",
             f"Hierarchy: {' -> '.join(finding.hierarchy)}",
             "Checks (YAML expectation vs. C++ implementation, static analysis):"]
    for name, c in finding.checks.items():
        lines.append(_check_line(name, c))
        lines += [f"    {e}" for e in c.evidence]
        if c.status not in ('implemented',) and c.note:
            lines.append(f"    Why: {c.note}")
    for title, items in (('Sources', finding.sources), ('Processors', finding.processors),
                         ('Validators', finding.validators), ('Consumers', finding.consumers)):
        lines.append(f"{title}: {', '.join(items) or 'none'}")
    return '\n'.join(lines)


_SENSOR_INTENTS = [   # (keywords, renderer) in priority order; the first whose keyword occurs wins
    (('why', 'inconclusive', 'explain'), lambda f, r: _sensor_why(f)),
    (('improve', 'recommend', 'fix', 'should be done', 'what should'), lambda f, r: _sensor_recommendations(f)),
    (('replay', 'csv', 'sample'), lambda f, r: _sensor_replay(f, r)),
    (('data flow', 'data-flow', 'dataflow', 'flow', 'path'), lambda f, r: _sensor_flow(f)),
    (('match', 'consistent', 'differ', 'agree'), lambda f, r: _sensor_match(f)),
    (('stale',), lambda f, r: _sensor_check(f, 'stale')),
    (('debounce',), lambda f, r: _sensor_check(f, 'debounce')),
    (('implemented', 'range check', 'corrupt'), lambda f, r: _sensor_check(f, 'corrupt')),
    (('checks', 'checked', 'applied'), lambda f, r: sensor_check_text(f)),
    (('valid range', 'range', 'limit', 'threshold'), lambda f, r: _sensor_config(f)),
    (('yaml', 'configuration', 'config', 'mapped', 'mapping'), lambda f, r: _sensor_config(f)),
    (('category', 'kind of', 'type of'), lambda f, r: f"{f.template.name}: category {f.template.category or 'not given in the YAML'}."),
    (('acquire', 'acquisition', 'acquired', 'read', 'source', 'obtained'), lambda f, r: _sensor_list(f, 'Acquisition (sources)', f.sources)),
    (('process',), lambda f, r: _sensor_list(f, 'Processing functions', f.processors)),
    (('validation function', 'validator', 'validates', 'validation'), lambda f, r: _sensor_list(f, 'Validation functions', f.validators)),
    (('consume', 'consumer', 'used by'), lambda f, r: _sensor_list(f, 'Consumers', f.consumers)),
]


def _unmapped_text(f: SensorFinding) -> Optional[str]:
    if f.status == 'found':
        return None
    return f"{f.template.name}: mapping {f.status.replace('_', ' ')} — {f.note}. This could not be established."


def _sensor_list(f: SensorFinding, title: str, items: List[str]) -> str:
    return _unmapped_text(f) or (f"{title} of {f.template.name}: " + (', '.join(items) if items else
                                 'none identified by the static analysis'))


def _sensor_flow(f: SensorFinding) -> str:
    if _unmapped_text(f):
        return _unmapped_text(f)  # type: ignore[return-value]
    return '\n'.join([f"Data flow of {f.template.name} (static analysis):"] + [
        f"  {title}: {', '.join(items) or 'none identified'}" for title, items in (
            ('Acquisition', f.sources), ('Processing', f.processors),
            ('Validation', f.validators), ('Consumers', f.consumers))])


def _sensor_config(f: SensorFinding) -> str:
    t = f.template
    rng = ('not configured' if t.valid_min is None and t.valid_max is None
           else f"[{'-inf' if t.valid_min is None else f'{t.valid_min:g}'}, {'inf' if t.valid_max is None else f'{t.valid_max:g}'}]")
    mapping = (f"mapped to C++ class {f.class_name}" if f.status == 'found'
               else f"not mapped ({f.status.replace('_', ' ')}: {f.note})")
    deb_thr = 'not configured' if t.debounce_threshold is None else f'{t.debounce_threshold:g}'
    debounce = 'off' if not t.debounce_enabled else f'threshold {deb_thr}, count {t.debounce_count}'
    return (f"YAML configuration {t.name}: {mapping}.\n"
            f"  category {t.category or 'not given'}, unit {t.unit or 'not given'}\n"
            f"  valid range {rng} (range check {'on' if t.corrupt_range_check else 'off'})\n"
            f"  stale threshold {'not configured' if t.stale_threshold_ms is None else f'{t.stale_threshold_ms:g} ms'}"
            f" (frozen-value detection {'on' if t.stale_detect_frozen else 'off'})\n"
            f"  debounce {debounce}")


def _sensor_check(f: SensorFinding, check: str) -> str:
    if _unmapped_text(f):
        return _unmapped_text(f)  # type: ignore[return-value]
    c = f.checks[check]
    where = f"; detected in {', '.join(c.functions)}" if c.functions else ''
    lines = [f"{f.template.name} {check} check: {c.status}"
             + (f" (confidence {c.confidence})" if c.confidence else '') + f". YAML: {c.expected}{where}."]
    if c.found:
        lines.append(f"  Code limits: {', '.join(f'{v:g}' for v in c.found)}")
    lines += [f"  Evidence: {e}" for e in c.evidence[:4]]
    if c.note:
        lines.append(f"  {c.note}")
    if c.recommendation:
        lines.append(f"  Recommendation: {c.recommendation}")
    return '\n'.join(lines)


def _sensor_match(f: SensorFinding) -> str:
    if _unmapped_text(f):
        return _unmapped_text(f)  # type: ignore[return-value]
    bad = [n for n, c in f.checks.items() if c.status in ('mismatch', 'missing', 'inconclusive')]
    verdict = ('The C++ implementation matches the YAML for every enabled check.' if not bad else
               f"Not consistent for: {', '.join(f'{n} ({f.checks[n].status})' for n in bad)}.")
    return '\n'.join([f"{f.template.name}: {verdict}"] + [_check_line(n, c) for n, c in f.checks.items()])


def _sensor_why(f: SensorFinding) -> str:
    if _unmapped_text(f):
        return _unmapped_text(f)  # type: ignore[return-value]
    lines = [f"{f.template.name}: {n} is {c.status} — {c.note}" for n, c in f.checks.items()
             if c.status != 'implemented' and c.note]
    return '\n'.join(lines) or f"{f.template.name}: every enabled check is implemented."


def _sensor_recommendations(f: SensorFinding) -> str:
    if f.status != 'found':
        return f"{f.template.name}: {f.note}. Recommendation: set class_name to the intended C++ class."
    recs = [f"  {n} ({c.status}): {c.recommendation}" for n, c in f.checks.items() if c.recommendation]
    return (f"Recommended changes for {f.template.name}:\n" + '\n'.join(recs)) if recs else \
        f"{f.template.name}: no change is recommended by the analysis."


def _sensor_replay(f: SensorFinding, results: AnalysisResults) -> str:
    replay = getattr(results, 'sensor_replay', None)
    if replay is None:
        return "No CSV replay was run (no sensor_samples_file configured)."
    r = replay.sensors.get(f.template.name)
    if r is None:
        return f"The CSV {replay.source} has no samples for {f.template.name}."
    flagged = [x for x in r.results if x.flags][:8]
    lines = [f"CSV replay of {f.template.name} ({replay.source}): {r.counts['total']} samples, corrupt {r.counts['corrupt']}, "
             f"stale {r.counts['stale']}, frozen {r.counts['frozen']}, debounce {r.counts['debounce']}."]
    lines += [f"  t={x.timestamp_ms:g} ms value={x.value:g}: {', '.join(x.flags)}" for x in flagged]
    lines.append("  (How the configured rules treat these samples; not proof of the C++ runtime behaviour.)")
    return '\n'.join(lines)


def _mentioned_sensor(q: str, results: AnalysisResults) -> Optional[SensorFinding]:
    best: Tuple[int, Optional[SensorFinding]] = (0, None)
    for f in results.sensors.findings:  # type: ignore[union-attr]
        for name in {f.template.name, f.template.class_name, f.class_name} - {None, ''}:
            short = str(name).rsplit('::', 1)[-1]
            if len(short) > best[0] and re.search(rf'\b{re.escape(short)}\b', q, re.IGNORECASE):
                best = (len(short), f)
    return best[1]


def _project_sensor_answer(q: str, results: AnalysisResults) -> Optional[str]:
    sensors = results.sensors
    findings = sensors.findings  # type: ignore[union-attr]

    def listing(statuses, title):
        rows = [f"  {f.template.name}.{n}: {c.status} — {c.note}" for f in findings
                for n, c in f.checks.items() if c.status in statuses]
        return f"{title}: {len(rows)}\n" + ('\n'.join(rows) or '  none')
    if 'inconclusive' in q:
        return listing(('inconclusive',), 'Inconclusive sensor checks')
    if 'mismatch' in q or 'do not match' in q or "don't match" in q:
        return listing(('mismatch',), 'Configuration-to-code mismatches')
    if 'missing' in q:
        return listing(('missing',), 'Missing sensor checks')
    if any(k in q for k in ('unmapped', 'without', 'no yaml', 'no configuration', 'unused', 'ambiguous')):
        unused = [f"  {f.template.name}: {f.status.replace('_', ' ')} — {f.note}" for f in findings if f.status != 'found']
        return ("Sensor classes without a YAML configuration: "
                + (', '.join(sensors.unmapped_sensor_classes) or 'none')  # type: ignore[union-attr]
                + "\nYAML configurations not mapped to a class:\n" + ('\n'.join(unused) or '  none'))
    if any(k in q for k in ('replay', 'csv', 'samples')):
        replay = getattr(results, 'sensor_replay', None)
        if replay is None:
            return "No CSV replay was run (no sensor_samples_file configured)."
        return '\n'.join([f"CSV replay ({replay.source}):"] + [
            f"  {n}: {r.counts['total']} samples, corrupt {r.counts['corrupt']}, stale {r.counts['stale']}, "
            f"frozen {r.counts['frozen']}, debounce {r.counts['debounce']}" for n, r in replay.sensors.items()]
            + ([f"  No configuration for: {', '.join(replay.unknown_sensors)}"] if replay.unknown_sensors else []))
    if any(k in q for k in ('discovered', 'found', 'detected', 'list', 'which sensors', 'what sensors', 'how many sensors')):
        rows = [f"  {f.template.name} ({f.template.category or 'no category'}): "
                + (f"class {f.class_name}" if f.status == 'found' else f"{f.status.replace('_', ' ')}")
                for f in findings]
        extra = sensors.unmapped_sensor_classes  # type: ignore[union-attr]
        return (f"Sensor classes discovered in the code: {len(sensors.discovered_sensor_classes)}; "  # type: ignore[union-attr]
                f"YAML configurations: {len(findings)}\n" + '\n'.join(rows)
                + (f"\n  Without YAML configuration: {', '.join(extra)}" if extra else ''))
    return None


def _sensor_question(q: str, results: AnalysisResults) -> Optional[str]:
    """Sensor questions answered from the structured sensor analysis (and CSV replay) only."""
    if results.sensors is None:
        return None
    f = _mentioned_sensor(q, results)
    ql = q.lower()
    if f is None:
        return _project_sensor_answer(ql, results) if 'sensor' in ql else None
    for keywords, render in _SENSOR_INTENTS:
        if any(k in ql for k in keywords):
            return render(f, results)
    return sensor_check_text(f)


def _checks_of_sensor(q: str, results: AnalysisResults) -> Optional[str]:
    name = _first_match([_CHECKS_OF], q)
    if name is None:
        return None
    if results.sensors is None:
        if results.graph is not None and _find_class(results, name):
            return "Sensor analysis was not run: no sensor templates are configured (sensor_templates_file)."
        return None
    wanted = name.lower()
    for f in results.sensors.findings:
        if wanted in (f.template.name.lower(), (f.class_name or '').lower(), (f.template.class_name or '').lower()):
            return sensor_check_text(f)
    return _not_found('Sensor', name, [f.template.name for f in results.sensors.findings])


def _hierarchy_of_class(q: str, results: AnalysisResults) -> Optional[str]:
    name = _first_match([_HIERARCHY_OF], q)
    if name is None or results.graph is None:
        return None
    cls = _find_class(results, name)
    if cls is None:
        return _not_found('Class', name, results.graph.hierarchy.classes)
    h = results.graph.hierarchy
    bases, derived = h.ancestors(cls), h.descendants(cls)
    roots = [b for b in bases if not h.bases(b)] or [cls]
    return '\n'.join([f"Inheritance of {cls}:",
                      f"  Bases (nearest first): {', '.join(bases) or 'none'}",
                      f"  Derived: {', '.join(derived) or 'none'}",
                      *(h.inheritance_tree(root) for root in roots)])


_STRUCTURAL: List[Callable[[str, AnalysisResults], Optional[str]]] = [
    _virtuals_of_class, _effect_of_method, _hierarchy_of_class, _sensor_question, _checks_of_sensor]


def _structural_answer(question: str, results: AnalysisResults) -> Tuple[Optional[str], Optional[_NotFound]]:
    """(answer of the first handler whose name resolved, the most telling not-found message of the others)."""
    missing: Optional[_NotFound] = None
    for handler in _STRUCTURAL:
        answer = handler(question, results)
        if isinstance(answer, _NotFound):
            if missing is None or (answer.likely_typo and not missing.likely_typo):
                missing = answer
        elif answer is not None:
            return answer, None
    return None, missing


def _virtual_inventory_line(virtual: VirtualAnalysisResult) -> str:
    overrides = [o for info in virtual.inventory for o in info.overridden_by]
    return (f"Virtual functions: {len(virtual.inventory)} declared, {len(overrides)} override(s), "
            f"{sum(not o.reachable for o in overrides)} not reachable from the entry points")


def _summary_answer(question: str, results: AnalysisResults) -> Optional[str]:
    """Counts and lists from the analysis summaries, plus general DO-178C answers."""
    q = question.lower()

    # Standards
    if results.standards:
        s = results.standards
        if any(k in q for k in ("critical", "critical violation", "most serious")):
            n = s.violations_by_severity.get("CRITICAL", 0)  # type: ignore[union-attr]
            items = [v for v in s.violations if v.severity == "CRITICAL"]  # type: ignore[union-attr]
            lines = [f"  {v.rule} in {v.file.split('/')[-1].split(chr(92))[-1]}:{v.line} — {v.message}" for v in items[:5]]
            return f"{n} CRITICAL violation(s):\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("score", "compliance", "percent")):
            return f"Compliance score: {s.compliance_score:.1f}%"
        if any(k in q for k in ("violation", "major", "medium", "minor")):
            bysev = s.violations_by_severity  # type: ignore[union-attr]
            return (f"Violations by severity:\n"
                    f"  CRITICAL: {bysev.get('CRITICAL',0)}\n"
                    f"  MAJOR:    {bysev.get('MAJOR',0)}\n"
                    f"  MEDIUM:   {bysev.get('MEDIUM',0)}\n"
                    f"  MINOR:    {bysev.get('MINOR',0)}\n"
                    f"  Total:    {len(s.violations)}")
        if "goto" in q:
            gotos = [v for v in s.violations if "GOTO" in v.rule]  # type: ignore[union-attr]
            if gotos:
                return "\n".join(f"  {v.file.split('/')[-1]}:{v.line} — {v.element}" for v in gotos)
            return "No goto violations found."

    # Virtual analysis
    if results.virtual:
        v = results.virtual
        if any(k in q for k in ("reverif", "category 2", "cat2", "cat 2")):
            cat2 = [c for c in v.changes if c.do178c_category == "Category 2"]
            lines = [f"  {c.function.name} ({c.change_type}) — {c.reverification_scope or 'full reverification'}" for c in cat2[:8]]
            return f"{len(cat2)} Category 2 change(s) requiring reverification:\n" + "\n".join(lines)
        if any(k in q for k in ("changed", "modified", "virtual", "change")):
            s = v.summary
            inventory = f"{_virtual_inventory_line(v)}\n" if v.inventory else ""
            return (inventory + f"Virtual function changes:\n"
                    f"  Added:     {s.get('added', 0)}\n"
                    f"  Removed:   {s.get('removed', 0)}\n"
                    f"  Modified:  {s.get('modified', 0)}\n"
                    f"  Unchanged: {s.get('unchanged', 0)}")

    # Dead code
    if results.dead:
        d = results.dead
        if any(k in q for k in ("remove", "dead code", "dead")):
            items = [i for i in d.items if i.category == "dead_code"]
            lines = [f"  {i.name} in {i.file_path.split('/')[-1].split(chr(92))[-1]}:{i.line_number}" for i in items[:8]]
            return f"{d.dead_count} dead function(s) to remove:\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("deactivated", "justify")):
            items = [i for i in d.items if i.category == "deactivated_code"]
            lines = [f"  {i.name} — {i.coverage_impact}" for i in items[:6]]
            return f"{d.deactivated_count} deactivated item(s):\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("total dead", "how much", "how many dead")):
            return f"Dead: {d.dead_count}  Deactivated: {d.deactivated_count}  Total: {d.dead_count + d.deactivated_count}"

    # Coupling
    if results.coupling:
        c = results.coupling
        if any(k in q for k in ("high risk", "highest risk", "most risk")):
            high = [(name, lru) for name, lru in c.lru_impacts.items() if lru.risk_level == "high"]
            if high:
                return "High-risk LRUs:\n" + "\n".join(f"  {n}: {len(l.control_coupling)} ctrl + {len(l.data_coupling)} data coupling items" for n, l in high)
            return "No high-risk LRUs found."
        if any(k in q for k in ("coupling", "lru", "risk")):
            lines = [f"  {n}: {l.risk_level.upper()} ({len(l.control_coupling)} ctrl, {len(l.data_coupling)} data)" for n, l in c.lru_impacts.items()]
            return "LRU coupling summary:\n" + ("\n".join(lines) or "  (no LRU documents provided)")

    # General DO-178C knowledge
    if "category 1" in q:
        return ("Category 1: comment or formatting change only.\n"
                "Requires regression tests only — no full reverification.")
    if "category 2" in q:
        return ("Category 2: signature, implementation, or existence change.\n"
                "Requires full reverification of affected functions and their callers.")
    if "dal" in q:
        return ("DAL levels:\n"
                "  A — Catastrophic failure. MC/DC coverage required.\n"
                "  B — Hazardous failure. Decision coverage required.\n"
                "  C — Major failure. Statement coverage required.\n"
                "  D — Minor failure. Basic coverage required.")
    if any(k in q for k in ("ready", "certif", "pass")):
        if results.standards:
            crit = results.standards.violations_by_severity.get("CRITICAL", 0)
            score = results.standards.compliance_score
            if crit == 0 and score >= 90:
                return f"Compliance score {score:.1f}%, 0 CRITICAL violations. Looks good for certification review."
            return (f"Compliance score {score:.1f}%, {crit} CRITICAL violation(s).\n"
                    f"Fix CRITICAL violations before submitting to DER.")
        return "Run analysis first to assess certification readiness."

    return None


def results_summary(results: AnalysisResults) -> str:
    """A few lines of headline numbers from every analysis that ran."""
    lines = ["Analysis summary:"]
    if results.standards:
        s, sev = results.standards, results.standards.violations_by_severity
        lines.append(f"  Standards: compliance {s.compliance_score:.1f}%, CRITICAL={sev.get('CRITICAL', 0)}, "
                     f"MAJOR={sev.get('MAJOR', 0)}, total violations={len(s.violations)}")
    if results.virtual:
        v = results.virtual
        lines.append(f"  {_virtual_inventory_line(v)}; changes: "
                     f"added={v.summary.get('added', 0)}, removed={v.summary.get('removed', 0)}, "
                     f"modified={v.summary.get('modified', 0)}, unchanged={v.summary.get('unchanged', 0)}")
    if results.dead:
        lines.append(f"  Dead code: dead={results.dead.dead_count}, deactivated={results.dead.deactivated_count}")
    if results.coupling:
        risks = ', '.join(f"{n}: {lru.risk_level}" for n, lru in results.coupling.lru_impacts.items())
        lines.append(f"  Coupling risk by LRU: {risks or 'no LRU documents'}")
    if results.sensors:
        findings = results.sensors.findings
        checks = Counter(c.status for f in findings for c in f.checks.values())
        lines.append(f"  Sensors: {sum(f.status == 'found' for f in findings)}/{len(findings)} templates mapped; "
                     f"checks: {', '.join(f'{k}={n}' for k, n in sorted(checks.items())) or 'none'}")
        replay = getattr(results, 'sensor_replay', None)
        if replay is not None:
            lines.append(f"  Sensor CSV replay: {len(replay.sensors)} sensor(s) replayed from {replay.source}")
    return '\n'.join(lines)


_RULE_WORDS = [   # (pattern, rule key) — which rule a standards question is about
    (r'cyclomatic|mccabe|complexity', 'complexity.cyclomatic_complexity_max'),
    (r'nest', 'complexity.nesting_depth_max'),
    (r'parameter|argument', 'complexity.parameter_count_max'),
    (r'function length|how long|lines|length', 'complexity.function_length_max'),
    (r'\bgoto\b', 'prohibited.goto'),
    (r'dynamic memory|malloc|heap|\bnew\b|allocat', 'prohibited.dynamic_memory'),
    (r'exception|try|catch|throw', 'prohibited.exceptions'),
    (r'recursi', 'prohibited.recursion'),
    (r'rtti|dynamic_cast|typeid', 'prohibited.rtti'),
    (r'class(?:es)? nam|nam\w* (?:of |for )?class', 'naming.class'),
    (r'naming|camel|pascal|snake|function names?', 'naming.function'),
    (r'comment|document', 'documentation.missing_docstring'),
]
_STANDARDS_TOPIC = re.compile(r'\b(?:standards?|clauses?|rules?|guidelines?|coding standard|document|allowed|permitted|'
                              r'prohibited|forbidden|limit|maximum|max|convention|required?|requires|can i|may i|ok to)\b', re.IGNORECASE)


def _rule_line(row: dict) -> str:
    origin = (f"from the document — {row['citation']}: \"{row['clause']}\"" if row['origin'] == 'document' and row['clause']
              else f"from the document ({row['citation']})" if row['origin'] == 'document'
              else 'built-in default (the document does not set it)')
    checked = '' if row['enforced'] else ' [not checked automatically]'
    return f"  {row['label']}: {row['value']}{checked} — {origin}"


def _standards_question(question: str, results: AnalysisResults) -> Optional[str]:
    """Coding-standard questions: the rules in force (deterministic), else clauses retrieved from the document."""
    std = results.standards
    doc = getattr(results, 'standards_doc', None)
    q = question.lower()
    if std is None or 'violation' in q:
        return None
    if not (_STANDARDS_TOPIC.search(q) or any(re.search(p, q) for p, _ in _RULE_WORDS)):
        return None
    rows = getattr(std, 'rule_table', None) or []
    source = f"Coding standard: {std.standards_source}"
    if re.search(r'not (?:be )?(?:checked|enforced|verified)|unchecked|cannot (?:be )?check', q):
        clauses = getattr(std, 'unchecked_clauses', []) or []
        return (f"{source}\nNormative clauses the tool does not check automatically: {len(clauses)}\n"
                + ('\n'.join(f"  {cite}: \"{text}\"" for text, cite in clauses) or '  none'))
    if re.search(r'which rules|what rules|rules (?:in force|extracted|apply|used)|from the document|default', q):
        doc_rows = [r for r in rows if r['origin'] == 'document']
        return (f"{source}\n"
                f"{len(doc_rows)} rule(s) set by the document, {len(rows) - len(doc_rows)} built-in default(s):\n"
                + '\n'.join(_rule_line(r) for r in rows))
    by_rule = {r['rule']: r for r in rows}
    for pattern, rule in _RULE_WORDS:
        if re.search(pattern, q) and rule in by_rule:
            answer = f"{source}\n{_rule_line(by_rule[rule])}"
            if doc is not None and by_rule[rule]['origin'] != 'document':
                hits = [h for h in doc.search(question, k=2) if h.lexical_rank is not None or h.dense_rank == 0]
                answer += ''.join(f"\n  Related clause (retrieved, not the rule source) — {h.chunk.cite()}: "
                                  f"\"{best_sentence(h.chunk.text, question)}\"" for h in hits)
            return answer
    if doc is not None:
        hits = doc.search(question, k=3)
        if hits:
            return (f"From {Path(doc.path).name} (retrieved by {doc.retriever.mode}; quoted, not interpreted):\n"
                    + '\n'.join(f"  {i}. {h.chunk.cite()}: \"{best_sentence(h.chunk.text, question)}\""
                                 for i, h in enumerate(hits, 1)))
    return None


def _facts(question: str, results: AnalysisResults) -> Optional[str]:
    """Structural answer, else summary answer, else a not-found message; None when nothing applies.

    A likely typo ("TemperatureSensr") gets its close names before the summary answers are tried; a name
    that is just a common word ("the impact of the critical violations") does not."""
    answer, missing = _structural_answer(question, results)
    if answer:
        return answer
    if missing is not None and missing.likely_typo:
        return missing
    return _standards_question(question, results) or _summary_answer(question, results) or missing


def offline_answer(question: str, results: AnalysisResults) -> str:
    return _facts(question, results) or _HELP


def answer_question(question: str, results: AnalysisResults,
                    narrator: Optional[RAGNarrativeGenerator] = None) -> str:
    """Deterministic facts first; with a *narrator* (local LLM) the facts are also explained.

    When no fact matches, the LLM is given the all-results summary as its facts and its answer is labelled."""
    facts = _facts(question, results)
    if narrator is not None:
        explanation = narrator.answer_question(question, facts or results_summary(results))
        if explanation:
            if facts:
                return f"{facts}\n\nExplanation (local LLM):\n{explanation}"
            return f"{_NO_FACTS_LABEL}\n{explanation}"
    return facts or _HELP
