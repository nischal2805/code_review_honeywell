"""Offline Q&A: answers questions from the structured analysis results, without an LLM.

The deterministic facts come first; when a local LLM is available it only explains them.
"""
from __future__ import annotations
import difflib
import re
from collections import Counter
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Set, Tuple

from rag_engine.features.function_effect import FunctionEffectAnalyzer, format_effect

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
         "  'what is the inheritance of <Class>'")


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


def sensor_check_text(finding: SensorFinding) -> str:
    """Check statuses and data flow of one sensor (R3), as plain text."""
    tpl = finding.template
    if finding.status != 'found':
        return f"Sensor {tpl.name}: no code class found for this template."
    lines = [f"Sensor {tpl.name} (class {finding.class_name}, {finding.file_path}:{finding.line_number})",
             f"Hierarchy: {' -> '.join(finding.hierarchy)}",
             "Checks applied:"]
    for name, c in finding.checks.items():
        found = f"; found {', '.join(f'{v:g}' for v in c.found)}" if c.found else ''
        lines.append(f"  {name}: {c.status} (expected {c.expected}{found})")
        lines += [f"    {e}" for e in c.evidence]
    for title, items in (('Sources', finding.sources), ('Processors', finding.processors),
                         ('Validators', finding.validators), ('Consumers', finding.consumers)):
        lines.append(f"{title}: {', '.join(items) or 'none'}")
    return '\n'.join(lines)


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
    _virtuals_of_class, _effect_of_method, _checks_of_sensor, _hierarchy_of_class]


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
    return '\n'.join(lines)


def _facts(question: str, results: AnalysisResults) -> Optional[str]:
    """Structural answer, else summary answer, else a not-found message; None when nothing applies.

    A likely typo ("TemperatureSensr") gets its close names before the summary answers are tried; a name
    that is just a common word ("the impact of the critical violations") does not."""
    answer, missing = _structural_answer(question, results)
    if answer:
        return answer
    if missing is not None and missing.likely_typo:
        return missing
    return _summary_answer(question, results) or missing


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
