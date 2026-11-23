"""Any uploaded coding-standards document -> enforced rules + a searchable index.

``load_standards_document(path)`` reads a PDF / DOCX / TXT / Markdown / YAML / JSON file, splits it into
sentences with page and section, and extracts the rules the validator enforces from normative clauses:

* complexity limits   "Cyclomatic complexity shall not exceed 12", "no more than 6 parameters", ...
* prohibited constructs (goto, dynamic memory, exceptions, recursion, RTTI) with a policy from the modal
  verb: shall not / prohibited -> FORBIDDEN; should not / unless justified -> RESTRICTED; may -> ALLOWED
* naming conventions  "Function names shall be camelCase", "Classes shall use PascalCase"
* documentation       "Every function shall have a header comment"
* DAL scope           "For DAL A, ..." / a section titled "Level B" limits a clause to those levels

Every extracted rule keeps the clause it came from (file, page, section), so violations can cite it.
Normative clauses that match no checkable rule are kept as ``unchecked`` so the report can show what the
tool does *not* enforce. Rules the document does not mention fall back to the built-in defaults and are
reported as such. The document's chunks are indexed for hybrid retrieval (BM25 + dense + reranking)."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from rag_engine.knowledge_base.standards_profile import parse_standards_document
from rag_engine.retrieval.chunking import Chunk, chunk_blocks, read_blocks, sentences
from rag_engine.retrieval.models import load_embedder, load_reranker
from rag_engine.retrieval.retriever import HybridRetriever

DALS = ('A', 'B', 'C', 'D')
_POLICY_RANK = {'ALLOWED': 0, 'RESTRICTED': 1, 'FORBIDDEN': 2}

# rule key -> (human label, retrieval query used to find the clause when the rule came from the defaults)
RULES: Dict[str, Tuple[str, str]] = {
    'complexity.cyclomatic_complexity_max': ('Maximum cyclomatic complexity', 'cyclomatic complexity limit'),
    'complexity.function_length_max': ('Maximum function length (lines)', 'maximum function length lines'),
    'complexity.nesting_depth_max': ('Maximum nesting depth', 'maximum nesting depth'),
    'complexity.parameter_count_max': ('Maximum parameters per function', 'maximum number of function parameters'),
    'prohibited.goto': ('goto', 'goto statement'),
    'prohibited.dynamic_memory': ('Dynamic memory', 'dynamic memory allocation heap'),
    'prohibited.exceptions': ('Exceptions', 'exception handling try catch throw'),
    'prohibited.recursion': ('Recursion', 'recursion recursive functions'),
    'prohibited.rtti': ('RTTI', 'run-time type information dynamic_cast typeid'),
    'naming.function': ('Function naming', 'function naming convention'),
    'naming.class': ('Class naming', 'class naming convention'),
    'naming.variable': ('Variable naming', 'variable naming convention'),
    'naming.constant': ('Constant naming', 'constant naming convention'),
    'documentation.missing_docstring': ('Function documentation', 'function header comment documentation'),
}
# validator rule id -> rule key
VIOLATION_RULES = {
    'DO178-CC': 'complexity.cyclomatic_complexity_max', 'DO178-FUNC-LEN': 'complexity.function_length_max',
    'DO178-NESTING': 'complexity.nesting_depth_max', 'DO178-PARAM-COUNT': 'complexity.parameter_count_max',
    'DO178-GOTO': 'prohibited.goto', 'DO178-DYNAMIC-MEM': 'prohibited.dynamic_memory',
    'DO178-EXCEPTION': 'prohibited.exceptions', 'DO178-RECURSION': 'prohibited.recursion',
    'DO178-RTTI': 'prohibited.rtti', 'DO178-NAMING': 'naming.function', 'DO178-CLASS-NAMING': 'naming.class',
    'DO178-NO-DOCSTRING': 'documentation.missing_docstring',
}
ENFORCED = {k for k in RULES if k not in ('naming.variable', 'naming.constant')}

CONVENTIONS = {
    'camelCase': r'^[a-z][a-zA-Z0-9]*$',
    'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
    'snake_case': r'^[a-z][a-z0-9_]*$',
    'SCREAMING_SNAKE_CASE': r'^[A-Z][A-Z0-9_]*$',
}
_CONVENTION_WORDS = [   # checked in this order: longer phrases first
    (r'screaming[_ ]snake[_ ]case|upper[_ ]?snake[_ ]case|all[ -]?(?:caps|upper[- ]?case)|upper[- ]?case(?: letters)?(?: with|and|separated by) underscores|UPPER_CASE|ALL_CAPS|macro case', 'SCREAMING_SNAKE_CASE'),
    (r'snake[_ ]case|lower[- ]?case(?: letters)?(?: with|and|separated by) underscores|lower_case', 'snake_case'),
    (r'pascal[ -]?case|upper[ -]?camel[ -]?case|UpperCamelCase|capitali[sz]ed camel', 'PascalCase'),
    (r'lower[ -]?camel[ -]?case|camel[ -]?case|lowerCamelCase|mixed[ -]?case', 'camelCase'),
]
_ENTITIES = [
    (r'\b(?:functions?|methods?|member functions?|routines?|procedures?|subprograms?)\b', 'function'),
    (r'\b(?:class(?:es)?|structs?|structures?|types?(?! information)|enums?)\b', 'class'),
    (r'\b(?:constants?|macros?|#defines?|enumerators?)\b', 'constant'),
    (r'\b(?:variables?|locals?|members?|fields?|attributes?|parameters?|arguments?)\b', 'variable'),
]
_CONSTRUCTS = [
    ('goto', r'\bgoto\b|\bgo to statements?\b'),
    ('dynamic_memory', r'dynamic(?:ally)? (?:memory|allocat\w*)|\bheap\b|\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\(\)|operator new|\bnew/delete\b|\bnew and delete\b|memory allocation'),
    ('exceptions', r'\bexceptions?\b|exception handling|\btry\s*/\s*catch\b|\btry\b[^.]{0,30}\bcatch\b|\bthrow\b'),
    ('recursion', r'recursi(?:on|ve)'),
    ('rtti', r'\brtti\b|dynamic_cast|\btypeid\b|run[- ]?time type'),
]
_FORBID = re.compile(r"\b(?:shall|must|may|will) not\b|\b(?:shall|must) never\b|\bis not (?:allowed|permitted)\b|"
                     r"\bare not (?:allowed|permitted)\b|\bnot (?:be )?(?:used|permitted|allowed)\b|\bprohibited\b|"
                     r"\bforbidden\b|\bbanned\b|\bdisallowed\b|\bnever\b|\bno\s+(?:use of\s+)?(?=\w)", re.IGNORECASE)
_SOFTEN = re.compile(r"\bunless\b|\bexcept\b|\bonly (?:if|when|with|after|where|in)\b|\bwithout (?:a |an |prior )?"
                     r"(?:justification|deviation|approval|waiver)\b|\bjustif\w+\b|\bdeviation\b|\bwaiver\b", re.IGNORECASE)
_RESTRICT = re.compile(r"\bshould not\b|\bshould be avoided\b|\bavoid(?:ed)?\b|\bdiscouraged\b|\brestricted\b|"
                       r"\bminimi[sz]ed?\b|\bwith care\b|\bsparingly\b", re.IGNORECASE)
_ALLOW = re.compile(r"\b(?:may be used|is allowed|are allowed|is permitted|are permitted|is acceptable|are acceptable|may)\b",
                    re.IGNORECASE)
_NORMATIVE = re.compile(r"\b(?:shall|must|should|required|requires?|prohibited|forbidden|not permitted|"
                        r"not allowed|mandatory)\b", re.IGNORECASE)
_LIMIT = re.compile(r"(?:shall not exceed|must not exceed|should not exceed|not exceed|no (?:more|greater) than|"
                    r"not (?:be )?(?:more|greater) than|at most|a maximum of|maximum(?: of| is| allowed(?: is)?)?|"
                    r"max(?:imum)?\.?\s*[:=]?|limited to|less than or equal to|up to|<=|≤|=<)\s*(\d+)", re.IGNORECASE)
_COMPARATIVE = re.compile(r"(?:longer|larger|more|greater|bigger|deeper) than\s*(\d+)", re.IGNORECASE)
_STRICT = re.compile(r"(?:less than|fewer than|below|under|<)\s*(\d+)", re.IGNORECASE)
_TRAILING = re.compile(r"(\d+)\s+(?:\w+\s+){0,2}(?:or (?:fewer|less)|maximum|max\b|at most)", re.IGNORECASE)
_DAL = re.compile(r"\b(?:[Dd][Aa][Ll]s?|[Ll]evels?|[Ss]oftware [Ll]evels?|[Dd]esign [Aa]ssurance [Ll]evels?)\s+"
                  r"([A-E](?:\s*(?:,|and|or|/|-|–|to|through)\s*[A-E])*)\b")


@dataclass
class RuleSource:
    rule: str                    # key of RULES
    value: Any                   # number, policy or convention
    dals: Tuple[str, ...]        # the levels it applies to
    clause: str                  # the sentence it came from
    citation: str                # "file.pdf, p.3, § 4.2 Function length"


@dataclass
class StandardsDocument:
    path: str
    profile: Dict[str, Any]                                    # what the validator merges over its defaults
    sources: Dict[str, List[RuleSource]] = field(default_factory=dict)
    unchecked: List[Tuple[str, str]] = field(default_factory=list)   # (normative clause, citation) not enforced
    chunks: List[Chunk] = field(default_factory=list)
    retriever: Optional[HybridRetriever] = None
    structured: bool = False                                   # YAML/JSON profile (no clause extraction)

    def citation_for(self, rule: str, dal: Optional[str] = None, value: Any = None) -> Optional[RuleSource]:
        """The clause behind *rule* at *dal*: the one giving the value in force, the most level-specific first."""
        options = [s for s in self.sources.get(rule, []) if dal is None or dal in s.dals]
        if value is not None and any(s.value == value for s in options):
            options = [s for s in options if s.value == value]
        return min(options, key=lambda s: len(s.dals)) if options else None

    def search(self, query: str, k: int = 3):
        return self.retriever.search(query, k=k) if self.retriever else []


def _dal_scope(text: str) -> Optional[Tuple[str, ...]]:
    found: List[str] = []
    for m in _DAL.finditer(text):
        spec = m.group(1)
        for a, b in re.findall(r'([A-E])\s*(?:-|–|to|through)\s*([A-E])', spec):
            found += [chr(c) for c in range(ord(a), ord(b) + 1)]
        found += re.findall(r'[A-E]', spec)
    dals = tuple(d for d in DALS if d in found)
    return dals or None


def _policy(sentence: str) -> Optional[str]:
    if _FORBID.search(sentence):
        return 'RESTRICTED' if _SOFTEN.search(sentence) else 'FORBIDDEN'
    if _RESTRICT.search(sentence):
        return 'RESTRICTED'
    if _ALLOW.search(sentence):
        return 'ALLOWED'
    return None


def _limit(sentence: str) -> Optional[int]:
    m = _LIMIT.search(sentence)
    if m:
        return int(m.group(1))
    m = _COMPARATIVE.search(sentence)
    if m and re.search(r"\b(?:no|not|never)\b|n't\b", sentence, re.IGNORECASE):   # "no body longer than 80"
        return int(m.group(1))
    m = _STRICT.search(sentence)
    if m:
        return int(m.group(1)) - 1
    m = _TRAILING.search(sentence)
    return int(m.group(1)) if m else None


def _metric(s: str) -> Optional[str]:
    low = s.lower()
    if 'cyclomatic' in low or 'mccabe' in low:
        return 'complexity.cyclomatic_complexity_max'
    if re.search(r'\bnest(?:ing|ed)\b', low):
        return 'complexity.nesting_depth_max'
    if re.search(r'\b(?:parameters?|arguments?)\b', low) and re.search(r'function|method|routine|procedure|subprogram|interface|signature|accept|take|pass', low):
        return 'complexity.parameter_count_max'
    if re.search(r'\b(?:lines?|loc|sloc|statements?|length|long)\b', low) and re.search(r'function|method|routine|procedure|subprogram|body|bodies', low):
        return 'complexity.function_length_max'
    if re.search(r'\bcomplexity\b', low):
        return 'complexity.cyclomatic_complexity_max'
    return None


def _naming(s: str) -> List[Tuple[str, str]]:
    """(entity, convention) pairs in a sentence; a convention binds to the nearest entity before it."""
    if not re.search(r'\b(?:nam\w*|identifiers?|case|convention)\b|_case\b', s, re.IGNORECASE):
        return []
    marks: List[Tuple[int, str, str]] = []
    for pattern, conv in _CONVENTION_WORDS:
        for m in re.finditer(pattern, s, re.IGNORECASE):
            if not any(a <= m.start() < b for a, b, _ in [(x[0], x[0] + len(x[2]), x[1]) for x in marks]):
                marks.append((m.start(), 'conv', conv))
    for pattern, entity in _ENTITIES:
        for m in re.finditer(pattern, s, re.IGNORECASE):
            marks.append((m.start(), 'entity', entity))
    pairs: List[Tuple[str, str]] = []
    pending: List[str] = []
    for _, kind, value in sorted(marks):
        if kind == 'entity':
            pending.append(value)
        elif pending:
            pairs += [(e, value) for e in dict.fromkeys(pending)]
            pending = []
    return list(dict.fromkeys(pairs))


def extract_rules(items: List[Tuple[str, Optional[int], str]], source: str
                  ) -> Tuple[Dict[str, List[RuleSource]], List[Tuple[str, str]]]:
    """Rules found in (sentence, page, section) items, and the normative sentences that gave none."""
    sources: Dict[str, List[RuleSource]] = {}
    unchecked: List[Tuple[str, str]] = []
    seen = set()
    section_dals: Dict[str, Optional[Tuple[str, ...]]] = {}
    name = Path(source).name
    for sentence, page, section in items:
        key = ' '.join(sentence.lower().split())
        if key in seen:
            continue
        seen.add(key)
        cite = ', '.join([name] + ([f'p.{page}'] if page else []) + ([f'§ {section}'] if section else []))
        if re.search(r'\bSCS-[LM]-\d{2}\b', sentence, re.IGNORECASE):
            continue              # per-DAL table rows: read by the table parser, not as prose
        if section not in section_dals:
            section_dals[section] = _dal_scope(section)
        dals = _dal_scope(sentence) or section_dals[section] or DALS
        found: List[RuleSource] = []

        metric = _metric(sentence)
        limit = _limit(sentence) if metric else None
        if metric and limit is not None and 0 < limit < 10000:
            found.append(RuleSource(metric, limit, dals, sentence, cite))
        policy = _policy(sentence)
        if policy:
            for construct, pattern in _CONSTRUCTS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    found.append(RuleSource(f'prohibited.{construct}', policy, dals, sentence, cite))
        for entity, conv in _naming(sentence):
            found.append(RuleSource(f'naming.{entity}', conv, DALS, sentence, cite))
        if (re.search(r'\b(?:functions?|methods?|routines?|subprograms?)\b', sentence, re.IGNORECASE)
                and re.search(r'\b(?:comment\w*|document\w*|header|doxygen|description|prologue)\b', sentence, re.IGNORECASE)
                and _NORMATIVE.search(sentence) and not metric):
            strength = 'RESTRICTED' if re.search(r'\b(?:shall|must|required?|requires|mandatory)\b', sentence, re.IGNORECASE) else 'ALLOWED'
            found.append(RuleSource('documentation.missing_docstring', strength, DALS, sentence, cite))
        for f in found:
            sources.setdefault(f.rule, []).append(f)
        if not found and _NORMATIVE.search(sentence) and len(sentence.split()) >= 4:
            unchecked.append((sentence, cite))
    return sources, unchecked


def _strictest(values: List[Any], rule: str) -> Any:
    if rule.startswith('complexity.'):
        return min(values)
    if rule.startswith('prohibited.'):
        return max(values, key=lambda v: _POLICY_RANK.get(v, 0))
    return values[0]


def build_profile(sources: Dict[str, List[RuleSource]]) -> Dict[str, Any]:
    """Validator profile from extracted rules; when clauses disagree for a level the strictest wins."""
    profile: Dict[str, Any] = {}
    for rule, items in sources.items():
        group, name = rule.split('.', 1)
        if group == 'naming':
            conv = items[0].value
            profile.setdefault('naming_conventions', {})[name] = {'regex': CONVENTIONS[conv], 'convention': conv}
            continue
        if group == 'documentation':
            profile.setdefault('universal_standards', {}).setdefault('documentation', {})[name] = \
                _strictest([i.value for i in items], 'prohibited.')
            continue
        if rule == 'prohibited.goto':
            profile.setdefault('universal_standards', {}).setdefault('prohibited_constructs', {})['goto'] = \
                _strictest([i.value for i in items], rule)
            continue
        for dal in DALS:
            values = [i.value for i in items if dal in i.dals]
            if not values:
                continue
            dal_cfg = profile.setdefault('dal_specific_standards', {}).setdefault(dal, {})
            if group == 'complexity':
                dal_cfg.setdefault('complexity', {})[name] = _strictest(values, rule)
            else:
                dal_cfg.setdefault('severities', {}).setdefault('prohibited_constructs', {})[name] = _strictest(values, rule)
    return profile


def _table_profile(text: str) -> Dict[str, Any]:
    """The per-DAL rule table format (SCS-L-xx / SCS-M-xx rows), when the document has one."""
    if not re.search(r'\bSCS-[LM]-\d{2}\b', text, re.IGNORECASE):
        return {}
    parsed = parse_standards_document(text)
    parsed.pop('naming_conventions', None)        # the table parser only has hard-coded naming defaults
    parsed.get('universal_standards', {}).pop('documentation', None)
    return parsed


_TABLE_ROW_WORDS = {
    'complexity.cyclomatic_complexity_max': 'cyclomatic', 'complexity.function_length_max': 'function length',
    'complexity.nesting_depth_max': 'nesting', 'complexity.parameter_count_max': 'parameters',
    'prohibited.goto': 'goto', 'prohibited.dynamic_memory': 'dynamic memory', 'prohibited.exceptions': 'exceptions',
    'prohibited.recursion': 'recursion', 'prohibited.rtti': 'rtti',
}


def _table_sources(profile: Dict[str, Any], items, source: str) -> Dict[str, List[RuleSource]]:
    """Citations for the values the table parser read: the row of each rule, per level."""
    out: Dict[str, List[RuleSource]] = {}
    rows = {s: (p, sec) for s, p, sec in items if re.search(r'\bSCS-[LM]-\d{2}\b', s, re.IGNORECASE)}

    def add(rule: str, value: Any, dal: Tuple[str, ...]) -> None:
        row = next((r for r in rows if _TABLE_ROW_WORDS[rule] in r.lower()), None)
        if row is None:
            return
        page, sec = rows[row]
        cite = ', '.join([Path(source).name] + ([f'p.{page}'] if page else []) + ([f'§ {sec}'] if sec else []))
        existing = next((r for r in out.get(rule, []) if r.value == value), None)
        if existing:
            existing.dals = tuple(d for d in DALS if d in set(existing.dals) | set(dal))
        else:
            out.setdefault(rule, []).append(RuleSource(rule, value, dal, row, cite))

    for dal, cfg in profile.get('dal_specific_standards', {}).items():
        for name, value in cfg.get('complexity', {}).items():
            if f'complexity.{name}' in _TABLE_ROW_WORDS:
                add(f'complexity.{name}', value, (dal,))
        for name, value in cfg.get('severities', {}).get('prohibited_constructs', {}).items():
            if f'prohibited.{name}' in _TABLE_ROW_WORDS:
                add(f'prohibited.{name}', value, (dal,))
    goto = profile.get('universal_standards', {}).get('prohibited_constructs', {}).get('goto')
    if goto:
        add('prohibited.goto', goto, DALS)
    return out


def load_standards_document(path: str, cache_dir: Optional[str] = None, embedding_model: Optional[str] = None,
                            reranker_model: Optional[str] = None) -> StandardsDocument:
    """Read, extract and index a standards file. Raises ValueError for unreadable or unsupported files."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f'Standards file not found: {path}')
    try:
        blocks = read_blocks(p)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Standards file {path} could not be read: {type(exc).__name__}: {exc}') from exc
    items = list(sentences(blocks))
    chunks = chunk_blocks(blocks, str(p))
    retriever = HybridRetriever(chunks, load_embedder(embedding_model), load_reranker(reranker_model),
                                cache_dir=cache_dir, collection=f'standards-{p.stem}') if chunks else None
    doc = StandardsDocument(str(p), {}, chunks=chunks, retriever=retriever)

    if p.suffix.lower() in ('.yaml', '.yml', '.json'):
        text = p.read_text(encoding='utf-8', errors='replace')
        data = json.loads(text) if p.suffix.lower() == '.json' else yaml.safe_load(text)
        if isinstance(data, dict) and any(k in data for k in ('universal_standards', 'dal_specific_standards',
                                                               'naming_conventions', 'standards_profile')):
            doc.profile, doc.structured = copy.deepcopy(data), True
            return doc

    full_text = '\n'.join(b.text for b in blocks)
    table = _table_profile(full_text)
    sources, doc.unchecked = extract_rules(items, str(p))
    for rule, srcs in _table_sources(table, items, str(p)).items():
        sources.setdefault(rule, [])[:0] = srcs        # table rows first: they are per-level and explicit
    doc.sources = sources
    doc.profile = _merge(table, build_profile({k: v for k, v in sources.items()
                                               if not any(s.clause.upper().startswith('SCS-') for s in v)}))
    logger.info(f'Standards {p.name}: {sum(len(v) for v in sources.values())} rule clause(s) for '
                f'{len(sources)} rule(s); {len(doc.unchecked)} normative clause(s) not automatically checked; '
                f'{len(chunks)} chunks indexed ({retriever.mode if retriever else "empty"})')
    return doc


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    from rag_engine.knowledge_base.standards_profile import deep_merge
    return deep_merge(base, override) if base else override
