from __future__ import annotations
import re
from typing import Dict, Iterator, List, Optional, Set, Tuple
import networkx as nx
from loguru import logger

from rag_engine.models import FunctionDef, ParseResult

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
# Per-call-site iteration
# ---------------------------------------------------------------------------

def _iter_call_sites(body: str, call_token: str) -> Iterator[List[str]]:
    """Yield the argument list (as a list of expression strings) for every
    call site of *call_token* in *body*.

    Each yielded value is the result of ``_split_args`` on the text between
    the parentheses of one call site.  This lets the caller perform
    per-site type inference rather than aggregating over all sites.

    ``call_token`` may be:
    - a bare name:               ``"process"``
    - a qualified name:          ``"Utils::log"``
    - a member-access token:     ``"c.calculate"``  /  ``"ptr->compute"``
    """
    escaped = re.escape(call_token)
    pattern = re.compile(escaped + r'\s*\(')
    pos = 0
    while True:
        m = pattern.search(body, pos)
        if not m:
            break
        start = m.end() - 1   # index of '('
        depth = 0
        found_close = False
        for i in range(start, len(body)):
            ch = body[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    yield _split_args(body[start + 1:i])
                    found_close = True
                    pos = i + 1
                    break
        if not found_close:
            break


# ---------------------------------------------------------------------------
# Overload-aware candidate filtering
# ---------------------------------------------------------------------------

def _filter_candidates(
    candidates: Set[str],
    call_token: str,
    body: str,
    all_funcs: Dict[str, FunctionDef],
) -> Set[str]:
    """Resolve *candidates* to the best-matching overload(s) given all call
    sites of *call_token* in *body*.

    Strategy (per call site, then union across sites):
    1. **Type-aware filtering**: infer C++ type categories from argument
       literals.  Score each candidate by the number of positional parameters
       whose type is compatible with the inferred type.  Keep only the
       highest-scoring candidates for this call site.
    2. **Arity-only fallback**: if type inference yields all-``None`` for a
       call site (arguments are variables/expressions, not literals), filter
       to candidates whose parameter count matches the observed arity.
    3. **Conservative fallback**: if arity filtering eliminates all candidates
       (e.g. default parameters), include all candidates for this site.

    The final result is the **union** over every call site found.  If *body*
    contains no call site for *call_token*, return all *candidates*
    (conservative — the call may be indirect or macro-generated).
    """
    if not candidates:
        return candidates

    result: Set[str] = set()
    found_any_site = False

    for arg_exprs in _iter_call_sites(body, call_token):
        found_any_site = True
        arity = len(arg_exprs)

        # ── Step 1: arity filter ──────────────────────────────────────────
        arity_matched = {
            sig for sig in candidates
            if sig in all_funcs and len(all_funcs[sig].parameters) == arity
        }
        # If arity filtering removes everything (default args, variadic), fall
        # back to all candidates for this site.
        pool = arity_matched if arity_matched else candidates

        # ── Step 2: type-aware scoring ────────────────────────────────────
        # Only attempt when at least one argument has an inferable type.
        inferred = [_infer_arg_type(e) for e in arg_exprs]

        if any(t is not None for t in inferred):
            # Score each candidate: count how many positional types match.
            best_score = -1
            scores: Dict[str, int] = {}
            for sig in pool:
                fn = all_funcs.get(sig)
                if fn is None:
                    continue
                score = 0
                for pos, inf_type in enumerate(inferred):
                    if inf_type is None:
                        continue   # unknown argument — neutral
                    if pos < len(fn.parameters):
                        if _types_match(inf_type, fn.param_types[pos]):
                            score += 1
                        else:
                            score -= 1   # penalise type mismatch
                scores[sig] = score
                if score > best_score:
                    best_score = score

            # Keep only candidates with the highest score.
            typed_filtered = {sig for sig, sc in scores.items() if sc == best_score}

            # If every candidate scored equally (no discrimination power),
            # do NOT narrow further — keep the arity-filtered pool.
            if len(typed_filtered) < len(pool):
                pool = typed_filtered

        result |= pool

    if not found_any_site:
        # No call site found in body — conservative: add all candidates.
        return candidates

    return result


# ---------------------------------------------------------------------------
# Main call-graph builder
# ---------------------------------------------------------------------------

class CallGraphBuilder:
    def __init__(self, parse_results: Dict[str, ParseResult]) -> None:
        self._results = parse_results
        self._graph: CallGraph = nx.DiGraph()
        # Keyed by sig_key (e.g. "Calculator::compute(int)") — not by bare name.
        self._all_funcs: Dict[str, FunctionDef] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def build(self) -> CallGraph:
        # Phase 1: register every function as a graph node, keyed by sig_key.
        for result in self._results.values():
            for fn in result.functions:
                self._all_funcs[fn.sig_key] = fn
                self._graph.add_node(fn.sig_key, function=fn)

        known_sig_keys: Set[str] = set(self._all_funcs.keys())

        # Phase 2: resolve call edges for every function.
        for result in self._results.values():
            for fn in result.functions:
                callees = self._extract_calls(fn, known_sig_keys)
                fn.calls.update(callees)
                for callee_key in callees:
                    self._graph.add_edge(fn.sig_key, callee_key)
                    if callee_key in self._all_funcs:
                        self._all_funcs[callee_key].called_by.add(fn.sig_key)

        logger.debug(
            f"Call graph: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
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

    # ── Call extraction ─────────────────────────────────────────────────────

    def _extract_calls(self, fn: FunctionDef, known_sig_keys: Set[str]) -> Set[str]:
        if not fn.body:
            return set()

        # Strip comments before any regex scanning so that commented-out calls
        # are not treated as live call sites.
        body = _strip_comments(fn.body)

        resolved: Set[str] = set()

        # Index: short base name → {sig_key, …}
        _by_short: Dict[str, Set[str]] = self._build_short_name_index(known_sig_keys)

        # Index: qualified name (no param list) → {sig_key, …}
        # e.g. "Utils::log" → {"Utils::log()", "Utils::log(const std::string &)"}
        _by_qualified: Dict[str, Set[str]] = {}
        for sig_key in known_sig_keys:
            qualified = sig_key.split('(')[0]
            _by_qualified.setdefault(qualified, set()).add(sig_key)

        # Track names that have been precisely resolved as FREE FUNCTIONS in
        # this body.  Pass 3 skips these to avoid double-counting.
        # Crucially this is SEPARATE from member-method resolution: resolving
        # obj.method() must NOT block the free-function pass for a same-named
        # free function called elsewhere in the same body.
        _resolved_free: Set[str] = set()

        # Track short method names that were PRECISELY resolved via type
        # inference in Pass 2.  Pass 3 must skip these to avoid conservatively
        # re-adding all same-named member methods across every class.
        # Example: calc.reset() resolved precisely to AdvancedCalculator::reset()
        # — Pass 3 must not then add Calculator::reset() via the conservative
        # member-expansion path.
        _resolved_member: Set[str] = set()

        # ── Pass 1: Fully-qualified calls ─────────────────────────────────────
        # Matches e.g. "Utils::log(...)", "AdvancedCalculator::compute(...)"
        for qcall in re.findall(r'\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\(', body):
            short = qcall.rsplit('::', 1)[-1]
            candidates = _by_qualified.get(qcall, set())
            if candidates:
                for sig_key in _filter_candidates(candidates, qcall, body, self._all_funcs):
                    if sig_key != fn.sig_key:
                        resolved.add(sig_key)
                # Qualified calls are always precise — mark the short name so
                # Pass 3 does not re-expand it as a free-function call.
                _resolved_free.add(short)
            else:
                # Qualified name not stored as-is (parser dropped the prefix).
                # Fall back to short-name expansion with type-aware filtering.
                candidates_short = _by_short.get(short, set())
                for sig_key in _filter_candidates(candidates_short, qcall, body, self._all_funcs):
                    if sig_key != fn.sig_key:
                        resolved.add(sig_key)

        # ── Pass 2: Member / pointer-member calls ─────────────────────────────
        # obj.method(…)  or  ptr->method(…)

        # Build var → TypeName from parameters and local declarations.
        _var_type: Dict[str, str] = {}

        # (a) Parameters — strip qualifiers; the parser may attach * / & to
        #     either the type or the name field.
        for p in fn.parameters:
            bare_type = re.sub(r'[\*&]', '', p.type_).strip().split()[-1] if p.type_.strip() else ''
            bare_name = re.sub(r'[\*&\s]', '', p.name)
            if bare_name and bare_type:
                _var_type[bare_name] = bare_type

        # (b) Local declarations: ClassName var;  /  ClassName* ptr = …;
        _LOCAL_DECL = re.compile(
            r'\b([A-Z][A-Za-z_]\w*)\s*[\*&]?\s+([a-z_]\w*)\s*(?:=|;|\()'
        )
        for type_name, var_name in _LOCAL_DECL.findall(body):
            _var_type[var_name] = type_name

        # Resolve member calls.
        _MEMBER_CALL = re.compile(r'\b([A-Za-z_]\w*)\s*(?:->|\.)([A-Za-z_]\w*)\s*\(')
        for receiver, method in _MEMBER_CALL.findall(body):
            candidates = _by_short.get(method, set())
            if not candidates:
                continue

            inferred_type = _var_type.get(receiver)
            if inferred_type:
                type_candidates = {
                    c for c in candidates
                    if c.split('(')[0].startswith(inferred_type + '::')
                }
                if type_candidates:
                    # Use "receiver.method" as the call token for per-site filtering.
                    token = f"{receiver}.{method}"
                    filtered = _filter_candidates(type_candidates, token, body, self._all_funcs)
                    if not filtered:
                        # Try pointer syntax
                        token = f"{receiver}->{method}"
                        filtered = _filter_candidates(type_candidates, token, body, self._all_funcs)
                    for sig_key in filtered:
                        if sig_key != fn.sig_key:
                            resolved.add(sig_key)
                    # Mark this method name as precisely resolved via member
                    # call so Pass 3 does not re-expand it conservatively.
                    # NOTE: do NOT add to _resolved_free — a same-named free
                    # function in this body must still be processed by Pass 3.
                    _resolved_member.add(method)
                    continue   # precise path taken — do not fall through

            # Type unknown or no type-precise match — conservative fallback,
            # still filtered by type/arity where possible.
            token = f"{receiver}.{method}"
            filtered = _filter_candidates(candidates, token, body, self._all_funcs)
            if not filtered:
                token = f"{receiver}->{method}"
                filtered = _filter_candidates(candidates, token, body, self._all_funcs)
            for sig_key in filtered:
                if sig_key != fn.sig_key:
                    resolved.add(sig_key)

        # ── Pass 3: Plain (unqualified) free-function calls ───────────────────
        # Skip tokens that were already precisely resolved:
        #   _resolved_free   — resolved as a free function in Pass 1
        #   _resolved_member — resolved as a member method via type inference in Pass 2
        #
        # The two sets are DISTINCT:
        # - _resolved_member blocks the conservative member-expansion that would
        #   otherwise fire for bare tokens like 'reset' extracted from 'calc.reset()'.
        # - _resolved_free blocks re-processing of free functions already handled.
        # - A token in _resolved_member but NOT in _resolved_free still runs the
        #   free-function branch (e.g. first.process() resolves member, but the
        #   free-function process() must still be resolved below).
        #
        # Fan-out rule: if at least one free function (no '::') exists for this
        # short name, use only free-function candidates; otherwise expand
        # conservatively to member candidates (type genuinely unknown).
        for bare in re.findall(r'\b([A-Za-z_]\w*)\s*\(', body):
            if bare in _resolved_free:
                continue

            candidates = _by_short.get(bare, set())
            if not candidates:
                continue

            free_candidates   = {c for c in candidates if '::' not in c.split('(')[0]}
            member_candidates = candidates - free_candidates

            if free_candidates:
                for sig_key in _filter_candidates(free_candidates, bare, body, self._all_funcs):
                    if sig_key != fn.sig_key:
                        resolved.add(sig_key)
            elif bare not in _resolved_member:
                # No free function AND not already precisely resolved as a
                # member call — conservative expansion to member overloads.
                for sig_key in _filter_candidates(member_candidates, bare, body, self._all_funcs):
                    if sig_key != fn.sig_key:
                        resolved.add(sig_key)
            # If bare in _resolved_member and no free candidates: already
            # handled precisely in Pass 2 — do nothing.

        # ── Pass 4: Constructor from variable declaration ─────────────────────
        # Matches:  AdvancedCalculator calc;   MyClass obj;
        for ctor_class in re.findall(r'\b([A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*;', body):
            ctor_qualified = f"{ctor_class}::{ctor_class}"
            for sig_key in _by_qualified.get(ctor_qualified, set()):
                if sig_key != fn.sig_key:
                    resolved.add(sig_key)

        # ── Pass 5: Constructor from new expression ───────────────────────────
        # Matches:  new ClassName(…)   new ClassName;
        for ctor_class in re.findall(r'\bnew\s+([A-Za-z_]\w*)\s*[\(;]', body):
            ctor_qualified = f"{ctor_class}::{ctor_class}"
            for sig_key in _by_qualified.get(ctor_qualified, set()):
                if sig_key != fn.sig_key:
                    resolved.add(sig_key)

        # ── Pass 6: Callback / function-pointer arguments ─────────────────────
        # Identifiers passed by name as arguments to call sites.
        # Only fire for free functions (no '::') to avoid spurious class edges.
        _ARG_CALL = re.compile(r'\b[A-Za-z_]\w*\s*\(([^)]*)\)')
        for arg_list in _ARG_CALL.findall(body):
            for arg in re.findall(r'\b([A-Za-z_]\w*)\b', arg_list):
                for sig_key in _by_short.get(arg, set()):
                    if sig_key != fn.sig_key:
                        if '::' not in sig_key.split('(')[0]:
                            resolved.add(sig_key)

        return resolved
