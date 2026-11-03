from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

from rag_engine.core.ast_builder import _declared_name, find_child, find_nodes, get_text
from rag_engine.models import FunctionDef, ParseResult

_CPP_LANGUAGE = Language(tscpp.language())

_FALSE_LITERALS = ('0', 'false')
_JUMPS = {'return_statement': 'return', 'throw_statement': 'throw', 'break_statement': 'break',
          'continue_statement': 'continue', 'goto_statement': 'goto'}
_LABELS = ('case_statement', 'labeled_statement')   # reachable again: by the switch or by a goto
_SCALAR_TYPES = ('primitive_type', 'sized_type_specifier')


@dataclass
class DeadFragment:
    function: str        # sig_key of the containing function
    file_path: str
    start_line: int
    end_line: int
    reason: str
    snippet: str         # first line of the fragment, stripped
    jump: str = ''       # keyword of the jump the fragment follows; '' for a constant-false condition
    jump_line: int = 0
    condition: str = ''  # the constant-false condition text, e.g. "0"
    calls: List[str] = field(default_factory=list)   # callees written inside the fragment


@dataclass
class UnusedVariable:
    function: str        # sig_key of the containing function
    file_path: str
    name: str
    line: int
    kind: Literal['local', 'parameter']
    side_effects: List[str] = field(default_factory=list)   # initializer parts that must survive removing a local


def parse_root(result: ParseResult) -> Node:
    return Parser(_CPP_LANGUAGE).parse(result.raw_source).root_node


def _function_nodes(result: ParseResult) -> Tuple[Node, List[Tuple[FunctionDef, Node]]]:
    """The syntax tree root, and each parsed function that has a body and is not under ``#if 0`` with its node.

    A function is told from the others on its line by its body text; functions that share line span
    and body are paired in source order.
    """
    source = result.raw_source
    pending: Dict[Tuple[int, int, str], List[FunctionDef]] = {}
    for fn in result.functions:
        if fn.body and not fn.never_built:
            pending.setdefault((fn.line_number, fn.end_line_number, fn.body), []).append(fn)
    root = parse_root(result)
    pairs = []
    for node in find_nodes(root, 'function_definition'):
        body = find_child(node, 'compound_statement')
        key = (node.start_point[0] + 1, node.end_point[0] + 1, get_text(body, source) if body is not None else '')
        if pending.get(key):
            pairs.append((pending[key].pop(0), node))
    return root, pairs


def _first_line(node: Node, source: bytes) -> str:
    text = get_text(node, source).strip()
    return text.splitlines()[0].strip() if text else ''


def _calls_in(nodes: List[Node], source: bytes) -> List[str]:
    names: List[str] = []
    for node in nodes:
        for call in find_nodes(node, 'call_expression'):
            callee = call.child_by_field_name('function')
            name = get_text(callee, source).strip() if callee is not None else ''
            if name and name not in names:
                names.append(name)
    return names


# ── Dead fragments ─────────────────────────────────────────────────────────────

def find_dead_fragments(result: ParseResult) -> List[DeadFragment]:
    """Code that can never execute: statements after an unconditional jump in the same block
    and the bodies of ``if (0)``/``if (false)``/``while (0)``."""
    source = result.raw_source
    fragments: List[DeadFragment] = []
    for fn, node in _function_nodes(result)[1]:
        found: List[DeadFragment] = []

        def add(start: Node, end: Node, text_node: Node, reason: str, covered: List[Node], **extra) -> None:
            found.append(DeadFragment(fn.sig_key, fn.file_path, start.start_point[0] + 1, end.end_point[0] + 1,
                                      reason, _first_line(text_node, source), calls=_calls_in(covered, source), **extra))

        for block in find_nodes(node, 'compound_statement', 'case_statement'):
            for jump, jump_node, run in _runs_after_jump(block):
                add(run[0], run[-1], run[0], f"statements after unconditional `{jump}` can never execute", run,
                    jump=jump, jump_line=jump_node.start_point[0] + 1)
        for stmt in find_nodes(node, 'if_statement', 'while_statement'):
            condition = stmt.child_by_field_name('condition')
            value = condition.child_by_field_name('value') if condition is not None else None
            body = stmt.child_by_field_name('consequence' if stmt.type == 'if_statement' else 'body')
            if value is not None and body is not None and get_text(value, source).strip() in _FALSE_LITERALS:
                keyword = 'if' if stmt.type == 'if_statement' else 'while'
                text = get_text(value, source).strip()
                add(body, body, stmt, f"`{keyword} ({text})` is constant false; its body can never execute", [body],
                    condition=text)

        # A fragment inside another one adds nothing.
        found.sort(key=lambda f: (f.start_line, -f.end_line))
        kept: List[DeadFragment] = []
        for f in found:
            if not kept or f.end_line > kept[-1].end_line:
                kept.append(f)
        fragments += kept
    return fragments


def _runs_after_jump(block: Node) -> Iterator[Tuple[str, Node, List[Node]]]:
    """``(jump keyword, jump statement, statements)`` for each run of statements that follows a jump in *block*."""
    jump: Optional[str] = None
    jump_node: Optional[Node] = None
    run: List[Node] = []
    for child in block.named_children:
        if child.type == 'comment':
            continue
        if child.type in _LABELS:
            if run and jump and jump_node is not None:
                yield jump, jump_node, run
            jump, run = None, []
        elif jump:
            run.append(child)
        elif child.type in _JUMPS:
            jump, jump_node = _JUMPS[child.type], child
    if run and jump and jump_node is not None:
        yield jump, jump_node, run


# ── Unused variables ───────────────────────────────────────────────────────────

def find_unused_variables(result: ParseResult) -> List[UnusedVariable]:
    """Parameters never used in their function and locals never used after their declaration."""
    # shortcut: uses are matched by name within the declaring scope (a shadowing inner variable
    # counts as a use); upgrade to real scope resolution if that hides findings.
    source = result.raw_source
    found: List[UnusedVariable] = []
    root, pairs = _function_nodes(result)
    volatile_names = _volatile_names(root, source)
    for fn, node in pairs:
        sig = fn.sig_key

        def add(name: str, at: Node, kind: Literal['local', 'parameter'], effects: Optional[List[str]] = None) -> None:
            found.append(UnusedVariable(sig, fn.file_path, name, at.start_point[0] + 1, kind, effects or []))

        params = next(iter(find_nodes(node.child_by_field_name('declarator'), 'parameter_list')), None)
        outside_params = Counter(
            get_text(n, source) for n in find_nodes(node, 'identifier')
            if params is None or not params.start_byte <= n.start_byte < params.end_byte)
        for p in (params.named_children if params is not None else []):
            if p.type not in ('parameter_declaration', 'optional_parameter_declaration') or _has_attribute(p, source):
                continue
            for declarator in p.children_by_field_name('declarator'):   # none for an unnamed parameter
                name = _declared_name(declarator, source)
                if name and not outside_params[name]:
                    add(name, p, 'parameter')

        for decl in find_nodes(node.child_by_field_name('body'), 'declaration'):
            # A variable declared in a condition is tested by that condition.
            if decl.parent.type == 'condition_clause' or _has_attribute(decl, source) or _is_volatile(decl, source):
                continue
            scope_ids = find_nodes(decl.parent, 'identifier')
            decl_type = decl.child_by_field_name('type')
            # shortcut: class-typed locals are skipped (their constructor/destructor may matter);
            # upgrade with a trivially-constructible type check.
            scalar_type = decl_type is not None and decl_type.type in _SCALAR_TYPES
            for declarator in decl.children_by_field_name('declarator'):
                name = _declared_name(declarator, source)
                scalar = scalar_type or get_text(declarator, source).lstrip().startswith(('*', '&'))
                if name and scalar and not any(
                        n.start_byte >= declarator.end_byte and get_text(n, source) == name for n in scope_ids):
                    value = declarator.child_by_field_name('value') if declarator.type == 'init_declarator' else None
                    add(name, declarator, 'local', _side_effects(value, source, volatile_names))
    return found


def _is_volatile(decl: Node, source: bytes) -> bool:
    return any(c.type == 'type_qualifier' and get_text(c, source) == 'volatile' for c in decl.children)


def _volatile_names(root: Node, source: bytes) -> Set[str]:
    """Names declared ``volatile`` anywhere in the file (reading one is an observable access)."""
    names: Set[str] = set()
    for decl in find_nodes(root, 'declaration', 'field_declaration'):
        if _is_volatile(decl, source):
            names.update(n for n in (_declared_name(d, source) for d in decl.children_by_field_name('declarator')) if n)
    return names


def _side_effects(node: Optional[Node], source: bytes, volatile_names: Set[str]) -> List[str]:
    """Parts of an initializer that do more than compute a value: calls, writes (``x++``, ``x = 1``,
    ``x += 2``), pointer dereferences, volatile reads."""
    if node is None:
        return []
    text = get_text(node, source).strip()
    if node.type in ('call_expression', 'update_expression', 'assignment_expression') or (
            node.type == 'pointer_expression' and text.startswith('*')) or (
            node.type == 'field_expression' and any(c.type == '->' for c in node.children)):
        return [text]
    if node.type == 'identifier' and text in volatile_names:
        return [f"volatile access to {text}"]
    return [e for child in node.children for e in _side_effects(child, source, volatile_names)]


def _has_attribute(node: Node, source: bytes) -> bool:
    """``[[maybe_unused]]`` or ``__attribute__((unused))`` on a declaration."""
    return any(c.type in ('attribute_declaration', 'attribute_specifier') and 'unused' in get_text(c, source)
               for c in node.children)


_DECLARATOR_PARENTS = ('init_declarator', 'declaration', 'field_declaration', 'parameter_declaration',
                       'optional_parameter_declaration', 'pointer_declarator', 'array_declarator',
                       'reference_declarator')


def declares_name(root: Node, start: int, end: int) -> bool:
    """The identifier at byte range [start, end) names a variable, parameter, field or enumerator being
    declared (``int total, count;``, ``enum { idle, busy }``), not a use of the name."""
    node = root.descendant_for_byte_range(start, end)
    parent = node.parent if node is not None else None
    if parent is None or node.type not in ('identifier', 'field_identifier'):
        return False
    if parent.type == 'enumerator':
        return parent.child_by_field_name('name') == node
    if parent.type == 'reference_declarator':   # no field names here: the declared name is the last child
        return bool(parent.named_children) and parent.named_children[-1] == node
    return parent.type in _DECLARATOR_PARENTS and any(d == node for d in parent.children_by_field_name('declarator'))
