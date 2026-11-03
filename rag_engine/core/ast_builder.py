from __future__ import annotations
from typing import List, Optional, Set, Tuple
from tree_sitter import Node


def get_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode('utf8', errors='replace')


def find_nodes(root: Node, *type_names: str) -> List[Node]:
    results: List[Node] = []

    def walk(node: Node) -> None:
        if node.type in type_names:
            results.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return results


def find_child(node: Node, *type_names: str) -> Optional[Node]:
    for child in node.children:
        if child.type in type_names:
            return child
    return None


def compute_cyclomatic_complexity(func_node: Node, source: bytes) -> int:
    branching = {
        'if_statement', 'while_statement', 'for_statement',
        'for_range_loop', 'case_statement', 'catch_clause',
        'conditional_expression',
    }
    count = 1
    for node in find_nodes(func_node, *branching):
        count += 1
    for node in find_nodes(func_node, 'binary_expression'):
        txt = get_text(node, source)
        count += txt.count('&&') + txt.count('||')
    return count


def compute_nesting_depth(func_node: Node) -> int:
    scoping = {
        'compound_statement', 'if_statement', 'while_statement',
        'for_statement', 'for_range_loop', 'switch_statement',
        'try_statement',
    }

    def depth(node: Node, current: int) -> int:
        if node.type in scoping:
            current += 1
        if not node.children:
            return current
        return max(depth(child, current) for child in node.children)

    return max(0, depth(func_node, 0) - 1)


def extract_identifiers(node: Node, source: bytes) -> List[str]:
    return [get_text(n, source) for n in find_nodes(node, 'identifier')]


def _declared_name(declarator: Node, source: bytes) -> Optional[str]:
    """Name introduced by a declarator, unwrapping pointer/reference/array/init wrappers."""
    node: Optional[Node] = declarator
    while node is not None and node.type not in ('identifier', 'field_identifier'):
        inner = node.child_by_field_name('declarator')
        if inner is None and node.named_children:
            inner = node.named_children[0]
        node = inner
    return get_text(node, source) if node is not None else None


def declared_names(body: Node, source: bytes) -> Set[str]:
    """Names of the variables declared anywhere inside *body* (parameters of the function itself excluded)."""
    names: Set[str] = set()
    for decl_node in find_nodes(body, 'declaration', 'for_range_loop', 'parameter_declaration'):
        for declarator in decl_node.children_by_field_name('declarator'):
            name = _declared_name(declarator, source)
            if name:
                names.add(name)
    return names


def extract_data_access(body: Node, source: bytes) -> Tuple[Set[str], Set[str]]:
    """Return ``(reads, writes)``: the names a function body reads and modifies.

    Writes are the base name of an assignment target or ``++``/``--`` operand
    (``this->x`` -> ``x``; ``obj.f``/``p->f`` -> ``obj``/``p``; ``a[i]`` -> ``a``; ``*p`` -> ``p``).
    Compound assignment and ``++``/``--`` also read.  A call receiver is read; the callee
    name is not.  Variables declared inside the body are excluded, parameters are not.
    """
    # shortcut: locals are collected function-wide, so a local shadowing a member or global
    # hides that access; upgrade to per-scope tracking if shadowing matters.
    local = declared_names(body, source)

    reads: Set[str] = set()
    writes: Set[str] = set()

    def record(name: str, also_read: bool) -> None:
        if name not in local:
            writes.add(name)
            if also_read:
                reads.add(name)

    def target(node: Optional[Node], also_read: bool) -> None:
        if node is None:
            return
        kind = node.type
        if kind == 'identifier':
            record(get_text(node, source), also_read)
        elif kind == 'qualified_identifier':
            target(node.child_by_field_name('name'), also_read)
        elif kind == 'field_expression':
            owner = node.child_by_field_name('argument')
            if owner is not None and owner.type == 'this':
                target(node.child_by_field_name('field'), also_read)
            else:
                target(owner, also_read)
        elif kind == 'field_identifier':
            record(get_text(node, source), also_read)
        elif kind == 'subscript_expression':
            target(node.child_by_field_name('argument'), also_read)
            visit(node.child_by_field_name('indices'))
        elif kind in ('pointer_expression', 'parenthesized_expression'):
            inner = node.child_by_field_name('argument')
            target(inner if inner is not None else (node.named_children or [None])[0], also_read)
        else:
            visit(node)

    def visit(node: Optional[Node]) -> None:
        if node is None:
            return
        kind = node.type
        if kind == 'assignment_expression':
            operator = node.child_by_field_name('operator')
            target(node.child_by_field_name('left'), operator is not None and get_text(operator, source) != '=')
            visit(node.child_by_field_name('right'))
        elif kind == 'update_expression':
            target(node.child_by_field_name('argument'), True)
        elif kind == 'call_expression':
            callee = node.child_by_field_name('function')
            if callee is not None and callee.type == 'field_expression':
                visit(callee.child_by_field_name('argument'))
            elif callee is not None and callee.type not in ('identifier', 'qualified_identifier', 'template_function'):
                visit(callee)
            visit(node.child_by_field_name('arguments'))
        elif kind == 'field_expression':
            owner = node.child_by_field_name('argument')
            field_node = node.child_by_field_name('field')
            visit(owner)
            if owner is not None and owner.type == 'this' and field_node is not None:
                name = get_text(field_node, source)
                if name not in local:
                    reads.add(name)
        elif kind == 'identifier':
            name = get_text(node, source)
            if name not in local:
                reads.add(name)
        else:
            for child in node.children:
                visit(child)

    visit(body)
    return reads, writes
