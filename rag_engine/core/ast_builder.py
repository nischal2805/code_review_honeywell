from __future__ import annotations
from typing import List, Optional
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
