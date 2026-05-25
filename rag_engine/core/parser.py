from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node
from loguru import logger

from rag_engine.models import FunctionDef, Parameter, ParseResult
from rag_engine.core.ast_builder import (
    get_text, find_nodes, find_child,
    compute_cyclomatic_complexity, compute_nesting_depth,
)

_CPP_LANGUAGE = Language(tscpp.language())


class CodeParser:
    def __init__(self) -> None:
        self._parser = Parser(_CPP_LANGUAGE)

    def parse_file(self, file_path: str) -> ParseResult:
        source = Path(file_path).read_bytes()
        tree = self._parser.parse(source)
        root = tree.root_node
        return ParseResult(
            file_path=file_path,
            functions=self._extract_functions(root, source, file_path),
            classes=self._extract_class_names(root, source),
            includes=self._extract_includes(root, source),
            raw_source=source,
        )

    def parse_directory(self, dir_path: str, pattern: str = "**/*.cpp") -> Dict[str, ParseResult]:
        results: Dict[str, ParseResult] = {}
        base = Path(dir_path)
        for glob in (pattern, "**/*.h", "**/*.hpp"):
            for p in base.glob(glob):
                if str(p) not in results:
                    try:
                        results[str(p)] = self.parse_file(str(p))
                    except Exception as exc:
                        logger.warning(f"Skip {p}: {exc}")
        return results

    def _extract_functions(self, root: Node, source: bytes, file_path: str) -> List[FunctionDef]:
        seen: set = set()
        functions: List[FunctionDef] = []
        for node in find_nodes(root, 'function_definition'):
            fn = self._parse_function(node, source, file_path)
            if fn and fn.name not in seen:
                seen.add(fn.name)
                functions.append(fn)
        return functions

    def _parse_function(self, node: Node, source: bytes, file_path: str) -> Optional[FunctionDef]:
        try:
            declarator = find_child(node, 'function_declarator')
            if not declarator:
                return None

            # Class methods use field_identifier; free functions use identifier
            name_node = find_child(
                declarator,
                'identifier', 'field_identifier', 'qualified_identifier',
                'destructor_name', 'operator_name',
            )
            if not name_node:
                return None

            name = get_text(name_node, source).strip()
            if not name:
                return None

            ret_type = 'unknown'
            for child in node.children:
                t = child.type
                if t not in ('function_declarator', 'compound_statement', 'comment',
                             'virtual_specifier', 'storage_class_specifier', 'type_qualifier',
                             'virtual'):
                    ctext = get_text(child, source).strip()
                    if ctext and ctext not in ('virtual', 'inline', 'static', 'explicit', 'constexpr'):
                        ret_type = ctext
                        break

            params_node = find_child(declarator, 'parameter_list')
            parameters = self._parse_params(params_node, source) if params_node else []

            body_node = find_child(node, 'compound_statement')
            body = get_text(body_node, source) if body_node else ''
            line_count = body.count('\n') + 1 if body else 1

            cc = compute_cyclomatic_complexity(node, source) if body_node else 1
            nd = compute_nesting_depth(node) if body_node else 0

            is_virtual = self._check_virtual(node, source)
            full_text = get_text(node, source)
            is_inline = 'inline' in full_text[:80]
            is_static = 'static' in full_text[:80]

            docstring = self._preceding_comment(node, source)

            req_trace: Optional[str] = None
            if docstring and 'DO-178C-REQ:' in docstring:
                req_trace = docstring.split('DO-178C-REQ:')[-1].strip()

            return FunctionDef(
                name=name,
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                return_type=ret_type,
                parameters=parameters,
                is_virtual=is_virtual,
                is_inline=is_inline,
                is_static=is_static,
                body=body,
                docstring=docstring,
                cyclomatic_complexity=cc,
                line_count=line_count,
                nesting_depth=nd,
                do178c_requirement_trace=req_trace,
            )
        except Exception as exc:
            logger.debug(f"parse_function error: {exc}")
            return None

    def _check_virtual(self, node: Node, source: bytes) -> bool:
        # tree-sitter represents 'virtual' keyword as a direct child of function_definition
        # with node type 'virtual' (a named node for the keyword)
        for child in node.children:
            if child.type == 'virtual':
                return True
            if child.type == 'virtual_specifier':
                return True
        return False

    def _parse_params(self, params_node: Node, source: bytes) -> List[Parameter]:
        params: List[Parameter] = []
        for child in params_node.children:
            if child.type == 'parameter_declaration':
                children = [c for c in child.children if c.type not in (',', '(', ')')]
                if len(children) >= 2:
                    type_str = get_text(children[0], source).strip()
                    name_str = get_text(children[1], source).strip()
                elif children:
                    type_str = get_text(children[0], source).strip()
                    name_str = ''
                else:
                    continue
                params.append(Parameter(name=name_str, type_=type_str))
        return params

    def _extract_class_names(self, root: Node, source: bytes) -> List[str]:
        names: List[str] = []
        for node in find_nodes(root, 'class_specifier', 'struct_specifier'):
            name_node = find_child(node, 'type_identifier')
            if name_node:
                names.append(get_text(name_node, source))
        return names

    def _extract_includes(self, root: Node, source: bytes) -> List[str]:
        includes: List[str] = []
        for node in find_nodes(root, 'preproc_include'):
            path_node = find_child(node, 'string_literal', 'system_lib_string')
            if path_node:
                raw = get_text(path_node, source)
                includes.append(raw.strip('<>"'))
        return includes

    def _preceding_comment(self, node: Node, source: bytes) -> Optional[str]:
        prev = node.prev_sibling
        if prev and prev.type == 'comment':
            return get_text(prev, source)
        return None
