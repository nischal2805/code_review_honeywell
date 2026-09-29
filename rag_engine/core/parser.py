from __future__ import annotations
import re
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
        # Deduplicate by sig_key (name + parameter types) so that overloaded
        # functions – which share the same .name but differ in parameter types –
        # are kept as separate FunctionDef entries.  Previously this set stored
        # fn.name, which caused the second overload to be silently discarded.
        seen: set = set()
        functions: List[FunctionDef] = []
        for node in find_nodes(root, 'function_definition'):
            fn = self._parse_function(node, source, file_path)
            if fn and fn.sig_key not in seen:
                seen.add(fn.sig_key)
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

            # Prefix with enclosing namespace(s) when the function is defined
            # inside a namespace block.  Tree-sitter does not add the namespace
            # prefix to the function's own identifier node; instead the
            # function_definition sits inside a namespace_definition whose
            # namespace_identifier gives us the name.
            #
            # Walk up the ancestor chain and prepend each namespace name found,
            # stopping at class/struct bodies (which already appear as the
            # qualified_identifier "ClassName::method").
            name = self._qualify_with_namespace(node, name, source)

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

    def _qualify_with_namespace(self, node: Node, name: str, source: bytes) -> str:
        """Prefix *name* with any enclosing ``namespace`` names.

        Walks the parent chain from *node* upward.  For each
        ``namespace_definition`` ancestor, prepend its identifier.  Stop when
        a ``class_specifier`` or ``struct_specifier`` is encountered (class
        scope is already represented by the ``ClassName::method`` qualified
        identifier that tree-sitter produces for out-of-line definitions).

        Also stop if the name already contains ``::`` (it was parsed as a
        qualified_identifier, meaning the scope is explicit in the source).
        """
        if '::' in name:
            # Already qualified (e.g. "Calculator::compute" from an out-of-line
            # definition) — namespace walking not needed.
            return name

        prefixes: List[str] = []
        parent = node.parent
        while parent is not None:
            if parent.type in ('class_specifier', 'struct_specifier'):
                # Inside a class body — the class name will already appear as
                # a qualifier when tree-sitter parses out-of-line definitions,
                # so stop here.
                break
            if parent.type == 'namespace_definition':
                ns_id = find_child(parent, 'namespace_identifier')
                if ns_id:
                    prefixes.append(get_text(ns_id, source).strip())
            parent = parent.parent

        if prefixes:
            # prefixes were collected innermost-first; reverse for correct order.
            prefixes.reverse()
            return '::'.join(prefixes) + '::' + name
        return name

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
        """Extract parameters from a ``parameter_list`` node.

        Tree-sitter splits a declaration like ``const std::string& msg`` into
        multiple sibling children inside ``parameter_declaration``:

            type_qualifier   → "const"
            qualified_identifier → "std::string"
            reference_declarator → "& msg"

        The old code took only ``children[0]`` as the type, producing
        ``type_="const"`` and ``name_="std::string"``, which destroyed the
        sig_key for any parameter involving ``const``, namespaced types,
        pointers, or references.

        The correct approach:
        - The **last** child is the *declarator*: ``identifier``,
          ``pointer_declarator`` (``* name``), or ``reference_declarator``
          (``& name``).  It carries the variable name.
        - All children **before** the declarator form the base type.
        - Pointer/reference symbols in the declarator belong to the type, not
          the name.
        """
        # Node types that introduce a variable name (the declarator).
        _DECLARATOR_TYPES = {
            'identifier',
            'pointer_declarator',
            'reference_declarator',
            'abstract_pointer_declarator',
            'abstract_reference_declarator',
        }

        params: List[Parameter] = []
        for child in params_node.children:
            if child.type != 'parameter_declaration':
                continue

            # Ignore punctuation nodes.
            kids = [c for c in child.children if c.type not in (',', '(', ')')]
            if not kids:
                continue

            # Identify the declarator (last child that is a declarator type).
            # Everything before it is the type.
            decl_idx = None
            for i in range(len(kids) - 1, -1, -1):
                if kids[i].type in _DECLARATOR_TYPES:
                    decl_idx = i
                    break

            if decl_idx is None:
                # No named declarator — anonymous parameter (e.g. just "int").
                type_str = ' '.join(get_text(k, source).strip() for k in kids)
                params.append(Parameter(name='', type_=type_str))
                continue

            # Reconstruct the full type from all nodes preceding the declarator.
            type_parts = [get_text(k, source).strip() for k in kids[:decl_idx]]
            base_type = ' '.join(p for p in type_parts if p)

            # Extract the variable name and any leading qualifier (* / &) from
            # the declarator node.
            decl_node = kids[decl_idx]
            decl_text = get_text(decl_node, source).strip()

            if decl_node.type in ('pointer_declarator', 'reference_declarator',
                                   'abstract_pointer_declarator',
                                   'abstract_reference_declarator'):
                # decl_text is e.g. "* ptr", "& ref", "* const ptr", "&&"
                # Split the leading qualifier(s) from the variable name.
                qualifier_match = re.match(r'^([*&\s]+)(.*)', decl_text)
                if qualifier_match:
                    qualifier = qualifier_match.group(1).strip()   # e.g. "*" or "&"
                    var_name  = qualifier_match.group(2).strip()   # e.g. "ptr"
                    # The qualifier is part of the type, not the variable name.
                    full_type = (base_type + ' ' + qualifier).strip() if qualifier else base_type
                else:
                    full_type = base_type
                    var_name  = decl_text
            else:
                # Plain identifier.
                full_type = base_type
                var_name  = decl_text

            params.append(Parameter(name=var_name, type_=full_type))
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
