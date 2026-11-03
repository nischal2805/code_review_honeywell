from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node
from loguru import logger

from rag_engine.models import (
    ClassDef, FieldDef, FunctionDef, MethodDecl, Parameter, ParseResult, make_sig_key,
)
from rag_engine.core.ast_builder import (
    get_text, find_nodes, find_child,
    compute_cyclomatic_complexity, compute_nesting_depth, extract_data_access,
)

_CPP_LANGUAGE = Language(tscpp.language())

_PREPROC_BRANCHES = ('preproc_if', 'preproc_ifdef', 'preproc_elif', 'preproc_else')
_DECLARATOR_WRAPPERS = ('pointer_declarator', 'reference_declarator', 'array_declarator', 'parenthesized_declarator')


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
            class_defs=self._extract_class_defs(root, source, file_path),
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
            declarator = self._function_declarator(node)
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

            raw_name = get_text(name_node, source).strip()
            if not raw_name:
                return None

            # Tree-sitter does not add enclosing scopes to the function's own
            # identifier node, so prefix every enclosing namespace and class.
            # In-class definitions thus become "Class::method" and out-of-line
            # ones inside a namespace block "ns::Class::method".
            scope = self._scope_chain(node, source)
            scope_names = [n for _, n in scope]
            name = '::'.join(scope_names + [raw_name])
            if scope and scope[-1][0]:
                class_name: Optional[str] = '::'.join(scope_names)
            elif '::' in raw_name:
                # Out-of-line "X::m": X is taken as the class even if it is a namespace.
                class_name = '::'.join(scope_names + [raw_name.rsplit('::', 1)[0]])
            else:
                class_name = None

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

            is_virtual, is_override, _ = self._virtual_flags(node, declarator, source)
            reads, writes = extract_data_access(body_node, source) if body_node else (set(), set())
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
                data_reads=reads,
                data_writes=writes,
                do178c_requirement_trace=req_trace,
                class_name=class_name,
                end_line_number=node.end_point[0] + 1,
                body_start_line=body_node.start_point[0] + 1 if body_node else 0,
                is_override=is_override,
                preproc_guard=self._preproc_guard(node, source),
            )
        except Exception as exc:
            logger.debug(f"parse_function error: {exc}")
            return None

    @staticmethod
    def _function_declarator(node: Node) -> Optional[Node]:
        """The function_declarator of a definition/declaration, looking through pointer or
        reference return declarators (``Foo* f()``)."""
        declarator = node.child_by_field_name('declarator')
        while declarator is not None and declarator.type in ('pointer_declarator', 'reference_declarator'):
            declarator = declarator.child_by_field_name('declarator') or next(iter(declarator.named_children), None)
        return declarator if declarator is not None and declarator.type == 'function_declarator' else None

    @staticmethod
    def _scope_chain(node: Node, source: bytes) -> List[Tuple[bool, str]]:
        """Enclosing ``(is_class, name)`` scopes of *node*, outermost first.

        Anonymous namespaces/classes contribute nothing.
        """
        chain: List[Tuple[bool, str]] = []
        parent = node.parent
        while parent is not None:
            if parent.type in ('namespace_definition', 'class_specifier', 'struct_specifier'):
                name_node = parent.child_by_field_name('name')
                if name_node is not None:
                    chain.append((parent.type != 'namespace_definition', get_text(name_node, source).strip()))
            parent = parent.parent
        chain.reverse()
        return chain

    @staticmethod
    def _virtual_flags(node: Node, declarator: Node, source: bytes) -> Tuple[bool, bool, bool]:
        """Return ``(is_virtual, is_override, is_final)``.

        Tree-sitter puts ``override``/``final`` (virtual_specifier) inside the
        function_declarator while the ``virtual`` keyword is a child of the
        declaration itself; either makes the method virtual.
        """
        specs = {get_text(c, source).strip() for c in declarator.children if c.type == 'virtual_specifier'}
        is_override, is_final = 'override' in specs, 'final' in specs
        has_keyword = any(c.type == 'virtual' for c in node.children)
        return has_keyword or is_override or is_final, is_override, is_final

    @staticmethod
    def _preproc_guard(node: Node, source: bytes) -> Optional[str]:
        """Condition of the nearest enclosing ``#if``/``#ifdef``/``#elif`` (``#ifndef`` is skipped
        so include guards do not count)."""
        # shortcut: code in an #else/#elif branch ignores the conditions it excludes (negation
        # is not built), so it may report no guard; upgrade if negated guards are needed.
        child, parent = node, node.parent
        while parent is not None:
            if child.type not in ('preproc_else', 'preproc_elif'):
                condition = None
                if parent.type in ('preproc_if', 'preproc_elif'):
                    condition = parent.child_by_field_name('condition')
                elif parent.type == 'preproc_ifdef' and parent.children[0].type == '#ifdef':
                    condition = parent.child_by_field_name('name')
                if condition is not None:
                    return get_text(condition, source).strip()
            child, parent = parent, parent.parent
        return None

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
            'variadic_declarator',
        }

        params: List[Parameter] = []
        for child in params_node.children:
            if child.type == '...':
                # C-style varargs belong to the signature: f(const char*, ...)
                params.append(Parameter(name='', type_='...'))
                continue
            if child.type not in ('parameter_declaration', 'optional_parameter_declaration',
                                  'variadic_parameter_declaration'):
                continue

            # Ignore punctuation nodes, and for ``int x = 0`` the "=" and the default
            # expression, which are not part of the type or the name.
            default_node = child.child_by_field_name('default_value')
            default_value = get_text(default_node, source).strip() if default_node is not None else None
            kids = [c for i, c in enumerate(child.children)
                    if c.type not in (',', '(', ')', '=') and child.field_name_for_child(i) != 'default_value']
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
                params.append(Parameter(name='', type_=type_str, default_value=default_value))
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
                                   'abstract_reference_declarator',
                                   'variadic_declarator'):
                # decl_text is e.g. "* ptr", "& ref", "* const ptr", "&&", "... args"
                # Split the leading qualifier(s) from the variable name.
                qualifier_match = re.match(r'^([*&.\s]+)(.*)', decl_text)
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

            params.append(Parameter(name=var_name, type_=full_type, default_value=default_value))
        return params

    def _extract_class_names(self, root: Node, source: bytes) -> List[str]:
        names: List[str] = []
        for node in find_nodes(root, 'class_specifier', 'struct_specifier'):
            name_node = find_child(node, 'type_identifier')
            if name_node:
                names.append(get_text(name_node, source))
        return names

    def _extract_class_defs(self, root: Node, source: bytes, file_path: str) -> List[ClassDef]:
        class_defs: List[ClassDef] = []
        for node in find_nodes(root, 'class_specifier', 'struct_specifier'):
            try:
                cdef = self._class_def(node, source, file_path)
            except Exception as exc:
                # A malformed class must not cost the whole file its functions.
                logger.debug(f"class_def error: {exc}")
                continue
            if cdef:
                class_defs.append(cdef)
        return class_defs

    def _class_def(self, node: Node, source: bytes, file_path: str) -> Optional[ClassDef]:
        body = node.child_by_field_name('body')
        name_node = node.child_by_field_name('name')
        if body is None or name_node is None:
            return None
        scope_names = [n for _, n in self._scope_chain(node, source)]
        clause = find_child(node, 'base_class_clause')
        cdef = ClassDef(
            name='::'.join(scope_names + [get_text(name_node, source).strip()]),
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            end_line_number=node.end_point[0] + 1,
            bases=[get_text(c, source).strip() for c in clause.named_children
                   if c.type not in ('access_specifier', 'comment')] if clause else [],
            is_struct=node.type == 'struct_specifier',
        )
        for member in self._class_members(body):
            declarator = (self._function_declarator(member)
                          if member.type in ('function_definition', 'field_declaration', 'declaration')
                          else None)
            if declarator is not None:
                method = self._method_decl(cdef.name, member, declarator, source)
                if method:
                    cdef.methods.append(method)
            elif member.type == 'field_declaration':
                cdef.fields.extend(self._field_defs(member, source))
        return cdef

    def _class_members(self, node: Node) -> Iterator[Node]:
        """Direct members of a class body, looking inside ``#if`` blocks."""
        for child in node.children:
            if child.type in _PREPROC_BRANCHES:
                yield from self._class_members(child)
            else:
                yield child

    def _method_decl(self, class_name: str, member: Node, declarator: Node, source: bytes) -> Optional[MethodDecl]:
        name_node = find_child(declarator, 'identifier', 'field_identifier', 'destructor_name', 'operator_name')
        if name_node is None:
            return None
        name = get_text(name_node, source).strip()
        params_node = find_child(declarator, 'parameter_list')
        parameters = self._parse_params(params_node, source) if params_node else []
        is_virtual, is_override, is_final = self._virtual_flags(member, declarator, source)
        default = member.child_by_field_name('default_value')
        return MethodDecl(
            name=name,
            sig_key=make_sig_key(f"{class_name}::{name}", parameters),
            line_number=member.start_point[0] + 1,
            is_virtual=is_virtual,
            is_pure=default is not None and get_text(default, source).strip() == '0',
            is_override=is_override,
            is_final=is_final,
            has_body=member.type == 'function_definition',
        )

    def _field_defs(self, member: Node, source: bytes) -> List[FieldDef]:
        base_parts: List[str] = []
        is_static = False
        for i, child in enumerate(member.children):
            if member.field_name_for_child(i) == 'declarator':
                break
            if child.type == 'storage_class_specifier':
                is_static = is_static or get_text(child, source).strip() == 'static'
            elif child.is_named and child.type != 'comment':
                base_parts.append(get_text(child, source).strip())
        base_type = ' '.join(base_parts)

        fields: List[FieldDef] = []
        for declarator in member.children_by_field_name('declarator'):
            node: Optional[Node] = declarator
            suffix = ''
            while node is not None and node.type in _DECLARATOR_WRAPPERS:
                if node.type == 'pointer_declarator':
                    suffix += '*'
                elif node.type == 'reference_declarator':
                    suffix += get_text(node.children[0], source).strip()
                node = node.child_by_field_name('declarator') or next(iter(node.named_children), None)
            if node is None or node.type not in ('field_identifier', 'identifier'):
                continue
            fields.append(FieldDef(
                name=get_text(node, source).strip(),
                type_=f"{base_type} {suffix}".strip(),
                line_number=declarator.start_point[0] + 1,
                is_static=is_static,
            ))
        return fields

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
