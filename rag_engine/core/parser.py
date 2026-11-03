from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node
from loguru import logger

from rag_engine.models import (
    ClassDef, FieldDef, FunctionDef, MethodDecl, Parameter, ParseResult, make_sig_key,
)
from rag_engine.core.ast_builder import (
    get_text, find_nodes, find_child, constructed_by_arguments, declared_type_names, _declared_name,
    compute_cyclomatic_complexity, compute_nesting_depth, declared_names, extract_data_access,
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
        functions = self._extract_functions(root, source, file_path)
        objects, inits = self._file_scope(root, source, lambda: declared_type_names(root, source),
                                          {fn.name.rsplit('::', 1)[-1] for fn in functions})
        return ParseResult(
            file_path=file_path,
            functions=functions,
            classes=self._extract_class_names(root, source),
            includes=self._extract_includes(root, source),
            raw_source=source,
            class_defs=self._extract_class_defs(root, source, file_path),
            file_scope_objects=objects,
            static_inits=inits,
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
        # Group by sig_key (name + parameter types) so overloads stay separate entries; several
        # definitions of one sig_key are the alternatives of a conditional compilation.
        groups: Dict[str, List[Tuple[FunctionDef, bool]]] = {}
        for node in find_nodes(root, 'function_definition'):
            fn = self._parse_function(node, source, file_path)
            if fn:
                groups.setdefault(fn.sig_key, []).append((fn, self._conditionally_compiled(node)))
        return [self._pick_definition(alts) for alts in groups.values()]

    @staticmethod
    def _pick_definition(alts: List[Tuple[FunctionDef, bool]]) -> FunctionDef:
        """The definition the build uses for one sig_key.

        Without conditional compilation the first wins (a ``const`` overload shares the key).  Under
        ``#if``: definitions excluded by a constant-false condition lose to any other; if several may
        be compiled (``#ifdef X`` / ``#else``) the first is kept, the others' bodies and data access
        are merged into it and its preproc_guard names the alternatives (``X | !X``).
        """
        first = alts[0][0]
        if len(alts) == 1 or not any(conditional for _, conditional in alts):
            return first
        live = [fn for fn, _ in alts if not fn.never_built] or [first]
        keep = live[0]
        if len(live) > 1:
            for other in live[1:]:
                keep.extra_bodies.append((other.body_start_line, other.body))
                keep.extra_bodies += other.extra_bodies
                keep.data_reads |= other.data_reads
                keep.data_writes |= other.data_writes
                keep.local_names |= other.local_names
            guards = [fn.preproc_guard for fn in live]
            keep.preproc_guard = (f"{guards[0]} | !{guards[0]}" if len(live) == 2 and guards[0] and not guards[1]
                                  else ' | '.join(g or 'else' for g in guards))
        return keep

    @staticmethod
    def _conditionally_compiled(node: Node) -> bool:
        """Inside an ``#if``/``#ifdef``/``#elif``/``#else`` branch that is not just an include guard."""
        parent = node.parent
        while parent is not None:
            if parent.type in ('preproc_if', 'preproc_elif', 'preproc_else') or (
                    parent.type == 'preproc_ifdef' and (parent.children[0].type == '#ifdef' or any(
                        c.type in ('preproc_else', 'preproc_elif') for c in parent.children))):
                return True
            parent = parent.parent
        return False

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
                             'virtual', 'field_initializer_list'):
                    ctext = get_text(child, source).strip()
                    if ctext and ctext not in ('virtual', 'inline', 'static', 'explicit', 'constexpr'):
                        ret_type = ctext
                        break
            ret_type = f"{ret_type} {self._return_declarators(node, source)}".strip()

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
                is_extern_c=self._in_extern_c(node, source),
                local_names=declared_names(body_node, source) if body_node else set(),
                extra_bodies=self._initialiser_arguments(node, source),
            )
        except Exception as exc:
            logger.debug(f"parse_function error: {exc}")
            return None

    @staticmethod
    def _return_declarators(node: Node, source: bytes) -> str:
        """The ``*``/``&`` (and ``const``) written on the declarator of ``Foo* f()``: part of the return type."""
        text = ''
        declarator = node.child_by_field_name('declarator')
        while declarator is not None and declarator.type in ('pointer_declarator', 'reference_declarator'):
            text += get_text(declarator.children[0], source).strip()
            text += ''.join(f" {get_text(c, source).strip()} " for c in declarator.children if c.type == 'type_qualifier')
            declarator = declarator.child_by_field_name('declarator') or next(iter(declarator.named_children), None)
        return ' '.join(text.split())

    @staticmethod
    def _initialiser_arguments(node: Node, source: bytes) -> List[Tuple[int, str]]:
        """``(line, text)`` of each argument list in a constructor's ``: a_(f()), Base(g())`` list; the
        member names are left out so ``size(size)`` does not read as a call of ``size()``."""
        init_list = find_child(node, 'field_initializer_list')
        found: List[Tuple[int, str]] = []
        for init in (init_list.named_children if init_list is not None else []):
            args = init.named_children[-1] if init.type == 'field_initializer' and init.named_children else None
            if args is not None and args.type in ('argument_list', 'initializer_list'):
                found.append((args.start_point[0] + 1, get_text(args, source)))
        return found

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
    def _in_extern_c(node: Node, source: bytes) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.type == 'linkage_specification':
                value = parent.child_by_field_name('value')
                if value is not None and get_text(value, source).strip() == '"C"':
                    return True
            parent = parent.parent
        return False

    def _file_scope(self, root: Node, source: bytes, type_names: Callable[[], Set[str]], fn_names: Set[str],
                    ) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[int, str, Optional[str]]]]:
        """``(objects, initialisers)`` of the file and namespace scope.

        Objects are the types of the variables defined there (``static Mgr g_mgr;``); initialisers are
        the ``(line, text, namespace)`` of each variable's initialiser (``= seedValue()``, ``{make()}``).
        """
        found: List[Tuple[str, Optional[str]]] = []
        inits: List[Tuple[int, str, Optional[str]]] = []

        def walk(node: Node, scope: Optional[str], only_alternatives: bool = False) -> None:
            for child in node.children:
                if only_alternatives and child.type not in ('preproc_else', 'preproc_elif'):
                    continue
                if child.type == 'namespace_definition':
                    name = child.child_by_field_name('name')
                    walk(child, '::'.join(p for p in (scope, get_text(name, source).strip() if name else None) if p) or None)
                elif child.type in (*_PREPROC_BRANCHES, 'linkage_specification', 'declaration_list'):
                    condition = child.child_by_field_name('condition') if child.type in ('preproc_if', 'preproc_elif') else None
                    # under #if 0 only the #else / #elif branches are compiled
                    walk(child, scope, condition is not None and get_text(condition, source).strip() in ('0', 'false'))
                elif child.type == 'declaration':
                    type_node = child.child_by_field_name('type')
                    declarators = child.children_by_field_name('declarator')
                    if (type_node is not None and type_node.type in ('type_identifier', 'qualified_identifier', 'template_type')
                            and any(self._is_object_declarator(d, source, type_names, fn_names) for d in declarators)):
                        found.append((get_text(type_node, source).strip(), scope))
                    for d in declarators:
                        name = d.child_by_field_name('declarator') if d.type == 'init_declarator' else None
                        text = source[name.end_byte:d.end_byte].decode('utf8', errors='replace') if name is not None else ''
                        if '(' in text:
                            inits.append((d.start_point[0] + 1, text, scope))

        walk(root, None)
        return found, inits

    @staticmethod
    def _is_object_declarator(declarator: Node, source: bytes, type_names: Callable[[], Set[str]],
                              fn_names: Set[str]) -> bool:
        """A plain object or array of objects; not a pointer, a reference or a function declaration.

        ``Uart g(UART1_BASE);`` reads as a function declaration to tree-sitter; it is an object when
        every "parameter" is a lone name that is no type (see constructed_by_arguments) and no function
        of that name is defined.
        """
        node: Optional[Node] = declarator
        if node is not None and node.type == 'init_declarator':
            node = node.child_by_field_name('declarator')
        while node is not None and node.type == 'array_declarator':
            node = node.child_by_field_name('declarator')
        if node is not None and node.type == 'function_declarator':
            name = node.child_by_field_name('declarator')
            return (name is not None and get_text(name, source).rsplit('::', 1)[-1] not in fn_names
                    and constructed_by_arguments(node, source, type_names))
        return node is not None and node.type in ('identifier', 'qualified_identifier')

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
            'array_declarator',
            'abstract_array_declarator',
            'function_declarator',
            'abstract_function_declarator',
            'parenthesized_declarator',
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

            if find_nodes(decl_node, 'array_declarator', 'abstract_array_declarator',
                          'function_declarator', 'abstract_function_declarator'):
                # Arrays and function pointers: the type keeps the shape (``unsigned char []``,
                # ``void (*)(int)``) but not the names, so a declaration and its definition agree.
                name = _declared_name(decl_node, source) or ''
                params.append(Parameter(name=name, type_=f"{base_type} {self._declarator_shape(decl_node, source)}".strip(),
                                        default_value=default_value))
                continue

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

    def _declarator_shape(self, node: Node, source: bytes) -> str:
        """A declarator written without the name it declares and without parameter names:
        ``(*cb)(int value)`` -> ``(*)(int)``, ``buf[16]`` -> ``[16]``."""
        kind = node.type
        inner = node.child_by_field_name('declarator')
        inner_text = self._declarator_shape(inner, source) if inner is not None else ''
        if kind in ('function_declarator', 'abstract_function_declarator'):
            plist = node.child_by_field_name('parameters')
            types = ', '.join(p.type_ for p in self._parse_params(plist, source)) if plist is not None else ''
            return f"{inner_text}({types})"
        if kind in ('array_declarator', 'abstract_array_declarator'):
            size = node.child_by_field_name('size')
            return f"{inner_text}[{get_text(size, source).strip() if size is not None else ''}]"
        if kind in ('parenthesized_declarator', 'abstract_parenthesized_declarator'):
            return '(' + ''.join(self._declarator_shape(c, source) for c in node.named_children if c.type != 'comment') + ')'
        if kind in ('pointer_declarator', 'abstract_pointer_declarator', 'reference_declarator',
                    'abstract_reference_declarator'):
            quals = ''.join(' ' + get_text(c, source).strip() for c in node.children if c.type == 'type_qualifier')
            if inner is None:   # an abstract declarator keeps its nested declarator among the named children
                inner_text = ''.join(self._declarator_shape(c, source) for c in node.named_children
                                     if c.type not in ('type_qualifier', 'comment'))
            return f"{get_text(node.children[0], source).strip()}{quals} {inner_text}".rstrip()
        return ''

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
                logger.warning(f"{file_path}:{node.start_point[0] + 1}: class not analysed ({exc})")
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
                for i, child in enumerate(member.children):
                    if member.field_name_for_child(i) == 'default_value':
                        cdef.field_inits.append((child.start_point[0] + 1, get_text(child, source)))
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
            is_static=any(c.type == 'storage_class_specifier' and get_text(c, source).strip() == 'static'
                          for c in member.children),
            parameters=parameters,
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
