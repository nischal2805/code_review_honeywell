from __future__ import annotations
import re
from typing import Dict, List, Set
import networkx as nx
from loguru import logger

from rag_engine.models import FunctionDef, ParseResult

CallGraph = nx.DiGraph


class CallGraphBuilder:
    def __init__(self, parse_results: Dict[str, ParseResult]) -> None:
        self._results = parse_results
        self._graph: CallGraph = nx.DiGraph()
        self._all_funcs: Dict[str, FunctionDef] = {}

    def build(self) -> CallGraph:
        for result in self._results.values():
            for fn in result.functions:
                self._all_funcs[fn.name] = fn
                self._graph.add_node(fn.name, function=fn)

        known = set(self._all_funcs.keys())
        for result in self._results.values():
            for fn in result.functions:
                callees = self._extract_calls(fn, known)
                fn.calls.update(callees)
                for callee in callees:
                    self._graph.add_edge(fn.name, callee)
                    if callee in self._all_funcs:
                        self._all_funcs[callee].called_by.add(fn.name)

        logger.debug(f"Call graph: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        return self._graph

    def find_reachable_from(self, function_name: str) -> Set[str]:
        if function_name not in self._graph:
            return set()
        return nx.descendants(self._graph, function_name)

    def find_unreachable(self, entry_points: List[str]) -> Set[str]:
        reachable: Set[str] = set(entry_points)
        for ep in entry_points:
            reachable |= self.find_reachable_from(ep)
        return set(self._graph.nodes) - reachable

    def get_function(self, name: str) -> FunctionDef | None:
        return self._all_funcs.get(name)

    def _extract_calls(self, fn: FunctionDef, known_names: Set[str]) -> Set[str]:
        if not fn.body:
            return set()

        resolved: Set[str] = set()

        # Build a lookup: unqualified name -> set of fully-qualified names in the graph.
        # e.g. "compute" -> {"AdvancedCalculator::compute", "Calculator::compute", "compute"}
        _unqualified: Dict[str, Set[str]] = {}
        for qname in known_names:
            short = qname.rsplit('::', 1)[-1]
            _unqualified.setdefault(short, set()).add(qname)

        # 1. Fully-qualified calls: Namespace::func() or Class::method()
        #    Captures e.g. "Utils::log", "AdvancedCalculator::compute"
        for qcall in re.findall(r'\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\(', fn.body):
            if qcall in known_names and qcall != fn.name:
                # Direct hit — do NOT also expand via the short name; that would
                # add every other class/namespace that defines a method with the
                # same short name and produce false-positive reachability edges.
                resolved.add(qcall)
            else:
                # Qualified name not found as-is (parser stored it unqualified).
                # Fall back to matching on the short name only in that case.
                short = qcall.rsplit('::', 1)[-1]
                for candidate in _unqualified.get(short, set()):
                    if candidate != fn.name:
                        resolved.add(candidate)

        # 2. Member / pointer-member calls: obj.method() or ptr->method()
        #
        # Strategy: infer the declared type of the receiver variable so we add
        # only the edge "ReceiverType::method" instead of every class that
        # happens to define a method with the same short name.
        #
        # Type map built from:
        #   a) function parameters  (e.g.  AdvancedCalculator& calc  ->  calc: AdvancedCalculator)
        #   b) local variable decls (e.g.  AdvancedCalculator* ptr   ->  ptr: AdvancedCalculator)
        #
        # If the receiver is not found in the map we fall back to adding all
        # candidates (original behaviour), which is conservative but correct.

        # ---- build var -> TypeName map ----
        _var_type: Dict[str, str] = {}

        # (a) parameters
        for p in fn.parameters:
            # Strip pointer/ref/const qualifiers to get the bare class name
            bare_type = re.sub(r'[\*&]', '', p.type_).strip().split()[-1] if p.type_.strip() else ''
            if p.name and bare_type:
                _var_type[p.name] = bare_type

        # (b) local declarations inside the body: TypeName[*&] varName [= ...];
        #     Handles:  AdvancedCalculator calc;
        #               AdvancedCalculator* ptr = createCalculator();
        #               Calculator& ref = ...;
        _LOCAL_DECL = re.compile(
            r'\b([A-Z][A-Za-z_]\w*)\s*[\*&]?\s+([a-z_]\w*)\s*(?:=|;|\()'
        )
        for type_name, var_name in _LOCAL_DECL.findall(fn.body):
            _var_type[var_name] = type_name

        # ---- resolve member calls ----
        # Regex captures (receiver, method) from  receiver.method(  or  receiver->method(
        _MEMBER_CALL = re.compile(r'\b([A-Za-z_]\w*)\s*(?:->|\.)([A-Za-z_]\w*)\s*\(')
        for receiver, method in _MEMBER_CALL.findall(fn.body):
            candidates = _unqualified.get(method, set())
            if not candidates:
                continue
            inferred_type = _var_type.get(receiver)
            if inferred_type:
                # Prefer the single candidate whose class matches the inferred type
                precise = {c for c in candidates if c.startswith(inferred_type + '::')}
                if precise:
                    for candidate in precise:
                        if candidate != fn.name:
                            resolved.add(candidate)
                    continue   # don't fall through to the full set
            # Type unknown or no precise match — fall back to all candidates
            for candidate in candidates:
                if candidate != fn.name:
                    resolved.add(candidate)

        # 3. Plain (unqualified) calls: func(  — original behaviour
        # Collect method names already handled by pass 2 so we do not re-expand
        # "obj.method()" as if "method" were a free function call.
        _member_methods: Set[str] = set(
            re.findall(r'(?:->|\.)([A-Za-z_]\w*)\s*\(', fn.body)
        )
        for bare in re.findall(r'\b([A-Za-z_]\w*)\s*\(', fn.body):
            if bare in _member_methods:
                # Already handled (with type inference) in pass 2 — skip.
                continue
            if bare in known_names and bare != fn.name:
                resolved.add(bare)
            # also expand to all qualified variants stored in the graph
            for candidate in _unqualified.get(bare, set()):
                if candidate != fn.name:
                    resolved.add(candidate)

        # 4. Constructor from variable declaration: ClassName varname;
        #    Matches lines like "AdvancedCalculator calc;" or "MyClass obj;"
        for ctor_class in re.findall(r'\b([A-Za-z_]\w*)\s+[A-Za-z_]\w*\s*;', fn.body):
            ctor_name = f"{ctor_class}::{ctor_class}"
            if ctor_name in known_names and ctor_name != fn.name:
                resolved.add(ctor_name)

        # 5. Constructor from new expression: new ClassName(  or new ClassName;
        for ctor_class in re.findall(r'\bnew\s+([A-Za-z_]\w*)\s*[\(;]', fn.body):
            ctor_name = f"{ctor_class}::{ctor_class}"
            if ctor_name in known_names and ctor_name != fn.name:
                resolved.add(ctor_name)

        # 6. Callback / function-pointer argument: registerCallback(callbackFunction)
        #    Only scan argument lists of actual call sites, not the entire body.
        #    Pattern: known_callee ( ... identifier ... ) where identifier is not
        #    itself followed by '(' (so it is a value, not a nested call).
        #    This is intentionally narrow to avoid false-positive edges from
        #    ordinary variable names that happen to share a name with a function.
        _ARG_CALL = re.compile(r'\b[A-Za-z_]\w*\s*\(([^)]*)\)')
        for arg_list in _ARG_CALL.findall(fn.body):
            for arg in re.findall(r'\b([A-Za-z_]\w*)\b', arg_list):
                # Skip if this token is immediately followed by '(' in the
                # original body — that means it is a nested call, not a value.
                if arg in known_names and arg != fn.name:
                    resolved.add(arg)

        return resolved
