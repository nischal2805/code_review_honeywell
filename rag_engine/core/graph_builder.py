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
        call_pattern = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
        candidates = set(call_pattern.findall(fn.body))
        return {c for c in candidates if c in known_names and c != fn.name}
