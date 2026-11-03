from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger

from rag_engine.core.class_hierarchy import ClassHierarchy, MethodInfo
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.models import CallSite, FunctionDef, ParseResult, VirtualChange


@dataclass
class OverrideInfo:
    function: str               # "TemperatureSensor::process()"
    class_name: str
    file_path: str              # where the override is declared
    line_number: int
    class_instantiated: bool
    reachable: bool             # reachable from the entry points in the call graph
    call_sites: List[CallSite]  # calls whose static target is this override


@dataclass
class VirtualFunctionInfo:
    function: str               # "Sensor::process()"
    class_name: str
    short_name: str
    file_path: str              # where the function is declared
    line_number: int
    is_pure: bool
    has_definition: bool
    overridden_by: List[OverrideInfo]
    call_sites: List[CallSite]  # calls whose static target is this function
    reachable: bool             # base implementation reachable (False when pure / no definition)
    requirement_trace: Optional[str]
    inheritance_tree: str = ''  # subtree rooted at the declaring class


@dataclass
class VirtualAnalysisResult:
    changes: List[VirtualChange]
    summary: Dict[str, int]
    base_virtual_count: int
    current_virtual_count: int
    inventory: List[VirtualFunctionInfo] = field(default_factory=list)
    hierarchy_trees: Dict[str, str] = field(default_factory=dict)   # root class -> tree, roots with subclasses only
    entry_points: List[str] = field(default_factory=list)


class VirtualAnalyzer:
    def __init__(self, base_results: Dict[str, ParseResult], current_results: Dict[str, ParseResult],
                 graph_builder: Optional[CallGraphBuilder] = None,
                 entry_points: Optional[List[str]] = None) -> None:
        """*graph_builder* is a builder over *current_results* (built here if it is not yet); omitted, one is made."""
        self._base = base_results
        self._current = current_results
        self._graph = graph_builder
        self._entry_points = list(entry_points) if entry_points else ['main']

    def analyze(self) -> VirtualAnalysisResult:
        gb = self._graph or CallGraphBuilder(self._current)
        if not gb.built:
            gb.build()
        if self._base is not self._current:
            ClassHierarchy(self._base)   # completes the base build's FunctionDefs (out-of-line virtuals)
        base_virtuals = self._extract_virtuals(self._base)
        current_virtuals = self._extract_virtuals(self._current)
        changes = self._compare(base_virtuals, current_virtuals)
        summary = {
            'added': sum(1 for c in changes if c.change_type == 'added'),
            'removed': sum(1 for c in changes if c.change_type == 'removed'),
            'modified': sum(1 for c in changes if c.change_type == 'modified'),
            'unchanged': sum(1 for c in changes if c.change_type == 'unchanged'),
        }
        logger.info(f"Virtual analysis: {summary}")
        hierarchy = gb.hierarchy
        return VirtualAnalysisResult(
            changes=changes, summary=summary,
            base_virtual_count=len(base_virtuals),
            current_virtual_count=len(current_virtuals),
            inventory=self._inventory(gb),
            hierarchy_trees={c: hierarchy.inheritance_tree(c) for c in sorted(hierarchy.classes)
                             if not hierarchy.bases(c) and hierarchy.derived(c)},
            entry_points=list(self._entry_points),
        )

    # ── §12 base-vs-current change classification ───────────────────────────

    def _extract_virtuals(self, results: Dict[str, ParseResult]) -> Dict[str, FunctionDef]:
        return {fn.sig_key: fn for r in results.values() for fn in r.functions if fn.is_virtual}

    def _compare(self, base: Dict[str, FunctionDef], current: Dict[str, FunctionDef]) -> List[VirtualChange]:
        # A changed parameter list shows up as one removed and one added sig_key; when a name has
        # exactly one of each, they are the same function with a new signature.
        gone = [k for k in base if k not in current]
        new = [k for k in current if k not in base]
        resigned: Dict[str, str] = {}
        for old_key in gone:
            gone_same_name = [k for k in gone if base[k].name == base[old_key].name]
            new_same_name = [k for k in new if current[k].name == base[old_key].name]
            if len(gone_same_name) == 1 and len(new_same_name) == 1:
                resigned[old_key] = new_same_name[0]

        changes: List[VirtualChange] = []
        for key in sorted(set(base) | set(current)):
            if key in resigned.values():
                continue   # reported under its old signature
            base_fn = base.get(key)
            curr_fn = current.get(resigned.get(key, key))
            if base_fn is None and curr_fn is not None:
                changes.append(VirtualChange(
                    change_type='added', function=curr_fn,
                    base_version=None, current_version=curr_fn,
                    do178c_category='Category 2',
                    reverification_scope=f'Full reverification required — new virtual function {key}',
                ))
            elif base_fn is not None and curr_fn is None:
                changes.append(VirtualChange(
                    change_type='removed', function=base_fn,
                    base_version=base_fn, current_version=None,
                    do178c_category='Category 2',
                    reverification_scope=f'Verify all callers updated — {key} removed',
                ))
            elif base_fn is not None and curr_fn is not None:
                change_type, category, scope = self._classify(base_fn, curr_fn)
                changes.append(VirtualChange(
                    change_type=change_type, function=curr_fn,
                    base_version=base_fn, current_version=curr_fn,
                    do178c_category=category, reverification_scope=scope,
                ))
        return changes

    def _classify(self, base_fn: FunctionDef, curr_fn: FunctionDef):
        if self._semantic_sig(base_fn) == self._semantic_sig(curr_fn):
            return 'unchanged', 'Category 1', 'No reverification required'
        if self._structural_sig(base_fn) == self._structural_sig(curr_fn):
            return 'modified', 'Category 1', 'Comment/formatting change only'
        return 'modified', 'Category 2', f'Signature/implementation changed — reverify callers of {curr_fn.name}'

    @staticmethod
    def _semantic_sig(fn: FunctionDef) -> str:
        parts = [fn.return_type, fn.name, ','.join(p.type_ for p in fn.parameters),
                 str(fn.cyclomatic_complexity), str(fn.line_count)]
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()

    @staticmethod
    def _structural_sig(fn: FunctionDef) -> str:
        body = re.sub(r'//[^\n]*', '', fn.body)
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
        body = re.sub(r'\s+', ' ', body).strip()
        return hashlib.sha256(f"{fn.return_type}|{fn.name}|{','.join(fn.param_types)}|{body}".encode()).hexdigest()

    # ── R1 virtual function inventory ───────────────────────────────────────

    def _inventory(self, gb: CallGraphBuilder) -> List[VirtualFunctionInfo]:
        hierarchy = gb.hierarchy
        funcs = {fn.sig_key: fn for r in self._current.values() for fn in r.functions}
        reachable = set(funcs) - gb.find_unreachable(self._entry_points)
        if funcs and not reachable:
            logger.warning(f"None of the entry points {self._entry_points} is in the call graph; "
                           "no virtual function will be reported reachable")
        sites: Dict[str, List[CallSite]] = {}
        for site in sorted(gb.call_sites, key=lambda s: (s.file_path, s.line, s.caller)):
            sites.setdefault(site.target, []).append(site)

        inventory: List[VirtualFunctionInfo] = []
        for sig_key in hierarchy.root_virtual_methods():
            info = hierarchy.methods[sig_key]
            overrides = []
            for over_key in hierarchy.overrides_of(sig_key):
                over = hierarchy.methods[over_key]
                overrides.append(OverrideInfo(
                    function=over_key, class_name=over.class_name, file_path=over.file_path,
                    line_number=over.line_number,
                    class_instantiated=over.class_name in gb.instantiated_classes,
                    reachable=over_key in reachable, call_sites=sites.get(over_key, [])))
            fn = funcs.get(sig_key)
            inventory.append(VirtualFunctionInfo(
                function=sig_key, class_name=info.class_name, short_name=info.short_name,
                file_path=info.file_path, line_number=info.line_number, is_pure=info.is_pure,
                has_definition=fn is not None, overridden_by=overrides,
                call_sites=sites.get(sig_key, []), reachable=sig_key in reachable,
                requirement_trace=(fn.do178c_requirement_trace if fn else None) or self._declared_trace(info),
                inheritance_tree=hierarchy.inheritance_tree(info.class_name)))
        return inventory

    def _declared_trace(self, info: MethodInfo) -> Optional[str]:
        """``DO-178C-REQ:`` tag in the comment right above the method's declaration."""
        result = self._current.get(info.file_path)
        if result is None:
            return None
        lines = result.raw_source.decode('utf-8', errors='replace').splitlines()
        i = info.line_number - 2
        while 0 <= i < len(lines) and lines[i].strip().startswith(('//', '/*', '*')):
            m = re.search(r'DO-178C-REQ:\s*(.+?)\s*(?:\*/)?\s*$', lines[i])
            if m:
                return m.group(1)
            i -= 1
        return None
