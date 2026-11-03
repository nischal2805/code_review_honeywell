from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from rag_engine.config import Config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.document_processor.code_reader import CodeReader
from rag_engine.document_processor.doc_reader import DocReader
from rag_engine.features.coupling_analysis import CouplingAnalyzer
from rag_engine.features.dead_code_detector import DeadCodeDetector
from rag_engine.features.sensor_analysis import SensorAnalyzer
from rag_engine.features.sensor_template import SensorTemplate, load_sensor_templates
from rag_engine.features.standards_validator import StandardsValidator
from rag_engine.features.virtual_analysis import VirtualAnalyzer
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
from rag_engine.models import FunctionDef, ParseResult
from rag_engine.reporting.checklist_filler import ChecklistFiller
from rag_engine.reporting.report_generator import ReportGenerator


@dataclass
class AnalysisResults:
    virtual: Any = None
    coupling: Any = None
    dead: Any = None
    standards: Any = None
    checklist: Any = None
    search: Any = None
    error: Optional[str] = None
    sensors: Any = None                   # None unless cfg.sensor_templates_file is set
    graph: Any = None                     # the built CallGraphBuilder
    parse_results: Dict[str, ParseResult] = field(default_factory=dict)
    report_paths: List[str] = field(default_factory=list)


ProgressFn = Callable[[str, int], None]


class InputError(ValueError):
    """The user's input is unusable (no C++ files, unreadable or invalid sensor templates): report it, no traceback."""


def _semantic_search(cfg: Config, functions: List[FunctionDef]) -> Any:
    """The indexed SemanticSearch, or None when the embedding model cannot be loaded offline."""
    try:
        from rag_engine.core.embeddings import SemanticSearch   # heavy: pulls in torch and the model
        search = SemanticSearch(model_name=cfg.embedding_model)
        search.index_functions(functions)
        return search
    except Exception as exc:   # ImportError, NameError (torch/transformers mismatch), OSError (model not cached)
        logger.warning(f"Semantic search unavailable ({type(exc).__name__}: {exc}); "
                       "continuing with keyword matching")
        return None


def _load_templates(path: str) -> List[SensorTemplate]:
    try:
        return load_sensor_templates(path)
    except (ValueError, OSError) as exc:
        raise InputError(f"Sensor templates {path}: {exc}") from exc


def _narrative_tasks(cfg: Config, r: AnalysisResults) -> Dict[str, Tuple[str, str, int]]:
    """key -> (retrieval query, structured summary, k)."""
    tasks = {
        'virtual': (
            "virtual function change reverification DO-178C category",
            f"Virtual function analysis:\n"
            f"Added: {r.virtual.summary.get('added', 0)}, "
            f"Removed: {r.virtual.summary.get('removed', 0)}, "
            f"Modified: {r.virtual.summary.get('modified', 0)}, "
            f"Unchanged: {r.virtual.summary.get('unchanged', 0)}.\n"
            f"Changes: {[f'{c.function.name} ({c.do178c_category})' for c in r.virtual.changes[:5]]}\n"
            f"Assess reverification scope required per DO-178C §12.", 8),
        'coupling': (
            "LRU control data coupling dependency interface",
            f"Coupling analysis for {len(r.coupling.lru_impacts)} LRUs.\n"
            f"Risk levels: {[(k, v.risk_level) for k, v in r.coupling.lru_impacts.items()]}\n"
            f"Assess control/data coupling impact on DAL {cfg.dal_level} certification.", 8),
        'dead_code': (
            "unreachable function structural coverage",
            f"Dead code analysis: {r.dead.dead_count} dead functions, "
            f"{r.dead.deactivated_count} deactivated, "
            f"out of {r.dead.total_functions} total.\n"
            f"Coverage impact: {r.dead.structural_coverage_impact}\n"
            f"Assess DO-178C §6.4.2.2 compliance and structural coverage risk.", 3),
        'standards': (
            "MISRA C++ DO-178C code standard naming complexity violation",
            f"Standards compliance: score {r.standards.compliance_score:.1f}%, "
            f"violations by severity: {r.standards.violations_by_severity}.\n"
            f"Top rules: {list(set(v.rule for v in r.standards.violations[:10]))}\n"
            f"Assess DO-178C §5.1 and MISRA C++ compliance status.", 5),
    }
    if r.sensors is not None:
        found = [f for f in r.sensors.findings if f.status == 'found']
        tasks['sensors'] = (
            "sensor stale corrupt debounce validation range check",
            f"Sensor analysis: {len(found)}/{len(r.sensors.findings)} templates mapped to a class.\n"
            f"Check statuses: {[(f.template.name, {k: c.status for k, c in f.checks.items()}) for f in found]}\n"
            f"Unmapped sensor classes: {r.sensors.unmapped_sensor_classes}\n"
            f"Assess stale, corrupt and debounce handling for DAL {cfg.dal_level}.", 5)
    return tasks


def run_pipeline(cfg: Config, codebase: str, base_build: Optional[str] = None, lru_docs: Optional[str] = None,
                 output_format: str = 'markdown', progress: Optional[ProgressFn] = None,
                 use_semantic_search: bool = True, write_reports: bool = True) -> AnalysisResults:
    """Parse, analyze and (optionally) report. Raises InputError when *codebase* has no C++ files or the
    sensor template file is unreadable or invalid."""
    def step(name: str, pct: int) -> None:
        logger.info(name)
        if progress:
            progress(name, pct)

    r = AnalysisResults()
    # Bad templates are the cheapest mistake to report, so check them before parsing.
    templates = _load_templates(cfg.sensor_templates_file) if cfg.sensor_templates_file else None

    step('Parsing C++ files', 5)
    reader = CodeReader(cache_dir=cfg.cache_dir)
    current = reader.read_directory(codebase)
    if not current:
        raise InputError(f"No C++ files found in {codebase}")
    base = reader.read_directory(base_build) if base_build else current
    r.parse_results = current

    step('Building call graph', 15)
    gb = CallGraphBuilder(current)
    gb.build()
    r.graph = gb
    if use_semantic_search:
        r.search = _semantic_search(cfg, reader.all_functions(current))
    lru_doc_texts = DocReader().read_lru_documents(lru_docs) if lru_docs else {}

    step('Virtual function analysis', 30)
    r.virtual = VirtualAnalyzer(base, current, graph_builder=gb, entry_points=cfg.entry_points,
                                entry_point_patterns=cfg.entry_point_patterns).analyze()
    step('Dead code detection', 45)
    r.dead = DeadCodeDetector(gb, current, cfg.entry_points, cfg.entry_point_patterns,
                              cfg.dead_code_dispositions or None).analyze()
    step('Standards validation', 60)
    r.standards = StandardsValidator(cfg, current).analyze()
    step('Coupling analysis', 70)
    r.coupling = CouplingAnalyzer(current, gb, r.search, lru_doc_texts).analyze()
    if templates is not None:
        step('Sensor analysis', 80)
        r.sensors = SensorAnalyzer(current, gb, templates).analyze()

    llm_narratives: Dict[str, str] = {}
    llm = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)
    if write_reports and cfg.ollama_enabled and llm.is_available():
        step(f"LLM enrichment via {cfg.ollama_model}", 85)
        rag = RAGNarrativeGenerator(llm, r.search)
        for key, (query, summary, k) in _narrative_tasks(cfg, r).items():
            try:
                narrative = rag.generate(query, summary, k=k, max_tokens=800)
                if narrative:
                    llm_narratives[key] = narrative
                    logger.info(f"  LLM narrative: {key}")
            except Exception as exc:
                logger.warning(f"  LLM failed for {key}: {exc}")

    if write_reports:
        step('Writing reports and checklist', 95)
        r.report_paths = ReportGenerator(cfg).generate_all(
            r.virtual, r.coupling, r.dead, r.standards, format=output_format,  # type: ignore[arg-type]
            llm_narratives=llm_narratives, sensors=r.sensors)
        r.checklist = ChecklistFiller(cfg).fill_sqa_checklist(r.virtual, r.dead, r.standards, sensors=r.sensors)
        r.report_paths.append(str(Path(cfg.output_dir) / 'sqa_checklist.md'))
    step('Done', 100)
    return r
