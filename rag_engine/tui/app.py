from __future__ import annotations
import os
import json
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')

import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, DataTable, Footer, Header, Input, Label,
    ProgressBar, RichLog, Select, Static,
    TabbedContent, TabPane, Tree,
)
from textual.widgets.tree import TreeNode
from textual.worker import get_current_worker

from rag_engine.config import Config, load_config
from rag_engine.features.function_effect import FunctionEffectAnalyzer, format_effect
from rag_engine.features.sensor_data_checker import (
    Sample, SampleResult, check_samples, load_samples_csv, parse_inline_samples, summarize,
)
from rag_engine.features.sensor_template import SensorTemplate
from rag_engine.knowledge_base.standards_profile import load_raw_standards_profile
from rag_engine.llm.offline_qa import answer_question, function_names, not_found_message
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
from rag_engine.pipeline import AnalysisResults, InputError, run_pipeline   # AnalysisResults: screens and tests import it from here


_stderr_sink = 0   # loguru's default stderr sink; a new id once the TUI has put it back


def _t(value: Any, style: str = '') -> Text:
    """Raw analysis text as a Text: never parsed as markup, so `a[i]` or `[-54, 54]` stay literal."""
    return Text(str(value), style=style)


def _mark(ok: bool) -> Text:
    return Text('✓', style='green') if ok else Text('✗', style='red')


# ─────────────────────────────────────────────────────────────────────────────
# Config screen
# ─────────────────────────────────────────────────────────────────────────────

class ConfigScreen(Screen):
    CSS = """
    ConfigScreen {
        align: center middle;
    }
    #config-panel {
        width: 70;
        height: auto;
        max-height: 100%;
        overflow-y: auto;
        border: solid $primary;
        padding: 1 2;
    }
    #config-panel Label {
        margin-top: 1;
    }
    #config-panel Input {
        margin-bottom: 1;
    }
    #btn-row {
        margin-top: 1;
        align: center middle;
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="config-panel"):
            yield Label("DO-178C RAG Engine", id="title")
            yield Static("Codebase directory (required)")
            yield Input(placeholder="/path/to/cpp/codebase", id="codebase")
            yield Static("Base build directory (optional — for virtual analysis diff)")
            yield Input(placeholder="/path/to/base_build", id="base-build")
            yield Static("LRU documents directory (optional)")
            yield Input(placeholder="/path/to/lru_docs", id="lru-docs")
            yield Static("Standards profile file (optional, YAML/JSON/PDF/DOCX/TXT)")
            yield Input(placeholder="/path/to/standards.yaml", id="standards-file")
            yield Static("Sensor templates file (optional — default: sensor_templates_file in config.yaml)")
            yield Input(placeholder="/path/to/sensor_templates.yaml", id="sensor-templates")
            yield Checkbox("Use semantic search (needs sentence-transformers)", value=True, id="semantic-search")
            yield Static("Output format")
            yield Select(
                [("DOCX (Word)", "docx"), ("Markdown", "markdown")],
                value="markdown", id="output-format",
            )
            yield Static("DAL level")
            yield Select(
                [("DAL A (MC/DC)", "A"), ("DAL B (Decision)", "B"),
                 ("DAL C (Statement)", "C"), ("DAL D", "D")],
                value="B", id="dal-level",
            )
            with Horizontal(id="btn-row"):
                yield Button("Preview Standards", variant="default", id="btn-preview")
                yield Button("Run Analysis", variant="primary", id="btn-run")
                yield Button("Quit", variant="error", id="btn-quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-quit":
            self.app.exit()
        elif event.button.id == "btn-preview":
            standards_file = self.query_one("#standards-file", Input).value.strip()
            if not standards_file:
                self.notify("Standards file is required for preview", severity="error")
                return
            path = Path(standards_file).expanduser()
            if not path.exists():
                self.notify(f"Standards file not found: {standards_file}", severity="error", markup=False)
                return
            try:
                profile = load_raw_standards_profile(path)
            except Exception as exc:
                self.notify(f"Failed to load standards profile: {exc}", severity="error", markup=False)
                return
            self.app.push_screen(StandardsPreviewScreen(str(path), profile))
        elif event.button.id == "btn-run":
            codebase = self.query_one("#codebase", Input).value.strip()
            if not codebase:
                self.notify("Codebase path is required", severity="error")
                return
            if not Path(codebase).exists():
                self.notify(f"Path not found: {codebase}", severity="error", markup=False)
                return
            base_build = self.query_one("#base-build", Input).value.strip() or None
            lru_docs = self.query_one("#lru-docs", Input).value.strip() or None
            standards_file = self.query_one("#standards-file", Input).value.strip() or None
            sensor_templates = self.query_one("#sensor-templates", Input).value.strip() or None
            output_fmt = self.query_one("#output-format", Select).value
            dal = self.query_one("#dal-level", Select).value
            if standards_file and not Path(standards_file).expanduser().exists():
                self.notify(f"Standards file not found: {standards_file}", severity="error", markup=False)
                return
            if sensor_templates and not Path(sensor_templates).expanduser().exists():
                self.notify(f"Sensor templates file not found: {sensor_templates}", severity="error", markup=False)
                return
            semantic_search = self.query_one("#semantic-search", Checkbox).value
            self.app.push_screen(AnalysisScreen(codebase, base_build, lru_docs, standards_file, str(output_fmt),
                                                str(dal), sensor_templates, semantic_search=semantic_search))


def _format_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, sort_keys=False)


class StandardsPreviewScreen(Screen):
    CSS = """
    StandardsPreviewScreen {
        align: center middle;
    }
    #preview-panel {
        width: 100;
        height: 34;
        border: solid $primary;
        padding: 1 2;
    }
    #preview-title {
        margin-bottom: 1;
    }
    #preview-log {
        height: 1fr;
        border: solid $surface;
        margin: 1 0;
    }
    #preview-buttons {
        height: 3;
        align: right middle;
        margin-top: 1;
    }
    """

    def __init__(self, standards_path: str, profile: dict[str, Any]) -> None:
        super().__init__()
        self._standards_path = standards_path
        self._profile = profile

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="preview-panel"):
            yield Static(f"Structured standards profile extracted from {self._standards_path}",
                         id="preview-title", markup=False)
            yield RichLog(id="preview-log", highlight=False, markup=False, wrap=True)
            with Horizontal(id="preview-buttons"):
                yield Button("Back", variant="primary", id="btn-back")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#preview-log", RichLog)
        log.write(_format_profile(self._profile))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis / progress screen
# ─────────────────────────────────────────────────────────────────────────────

# (progress bar id, label, prefixes of the run_pipeline step names that belong to it)
_STAGES = [
    ("pb-parse", "Parsing C++ files", ("pars", "build")),
    ("pb-virtual", "Virtual analysis", ("virtual",)),
    ("pb-dead", "Dead code detection", ("dead",)),
    ("pb-standards", "Standards validation", ("standards",)),
    ("pb-coupling", "Coupling analysis", ("coupling",)),
    ("pb-sensors", "Sensor analysis", ("sensor",)),
]


def stage_progress(step: str) -> Optional[List[int]]:
    """Percent done of each _STAGES row once pipeline step *step* starts.

    None for a step that has no row (LLM enrichment, writing reports, Done): the bars stay as they are."""
    key = step.lower()
    current = next((i for i, (_, _, prefixes) in enumerate(_STAGES) if key.startswith(prefixes)), None)
    if current is None:
        return None
    return [100 if i < current else 50 if i == current else 0 for i in range(len(_STAGES))]


class AnalysisScreen(Screen):
    BINDINGS = [Binding("escape", "go_back", "Back")]
    CSS = """
    AnalysisScreen {
        align: center middle;
    }
    #progress-panel {
        width: 72;
        height: auto;
        max-height: 100%;
        border: solid $primary;
        padding: 1 2;
    }
    .task-row {
        height: 1;
    }
    .task-label {
        width: 26;
    }
    ProgressBar {
        width: 1fr;
    }
    #log-box {
        height: 8;
        border: solid $surface;
        margin-top: 1;
    }
    """

    def __init__(self, codebase: str, base_build: Optional[str], lru_docs: Optional[str],
                 standards_file: Optional[str], output_fmt: str, dal: str,
                 sensor_templates: Optional[str] = None, cfg: Optional[Config] = None,
                 semantic_search: bool = True) -> None:
        super().__init__()
        self._semantic_search = semantic_search
        self._codebase = codebase
        self._base_build = base_build
        self._lru_docs = lru_docs
        self._standards_file = standards_file
        self._output_fmt = output_fmt
        self._dal = dal
        self._sensor_templates = sensor_templates
        self._cfg = cfg
        self._results = AnalysisResults()
        self._finished = False
        self._sensors_skipped = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="progress-panel"):
            yield Static(f"Analyzing: {self._codebase}", id="analysis-title", markup=False)
            for bar_id, label, _ in _STAGES:
                with Horizontal(classes="task-row"):
                    yield Label(label, classes="task-label", id=f"label-{bar_id}")
                    yield ProgressBar(total=100, show_eta=False, id=bar_id)
            yield RichLog(id="log-box", highlight=False, markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._run_analysis, exclusive=True, thread=True)

    # Runs on the app thread.
    def _write(self, msg: str | Text) -> None:
        self.query_one("#log-box", RichLog).write(msg)

    def _set_bars(self, percents: List[int]) -> None:
        for (bar_id, _, _), percent in zip(_STAGES, percents):
            if bar_id == "pb-sensors" and self._sensors_skipped:
                continue
            self.query_one(f"#{bar_id}", ProgressBar).update(progress=percent)

    def _show_progress(self, step: str) -> None:
        self._write(step)
        percents = stage_progress(step)
        if percents is not None:
            self._set_bars(percents)

    def _skip_sensors(self) -> None:
        self._sensors_skipped = True
        self.query_one("#label-pb-sensors", Label).update("Sensor analysis (skipped)")
        self._write("No sensor templates configured: sensor analysis skipped.")

    def _finish(self, results: AnalysisResults, cfg: Config) -> None:
        self._results = results
        self._set_bars([100] * len(_STAGES))
        functions = sum(len(r.functions) for r in results.parse_results.values())
        self._write(f"Parsed {len(results.parse_results)} files, {functions} functions.")
        if results.virtual is not None:
            s = results.virtual.summary
            self._write(f"Virtual: {len(results.virtual.inventory)} functions; changes: added {s['added']}, "
                        f"modified {s['modified']}, unchanged {s['unchanged']}.")
        if results.dead is not None:
            self._write(f"Dead code: {results.dead.dead_count} dead, {results.dead.deactivated_count} deactivated, "
                        f"{len(results.dead.items)} findings.")
        if results.standards is not None:
            self._write(f"Standards: score {results.standards.compliance_score:.1f}%, "
                        f"{results.standards.violations_by_severity.get('CRITICAL', 0)} CRITICAL.")
        if results.sensors is not None:
            found = sum(f.status == 'found' for f in results.sensors.findings)
            self._write(f"Sensors: {found}/{len(results.sensors.findings)} templates mapped to a class.")
        if self._semantic_search and results.search is None:
            self._write("Semantic search unavailable: keyword matching used for coupling analysis.")
        self._write(Text(f"Analysis complete. Reports written to {cfg.output_dir}", style="bold green"))
        self._finished = True
        self.app.push_screen(ResultsScreen(results, cfg.output_dir, cfg))

    def _show_failure(self, message: str) -> None:
        self._results.error = message
        self._finished = True
        self._write(Text(f"ERROR: {message}", style="bold red"))
        self.notify(f"Analysis failed: {message}", severity="error", timeout=10, markup=False)

    # Runs on the worker thread.
    def _run_analysis(self) -> None:
        def progress(step: str, pct: int) -> None:
            self.app.call_from_thread(self._show_progress, step)

        try:
            cfg = self._cfg or load_config()
            cfg.dal_level = self._dal  # type: ignore[assignment]
            if self._standards_file:
                cfg.standards_file = self._standards_file
            if self._sensor_templates:
                cfg.sensor_templates_file = self._sensor_templates
            if not cfg.sensor_templates_file:
                self.app.call_from_thread(self._skip_sensors)
            if not self._semantic_search:
                self.app.call_from_thread(self._write, "Semantic search is off: keyword matching is used for coupling analysis.")
            results = run_pipeline(cfg, self._codebase, self._base_build, self._lru_docs, self._output_fmt,
                                   progress=progress, use_semantic_search=self._semantic_search)
        except InputError as exc:
            self.app.call_from_thread(self._show_failure, str(exc))
        except Exception as exc:
            logger.exception("TUI analysis failed")
            self.app.call_from_thread(self._show_failure, f"{type(exc).__name__}: {exc}")
        else:
            self.app.call_from_thread(self._finish, results, cfg)

    def action_go_back(self) -> None:
        if self._finished:
            self.app.pop_screen()
        else:
            self.notify("Analysis is still running", markup=False)


# ─────────────────────────────────────────────────────────────────────────────
# Results browser screen
# ─────────────────────────────────────────────────────────────────────────────

def _lines(start: int, end: int) -> str:
    return str(start) if end in (0, start) else f"{start}-{end}"


SAMPLE_PREVIEW = 200   # sample tester rows shown besides the flagged ones
SAMPLE_CHUNK = 2500    # rows handed to the table per UI-thread call
_WARNINGS_SHOWN = 5


def visible_samples(results: List[SampleResult], first: int = SAMPLE_PREVIEW) -> List[SampleResult]:
    """The first *first* samples plus every flagged one, in time order: the flagged ones are why the tester is used."""
    return [res for i, res in enumerate(results) if i < first or res.flags]


def sample_row(res: SampleResult) -> tuple:
    style = "red" if "corrupt" in res.flags else "yellow" if res.flags else "green"
    return (_t(f"{res.timestamp_ms:.10g}"), _t(f"{res.value:.10g}"),
            _t("" if res.accepted_value is None else f"{res.accepted_value:.10g}"),
            _t(", ".join(res.flags) or "ok", style))


_CHECK_COLORS = {'implemented': 'green', 'mismatch': 'yellow', 'missing': 'red'}


class ResultsScreen(Screen):
    BINDINGS = [
        Binding("c", "chat", "Ask AI"),
        Binding("r", "rerun", "Re-run"),
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
    ]
    CSS = """
    ResultsScreen {
        layout: vertical;
    }
    TabbedContent {
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    #summary-bar {
        height: 3;
        background: $surface;
        padding: 0 2;
        content-align: left middle;
    }
    #virtual-tree {
        height: 2fr;
    }
    #dead-info, #sensor-hint, #sample-summary, #effect-help {
        height: auto;
        padding: 0 1;
    }
    #dead-info {
        max-height: 10;
    }
    #effect-row, #sample-row {
        height: 3;
    }
    #effect-input, #sample-input {
        width: 1fr;
    }
    #sensor-select-label {
        height: 3;
        width: auto;
        padding: 0 1 0 0;
        content-align: left middle;
    }
    #sensor-select {
        width: 30;
    }
    #effect-scroll {
        height: 1fr;
        border: solid $surface;
    }
    """

    def __init__(self, results: AnalysisResults, output_dir: str, cfg: Optional[Config] = None) -> None:
        super().__init__()
        self._results = results
        self._output_dir = output_dir
        self._cfg = cfg
        self._templates: Dict[str, SensorTemplate] = (
            {f.template.name: f.template for f in results.sensors.findings} if results.sensors else {})
        self._effects: Optional[FunctionEffectAnalyzer] = None
        self._effects_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        r = self._results
        summary_parts = []
        if r.standards:
            summary_parts.append(f"Score {r.standards.compliance_score:.1f}%")  # type: ignore[union-attr]
        if r.virtual:
            s = r.virtual.summary  # type: ignore[union-attr]
            summary_parts.append(f"Virtual: {len(r.virtual.inventory)} fn, +{s['added']} ~{s['modified']}")  # type: ignore[union-attr]
        if r.dead:
            summary_parts.append(f"Dead: {r.dead.dead_count}")  # type: ignore[union-attr]
        if r.sensors:
            summary_parts.append(f"Sensors: {sum(f.status == 'found' for f in r.sensors.findings)}"  # type: ignore[union-attr]
                                 f"/{len(r.sensors.findings)} mapped")  # type: ignore[union-attr]
        yield Static("  |  ".join(summary_parts), id="summary-bar", markup=False)
        with TabbedContent("Standards", "Virtual Functions", "Dead Code", "Function Effect", "Sensors",
                           "Coupling", "Checklist"):
            with TabPane("Standards", id="tab-standards"):
                yield self._build_standards_table()
            with TabPane("Virtual Functions", id="tab-virtual"):
                yield self._build_virtual_tree()
                yield self._build_virtual_table()
            with TabPane("Dead Code", id="tab-dead"):
                yield Static(self._dead_notes(), id="dead-info")
                yield self._build_dead_table()
            with TabPane("Function Effect", id="tab-effect"):
                yield Static("Method name: what it reads, writes and calls, and what is affected if it changes",
                             id="effect-help", markup=False)
                with Horizontal(id="effect-row"):
                    yield Input(placeholder="Method name, e.g. updateTemperature", id="effect-input")
                    yield Button("Analyze", variant="primary", id="btn-effect")
                with VerticalScroll(id="effect-scroll"):
                    yield Static("", id="effect-output", markup=False)
            with TabPane("Sensors", id="tab-sensors"):
                yield self._build_sensors_table()
                yield from self._build_sample_tester()
            with TabPane("Coupling", id="tab-coupling"):
                yield self._build_coupling_table()
            with TabPane("Checklist", id="tab-checklist"):
                yield self._build_checklist_table()
        yield Footer()

    def _build_standards_table(self) -> DataTable:
        t = DataTable(id="dt-standards", zebra_stripes=True)
        t.add_columns("Severity", "Rule", "File", "Line", "Element", "Message")
        if self._results.standards:
            for v in self._results.standards.violations:  # type: ignore[union-attr]
                sev_color = {"CRITICAL": "red", "MAJOR": "yellow", "MEDIUM": "cyan", "MINOR": "white"}.get(v.severity, "white")
                t.add_row(
                    _t(v.severity, sev_color),
                    _t(v.rule), _t(Path(v.file).name), _t(v.line), _t(v.element),
                    _t(v.message[:60] + ("…" if len(v.message) > 60 else "")),
                )
        return t

    def _build_virtual_tree(self) -> Tree:
        tree: Tree = Tree("Virtual functions", id="virtual-tree")
        tree.show_root = False
        tree.root.expand()
        inventory = self._results.virtual.inventory if self._results.virtual else []  # type: ignore[union-attr]
        if not inventory:
            tree.root.add_leaf("No virtual functions found")
        by_class: Dict[str, list] = {}
        for info in inventory:
            by_class.setdefault(info.class_name, []).append(info)
        hierarchy = self._results.graph.hierarchy if self._results.graph is not None else None
        for cls in sorted(by_class):
            cls_node = tree.root.add(_t(cls, "bold"), expand=True)
            if hierarchy is not None and hierarchy.derived(cls):
                self._add_derived(cls_node.add(_t("Inheritance", "italic"), expand=True), hierarchy, cls, {cls})
            for info in by_class[cls]:
                label = _t(info.function)
                if info.is_pure:
                    label.append("  pure virtual", style="dim")
                else:
                    label.append("  reachable ")
                    label.append_text(_mark(info.reachable))
                fn_node = cls_node.add(label, expand=True)
                for over in info.overridden_by:
                    label = _t(over.function)
                    label.append("  reachable ")
                    label.append_text(_mark(over.reachable))
                    label.append("  instantiated ")
                    label.append_text(_mark(over.class_instantiated))
                    self._add_call_sites(fn_node.add(label, expand=True), over.call_sites)
                self._add_call_sites(fn_node, info.call_sites)
        return tree

    @classmethod
    def _add_derived(cls, node: TreeNode, hierarchy, base: str, path: set) -> None:
        """The subclasses of *base* as nested nodes: the Tree form of ClassHierarchy.inheritance_tree."""
        for kid in hierarchy.derived(base):
            if kid in path:   # shortcut: a parsed inheritance cycle is cut here, not reported
                continue
            kids = [k for k in hierarchy.derived(kid) if k not in path | {kid}]
            child = node.add(_t(kid), expand=True) if kids else node.add_leaf(_t(kid))
            if kids:
                cls._add_derived(child, hierarchy, kid, path | {kid})

    @staticmethod
    def _add_call_sites(node: TreeNode, sites: list) -> None:
        for s in sites:
            where = f"{Path(s.file_path).name}{':%d' % s.line if s.line else ''}"
            node.add_leaf(_t(f"called from {s.caller} ({where})", "dim"))

    def _build_virtual_table(self) -> DataTable:
        t = DataTable(id="dt-virtual", zebra_stripes=True)
        t.add_columns("Change", "Function", "Category", "File", "Line", "Reverification")
        if self._results.virtual:
            for c in self._results.virtual.changes:  # type: ignore[union-attr]
                color = {"added": "green", "removed": "red", "modified": "yellow", "unchanged": "dim"}.get(c.change_type, "white")
                t.add_row(
                    _t(c.change_type.upper(), color),
                    _t(c.function.name), _t(c.do178c_category),
                    _t(Path(c.function.file_path).name), _t(c.function.line_number),
                    _t((c.reverification_scope or "")[:55]),
                )
        return t

    def _dead_notes(self) -> Text:
        d = self._results.dead
        if d is None:
            return _t("Dead code analysis did not run.")
        counts = ", ".join(f"{k}={n}" for k, n in sorted(d.counts_by_category.items()))  # type: ignore[union-attr]
        lines = [f"Entry points: {', '.join(d.entry_points_used) or 'none'}",  # type: ignore[union-attr]
                 f"Findings: {len(d.items)}" + (f" ({counts})" if counts else "")]  # type: ignore[union-attr]
        lines += [f"Warning: {w}" for w in d.warnings[:_WARNINGS_SHOWN]]  # type: ignore[union-attr]
        if len(d.warnings) > _WARNINGS_SHOWN:  # type: ignore[union-attr]
            lines.append(f"(+{len(d.warnings) - _WARNINGS_SHOWN} more)")  # type: ignore[union-attr]
        if d.unresolved_calls:  # type: ignore[union-attr]
            lines.append(f"Unresolved calls (treated as possible callers): {len(d.unresolved_calls)}, "  # type: ignore[union-attr]
                         f"e.g. {d.unresolved_calls[0]}")  # type: ignore[union-attr]
        return _t("\n".join(lines))

    def _build_dead_table(self) -> DataTable:
        t = DataTable(id="dt-dead", zebra_stripes=True)
        t.add_columns("ID", "File", "Symbol", "Lines", "Type", "Confidence", "Reason")
        if self._results.dead:
            for item in self._results.dead.items:  # type: ignore[union-attr]
                color = "red" if item.category == "dead_code" else "yellow"
                t.add_row(
                    _t(item.finding_id), _t(Path(item.file_path).name), _t(item.name),
                    _t(_lines(item.line_number, item.end_line)),
                    _t(f"{item.category} ({item.symbol_type})", color),
                    _t(item.confidence), _t(item.reason),
                )
        return t

    def _build_sensors_table(self) -> DataTable:
        t = DataTable(id="dt-sensors", zebra_stripes=True)
        t.add_columns("Sensor", "Class", "Corrupt", "Stale", "Debounce", "Instantiated")
        if self._results.sensors:
            for f in self._results.sensors.findings:  # type: ignore[union-attr]
                if f.status != 'found':
                    t.add_row(_t(f.template.name), _t("not found", "red"), *(_t("-"),) * 4)
                    continue
                statuses = [f.checks[k].status if k in f.checks else "-" for k in ("corrupt", "stale", "debounce")]
                t.add_row(_t(f.template.name), _t(f.class_name),
                          *(_t(s, _CHECK_COLORS.get(s, "dim")) for s in statuses),
                          _t("yes" if f.instantiated else "no", "" if f.instantiated else "yellow"))
        return t

    def _build_sample_tester(self):
        unmapped = self._results.sensors.unmapped_sensor_classes if self._results.sensors else []  # type: ignore[union-attr]
        if unmapped:
            yield Static(_t(f"Sensor classes without a template: {', '.join(unmapped)}"), id="sensor-notes")
        enabled = bool(self._templates)
        yield Static(
            "Sample tester: replay samples through the selected template's corrupt / stale / debounce rules"
            if enabled else
            "Sample tester disabled: no sensor templates loaded (set the sensor templates file on the first "
            "screen or sensor_templates_file in config.yaml)",
            id="sensor-hint", markup=False)
        with Horizontal(id="sample-row"):
            yield Label("Sensor template", id="sensor-select-label")
            yield Select([(name, name) for name in self._templates], prompt="Sensor template",
                         allow_blank=not self._templates, id="sensor-select", disabled=not enabled)
            yield Input(placeholder="Samples 't:v, t:v, ...' (t in ms) or a CSV path", id="sample-input",
                        disabled=not enabled)
            yield Button("Run checks", variant="primary", id="btn-sensor-run", disabled=not enabled)
        yield Static("", id="sample-summary", markup=False)
        t = DataTable(id="dt-samples", zebra_stripes=True)
        t.add_columns("t (ms)", "Value", "Accepted", "Flags")
        yield t

    def _build_coupling_table(self) -> DataTable:
        t = DataTable(id="dt-coupling", zebra_stripes=True)
        t.add_columns("LRU", "Risk", "Control Coupling", "Data Coupling")
        if self._results.coupling:
            for lru_name, coupling in self._results.coupling.lru_impacts.items():  # type: ignore[union-attr]
                risk_color = {"high": "red", "medium": "yellow", "low": "green"}.get(coupling.risk_level, "white")
                t.add_row(
                    _t(lru_name),
                    _t(coupling.risk_level.upper(), risk_color),
                    _t(str(len(coupling.control_coupling)) + " items"),
                    _t(str(len(coupling.data_coupling)) + " items"),
                )
        if self._results.coupling and not self._results.coupling.lru_impacts:  # type: ignore[union-attr]
            t.add_row(_t("—"), _t("—"), _t("No LRU documents provided"), _t("—"))
        return t

    def _build_checklist_table(self) -> DataTable:
        t = DataTable(id="dt-checklist", zebra_stripes=True)
        t.add_columns("ID", "Category", "Description", "Status", "Evidence")
        if self._results.checklist:
            for item in self._results.checklist.items:  # type: ignore[union-attr]
                status_color = {"PASS": "green", "FAIL": "red", "OPEN": "yellow", "N/A": "dim"}.get(item.status, "white")
                t.add_row(
                    _t(item.item_id), _t(item.category),
                    _t(item.description[:55] + ("…" if len(item.description) > 55 else "")),
                    _t(item.status, status_color),
                    _t(item.evidence),
                )
        return t

    # Both analyses run on a worker thread (the UI stays responsive) and post their text back.

    def _post(self, callback, *args) -> None:
        """Deliver a worker's result to the UI thread, unless a newer run replaced it or the screen went away."""
        if not get_current_worker().is_cancelled:
            self.app.call_from_thread(callback, *args)

    # ── Function effect (R2) ────────────────────────────────────────────────

    def _run_effect(self) -> None:
        name = self.query_one("#effect-input", Input).value.strip()
        r = self._results
        if not name:
            self._show_effect("Enter a method name.")
        elif r.graph is None or not r.parse_results:
            self._show_effect("No analyzed code model: run an analysis first.")
        else:
            self._show_effect("Running…")
            self.run_worker(lambda: self._effect_job(name), thread=True, exclusive=True, group="effect")

    def _effect_job(self, name: str) -> None:
        r = self._results
        try:
            with self._effects_lock:
                if self._effects is None:   # re-parses every file: built once per screen, off the UI thread
                    self._effects = FunctionEffectAnalyzer(r.parse_results, r.graph)
            effects = self._effects.analyze(name)
            text = ("\n\n".join(format_effect(e) for e in effects) if effects
                    else not_found_message("Function", name, function_names(r)))
        except Exception as exc:
            logger.exception("TUI function effect failed")
            text = f"{type(exc).__name__}: {exc}"
        self._post(self._show_effect, text)

    def _show_effect(self, text: str) -> None:
        try:
            self.query_one("#effect-output", Static).update(_t(text))
        except NoMatches:   # the screen was closed while the worker ran
            pass

    # ── Sensor sample tester (R3) ───────────────────────────────────────────

    def _samples_for(self, sensor: str, text: str) -> List[Sample]:
        path = Path(text).expanduser()
        if path.is_file():
            by_sensor = load_samples_csv(path)
            if sensor not in by_sensor:
                raise ValueError(f"No samples for {sensor} in {path}; the CSV has: {', '.join(sorted(by_sensor)) or 'none'}")
            return by_sensor[sensor]
        try:
            return parse_inline_samples(text)
        except ValueError:
            raise ValueError(f"Cannot read samples {text!r}: use 't:v, t:v, ...' (t in ms) "
                             "or the path of a sensor,timestamp_ms,value CSV") from None

    def _run_sample_check(self) -> None:
        sensor = self.query_one("#sensor-select", Select).value
        text = self.query_one("#sample-input", Input).value.strip()
        if not isinstance(sensor, str):
            self._show_samples([], "Select a sensor template.")
        elif not text:
            self._show_samples([], "Enter samples as 't:v, t:v, ...' or a CSV path.")
        else:
            self._show_samples([], "Running…")
            self.run_worker(lambda: self._sample_job(sensor, text), thread=True, exclusive=True, group="samples")

    def _sample_job(self, sensor: str, text: str) -> None:
        rows: List[tuple] = []
        try:
            results = check_samples(self._templates[sensor], self._samples_for(sensor, text))
            shown = visible_samples(results)
            rows = [sample_row(res) for res in shown]
            counts = summarize(results)
            message = (f"{counts['total']} samples: corrupt {counts['corrupt']}, stale {counts['stale']}, "
                       f"debounce {counts['debounce']}\n"
                       f"showing {len(shown)} of {counts['total']} samples (all flagged rows included)")
        except ValueError as exc:   # bad input: the message names the file and line
            message = str(exc)
        except Exception as exc:
            logger.exception("TUI sample check failed")
            message = f"{type(exc).__name__}: {exc}"
        self._post(self._show_samples, rows[:SAMPLE_CHUNK], message)
        for i in range(SAMPLE_CHUNK, len(rows), SAMPLE_CHUNK):   # thousands of flagged rows: a few at a time keeps the UI live
            self._post(self._add_sample_rows, rows[i:i + SAMPLE_CHUNK])

    def _show_samples(self, rows: List[tuple], message: str) -> None:
        try:
            self.query_one("#dt-samples", DataTable).clear()
            self.query_one("#sample-summary", Static).update(_t(message))
        except NoMatches:   # the screen was closed while the worker ran
            return
        self._add_sample_rows(rows)

    def _add_sample_rows(self, rows: List[tuple]) -> None:
        try:
            self.query_one("#dt-samples", DataTable).add_rows(rows)
        except NoMatches:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-effect":
            self._run_effect()
        elif event.button.id == "btn-sensor-run":
            self._run_sample_check()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "effect-input":
            self._run_effect()
        elif event.input.id == "sample-input":
            self._run_sample_check()

    def action_chat(self) -> None:
        self.app.push_screen(ChatScreen(self._results, self._cfg))

    def action_rerun(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────────────
# Chat screen — Q&A about analysis results (static-analysis facts first, optional Ollama explanation)
# ─────────────────────────────────────────────────────────────────────────────

_EXAMPLE_QUESTIONS = [
    "What are the virtual functions implemented by TemperatureSensor?",
    "What components are affected if updateTemperature() changes?",
    "What validation checks are applied to TemperatureSensor?",
]


class ChatScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]
    CSS = """
    ChatScreen {
        layout: vertical;
    }
    #chat-log {
        height: 1fr;
        border: solid $surface;
        margin: 0 1;
        padding: 0 1;
    }
    #chat-status {
        height: 1;
        margin: 0 1;
        color: $text-muted;
    }
    #input-row {
        height: 3;
        margin: 0 1;
    }
    #chat-input {
        width: 1fr;
    }
    #btn-send {
        width: 12;
        margin-left: 1;
    }
    """

    def __init__(self, results: AnalysisResults, cfg: Optional[Config] = None) -> None:
        super().__init__()
        self._results = results
        self._cfg = cfg or load_config()
        self._ollama = OllamaClient(base_url=self._cfg.ollama_url, model=self._cfg.ollama_model)
        self._ollama_ready: Optional[bool] = None   # None until the availability probe has run
        self._pending: deque[str] = deque()          # questions waiting for the one being answered
        self._answering = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="chat-log", highlight=False, markup=False, wrap=True)
        yield Static("", id="chat-status", markup=False)
        with Horizontal(id="input-row"):
            yield Input(placeholder="Ask about your analysis results…", id="chat-input")
            yield Button("Send", variant="primary", id="btn-send")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(_t("DO-178C Analysis Assistant", "bold cyan"))
        log.write("Ask questions about your analysis results. Type your question and press Enter or click Send.")
        log.write("Try:")
        for question in _EXAMPLE_QUESTIONS:
            log.write(f"  {question}")
        log.write("")
        self._update_status()
        if self._cfg.ollama_enabled:
            self.run_worker(self._probe_ollama, thread=True)

    def _probe_ollama(self) -> None:
        self._ollama_ready = self._ollama.is_available()
        self.app.call_from_thread(self._update_status)

    def _update_status(self, thinking: bool = False) -> None:
        if thinking or self._answering:   # an answer in flight keeps "Thinking…" whatever else updates the status
            text = _t("Thinking…", "dim")
        elif not self._cfg.ollama_enabled:
            text = _t("Ollama disabled in config: using structured offline answers", "yellow")
        elif self._ollama_ready is None:
            text = _t("Checking Ollama…", "dim")
        elif self._ollama_ready:
            text = _t(f"Ollama ready ({self._cfg.ollama_model}): facts from static analysis, explained by the LLM", "green")
        else:
            text = _t("Ollama unavailable: using structured offline answers", "yellow")
        try:
            self.query_one("#chat-status", Static).update(text)
        except NoMatches:   # a worker finished after the screen was closed
            pass

    def _ask(self, question: str) -> str:
        narrator = None
        if self._cfg.ollama_enabled and self._ollama.is_available():
            narrator = RAGNarrativeGenerator(self._ollama, self._results.search)
        return answer_question(question, self._results, narrator)

    def _answer(self, question: str) -> None:
        try:
            answer = self._ask(question)
        except Exception as exc:
            logger.exception("Chat answer failed")
            answer = f"Could not answer: {type(exc).__name__}: {exc}"
        self.app.call_from_thread(self._show_answer, answer)

    def _submit(self) -> None:
        inp = self.query_one("#chat-input", Input)
        question = inp.value.strip()
        if not question:
            return
        inp.value = ""
        self.query_one("#chat-log", RichLog).write(Text.assemble(("You: ", "bold cyan"), question))
        self._update_status(thinking=True)
        self._pending.append(question)
        if not self._answering:
            self._answer_next()

    def _answer_next(self) -> None:
        """One question at a time, in the order asked: answers cannot overtake each other."""
        question = self._pending.popleft()
        self._answering = True
        self.run_worker(lambda: self._answer(question), thread=True)

    def _show_answer(self, answer: str) -> None:
        try:
            log = self.query_one("#chat-log", RichLog)
        except NoMatches:   # the screen was closed while the answer was computed
            return
        log.write(Text.assemble(("Assistant: ", "bold green"), answer))   # Text: answers hold [...] (ranges, arrays)
        log.write("")
        self._answering = False
        if self._pending:   # "Thinking…" stays until the last answer
            self._answer_next()
        else:
            self._update_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-send":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._submit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear_chat(self) -> None:
        self.query_one("#chat-log", RichLog).clear()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngineApp(App):
    TITLE = "DO-178C RAG Engine"
    SUB_TITLE = "Codebase Analysis Tool"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._restore_logging: Callable[[], None] = lambda: None

    def on_mount(self) -> None:
        self._restore_logging = self._log_to_file()
        self.push_screen(ConfigScreen())

    def on_unmount(self) -> None:
        self._restore_logging()

    def _log_to_file(self) -> Callable[[], None]:
        """Textual owns the terminal, so loguru's stderr sink would draw over the screen: log to
        ``<cache_dir>/tui.log`` while the app runs.  Returns the function that puts stderr back."""
        try:
            cache_dir = load_config().cache_dir
        except Exception:      # a broken config.yaml is reported by the analysis, not by the logging set-up
            cache_dir = Config().cache_dir
        try:
            logger.remove(_stderr_sink)
            stderr_removed = True
        except ValueError:     # the host program already removed it
            stderr_removed = False
        file_sink: Optional[int] = None
        path = Path(cache_dir).expanduser() / 'tui.log'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_sink = logger.add(path, level='DEBUG', encoding='utf-8', rotation='5 MB', retention=3)
        except OSError as exc:
            self.notify(f"Cannot write the log file {path}: {exc}; log output is discarded",
                        severity="warning", timeout=10, markup=False)

        def restore() -> None:
            global _stderr_sink
            if file_sink is not None:
                logger.remove(file_sink)
            if stderr_removed:
                _stderr_sink = logger.add(sys.stderr)
        return restore


def launch() -> None:
    RAGEngineApp().run()


if __name__ == "__main__":
    launch()
