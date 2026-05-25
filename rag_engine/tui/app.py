from __future__ import annotations
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label,
    LoadingIndicator, ProgressBar, RichLog, Select, Static,
    TabbedContent, TabPane,
)
from textual.worker import Worker, WorkerState


# ─────────────────────────────────────────────────────────────────────────────
# Data holders (filled after analysis runs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResults:
    virtual: object = None
    coupling: object = None
    dead: object = None
    standards: object = None
    checklist: object = None
    error: Optional[str] = None


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
                yield Button("Run Analysis", variant="primary", id="btn-run")
                yield Button("Quit", variant="error", id="btn-quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-quit":
            self.app.exit()
        elif event.button.id == "btn-run":
            codebase = self.query_one("#codebase", Input).value.strip()
            if not codebase:
                self.notify("Codebase path is required", severity="error")
                return
            if not Path(codebase).exists():
                self.notify(f"Path not found: {codebase}", severity="error")
                return
            base_build = self.query_one("#base-build", Input).value.strip() or None
            lru_docs = self.query_one("#lru-docs", Input).value.strip() or None
            output_fmt = self.query_one("#output-format", Select).value
            dal = self.query_one("#dal-level", Select).value
            self.app.push_screen(AnalysisScreen(codebase, base_build, lru_docs, str(output_fmt), str(dal)))


# ─────────────────────────────────────────────────────────────────────────────
# Analysis / progress screen
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisScreen(Screen):
    CSS = """
    AnalysisScreen {
        align: center middle;
    }
    #progress-panel {
        width: 72;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    .task-row {
        height: 3;
        margin-bottom: 1;
    }
    .task-label {
        width: 22;
    }
    ProgressBar {
        width: 1fr;
    }
    #log-box {
        height: 12;
        border: solid $surface;
        margin-top: 1;
    }
    """

    def __init__(self, codebase: str, base_build: Optional[str], lru_docs: Optional[str],
                 output_fmt: str, dal: str) -> None:
        super().__init__()
        self._codebase = codebase
        self._base_build = base_build
        self._lru_docs = lru_docs
        self._output_fmt = output_fmt
        self._dal = dal
        self._results = AnalysisResults()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="progress-panel"):
            yield Static(f"Analyzing: {self._codebase}", id="analysis-title")
            for task_id, label in [
                ("pb-parse", "Parsing C++ files"),
                ("pb-virtual", "Virtual analysis"),
                ("pb-dead", "Dead code detection"),
                ("pb-standards", "Standards validation"),
            ]:
                with Horizontal(classes="task-row"):
                    yield Label(label, classes="task-label")
                    yield ProgressBar(total=100, show_eta=False, id=task_id)
            yield RichLog(id="log-box", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._run_analysis, exclusive=True, thread=True)

    def _log(self, msg: str) -> None:
        self.app.call_from_thread(
            lambda: self.query_one("#log-box", RichLog).write(msg)
        )

    def _advance(self, pb_id: str, value: int) -> None:
        self.app.call_from_thread(
            lambda: self.query_one(f"#{pb_id}", ProgressBar).advance(value)
        )

    def _run_analysis(self) -> None:
        try:
            from rag_engine.config import load_config, Config
            from rag_engine.core.embeddings import SemanticSearch
            from rag_engine.core.graph_builder import CallGraphBuilder
            from rag_engine.document_processor.code_reader import CodeReader
            from rag_engine.document_processor.doc_reader import DocReader
            from rag_engine.features.coupling_analysis import CouplingAnalyzer
            from rag_engine.features.dead_code_detector import DeadCodeDetector
            from rag_engine.features.standards_validator import StandardsValidator
            from rag_engine.features.virtual_analysis import VirtualAnalyzer
            from rag_engine.reporting.checklist_filler import ChecklistFiller
            from rag_engine.reporting.report_generator import ReportGenerator

            cfg = load_config()
            cfg.dal_level = self._dal  # type: ignore[assignment]
            cfg.output_dir = 'output'

            # Parse
            self._log("[bold]Parsing C++ files...[/bold]")
            reader = CodeReader(cache_dir=cfg.cache_dir)
            current = reader.read_directory(self._codebase)
            base = reader.read_directory(self._base_build) if self._base_build else current
            self._advance("pb-parse", 100)
            self._log(f"  Parsed [green]{len(current)}[/green] files, "
                      f"[green]{sum(len(r.functions) for r in current.values())}[/green] functions")

            gb = CallGraphBuilder(current)
            gb.build()
            search = SemanticSearch(model_name=cfg.embedding_model)
            search.index_functions([fn for r in current.values() for fn in r.functions])
            lru_docs = DocReader().read_lru_documents(self._lru_docs) if self._lru_docs else {}

            # Virtual
            self._log("[bold]Running virtual analysis...[/bold]")
            virtual = VirtualAnalyzer(base, current).analyze()
            self._advance("pb-virtual", 100)
            s = virtual.summary
            self._log(f"  Added [red]{s['added']}[/red]  Modified [yellow]{s['modified']}[/yellow]  "
                      f"Unchanged [green]{s['unchanged']}[/green]")

            # Dead code
            self._log("[bold]Running dead code detection...[/bold]")
            dead = DeadCodeDetector(gb, current, cfg.entry_points).analyze()
            self._advance("pb-dead", 100)
            self._log(f"  Dead [red]{dead.dead_count}[/red]  Deactivated [yellow]{dead.deactivated_count}[/yellow]")

            # Standards
            self._log("[bold]Running standards validation...[/bold]")
            standards = StandardsValidator(cfg, current).analyze()
            self._advance("pb-standards", 100)
            self._log(f"  Score [green]{standards.compliance_score:.1f}%[/green]  "
                      f"Critical [red]{standards.violations_by_severity.get('CRITICAL',0)}[/red]  "
                      f"Major [yellow]{standards.violations_by_severity.get('MAJOR',0)}[/yellow]")

            # Coupling
            coupling = CouplingAnalyzer(current, gb, search, lru_docs).analyze()

            # Reports + checklist
            gen = ReportGenerator(cfg)
            gen.generate_all(virtual, coupling, dead, standards, format=self._output_fmt)  # type: ignore[arg-type]
            checklist = ChecklistFiller(cfg).fill_sqa_checklist(virtual, dead, standards)

            self._results.virtual = virtual
            self._results.coupling = coupling
            self._results.dead = dead
            self._results.standards = standards
            self._results.checklist = checklist
            self._log("[bold green]Analysis complete. Reports written to output/[/bold green]")
            self.app.call_from_thread(
                lambda: self.app.push_screen(ResultsScreen(self._results, cfg.output_dir))
            )
        except Exception as exc:
            self._results.error = str(exc)
            self._log(f"[bold red]ERROR: {exc}[/bold red]")
            self.app.call_from_thread(
                lambda: self.notify(f"Analysis failed: {exc}", severity="error", timeout=10)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Results browser screen
# ─────────────────────────────────────────────────────────────────────────────

class ResultsScreen(Screen):
    BINDINGS = [
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
    """

    def __init__(self, results: AnalysisResults, output_dir: str) -> None:
        super().__init__()
        self._results = results
        self._output_dir = output_dir

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        r = self._results
        summary_parts = []
        if r.standards:
            summary_parts.append(f"Score {r.standards.compliance_score:.1f}%")  # type: ignore[union-attr]
        if r.virtual:
            s = r.virtual.summary  # type: ignore[union-attr]
            summary_parts.append(f"Virtual: +{s['added']} ~{s['modified']}")
        if r.dead:
            summary_parts.append(f"Dead: {r.dead.dead_count}")  # type: ignore[union-attr]
        yield Static("  |  ".join(summary_parts), id="summary-bar")
        with TabbedContent("Standards", "Virtual Changes", "Dead Code", "Coupling", "Checklist"):
            with TabPane("Standards", id="tab-standards"):
                yield self._build_standards_table()
            with TabPane("Virtual Changes", id="tab-virtual"):
                yield self._build_virtual_table()
            with TabPane("Dead Code", id="tab-dead"):
                yield self._build_dead_table()
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
                    f"[{sev_color}]{v.severity}[/{sev_color}]",
                    v.rule, Path(v.file).name, str(v.line), v.element,
                    v.message[:60] + ("…" if len(v.message) > 60 else ""),
                )
        return t

    def _build_virtual_table(self) -> DataTable:
        t = DataTable(id="dt-virtual", zebra_stripes=True)
        t.add_columns("Change", "Function", "Category", "File", "Line", "Reverification")
        if self._results.virtual:
            for c in self._results.virtual.changes:  # type: ignore[union-attr]
                color = {"added": "green", "removed": "red", "modified": "yellow", "unchanged": "dim"}.get(c.change_type, "white")
                t.add_row(
                    f"[{color}]{c.change_type.upper()}[/{color}]",
                    c.function.name, c.do178c_category,
                    Path(c.function.file_path).name, str(c.function.line_number),
                    (c.reverification_scope or "")[:55],
                )
        return t

    def _build_dead_table(self) -> DataTable:
        t = DataTable(id="dt-dead", zebra_stripes=True)
        t.add_columns("Category", "Name", "File", "Line", "Disposition", "Coverage Impact")
        if self._results.dead:
            for item in self._results.dead.items:  # type: ignore[union-attr]
                color = "red" if item.category == "dead_code" else "yellow"
                t.add_row(
                    f"[{color}]{item.category}[/{color}]",
                    item.name, Path(item.file_path).name, str(item.line_number),
                    item.do178c_disposition, item.coverage_impact[:50],
                )
        return t

    def _build_coupling_table(self) -> DataTable:
        t = DataTable(id="dt-coupling", zebra_stripes=True)
        t.add_columns("LRU", "Risk", "Control Coupling", "Data Coupling")
        if self._results.coupling:
            for lru_name, coupling in self._results.coupling.lru_impacts.items():  # type: ignore[union-attr]
                risk_color = {"high": "red", "medium": "yellow", "low": "green"}.get(coupling.risk_level, "white")
                t.add_row(
                    lru_name,
                    f"[{risk_color}]{coupling.risk_level.upper()}[/{risk_color}]",
                    str(len(coupling.control_coupling)) + " items",
                    str(len(coupling.data_coupling)) + " items",
                )
        if self._results.coupling and not self._results.coupling.lru_impacts:  # type: ignore[union-attr]
            t.add_row("—", "—", "No LRU documents provided", "—")
        return t

    def _build_checklist_table(self) -> DataTable:
        t = DataTable(id="dt-checklist", zebra_stripes=True)
        t.add_columns("ID", "Category", "Description", "Status", "Evidence")
        if self._results.checklist:
            for item in self._results.checklist.items:  # type: ignore[union-attr]
                status_color = {"PASS": "green", "FAIL": "red", "OPEN": "yellow", "N/A": "dim"}.get(item.status, "white")
                t.add_row(
                    item.item_id, item.category,
                    item.description[:55] + ("…" if len(item.description) > 55 else ""),
                    f"[{status_color}]{item.status}[/{status_color}]",
                    item.evidence,
                )
        return t

    def action_rerun(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngineApp(App):
    TITLE = "DO-178C RAG Engine"
    SUB_TITLE = "Codebase Analysis Tool"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen())


def launch() -> None:
    RAGEngineApp().run()


if __name__ == "__main__":
    launch()
