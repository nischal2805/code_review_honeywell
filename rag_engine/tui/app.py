from __future__ import annotations
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
# Offline Q&A helper — answers common questions from structured results
# without needing an LLM
# ─────────────────────────────────────────────────────────────────────────────

def _offline_answer(question: str, results: "AnalysisResults") -> str:
    q = question.lower()

    # Standards
    if results.standards:
        s = results.standards
        if any(k in q for k in ("critical", "critical violation", "most serious")):
            n = s.violations_by_severity.get("CRITICAL", 0)  # type: ignore[union-attr]
            items = [v for v in s.violations if v.severity == "CRITICAL"]  # type: ignore[union-attr]
            lines = [f"  {v.rule} in {v.file.split('/')[-1].split(chr(92))[-1]}:{v.line} — {v.message}" for v in items[:5]]
            return f"{n} CRITICAL violation(s):\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("score", "compliance", "percent")):
            return f"Compliance score: {s.compliance_score:.1f}%"
        if any(k in q for k in ("violation", "major", "medium", "minor")):
            bysev = s.violations_by_severity  # type: ignore[union-attr]
            return (f"Violations by severity:\n"
                    f"  CRITICAL: {bysev.get('CRITICAL',0)}\n"
                    f"  MAJOR:    {bysev.get('MAJOR',0)}\n"
                    f"  MEDIUM:   {bysev.get('MEDIUM',0)}\n"
                    f"  MINOR:    {bysev.get('MINOR',0)}\n"
                    f"  Total:    {len(s.violations)}")
        if "goto" in q:
            gotos = [v for v in s.violations if "GOTO" in v.rule]  # type: ignore[union-attr]
            if gotos:
                return "\n".join(f"  {v.file.split('/')[-1]}:{v.line} — {v.element}" for v in gotos)
            return "No goto violations found."

    # Virtual analysis
    if results.virtual:
        v = results.virtual
        if any(k in q for k in ("reverif", "category 2", "cat2", "cat 2")):
            cat2 = [c for c in v.changes if c.do178c_category == "Category 2"]
            lines = [f"  {c.function.name} ({c.change_type}) — {c.reverification_scope or 'full reverification'}" for c in cat2[:8]]
            return f"{len(cat2)} Category 2 change(s) requiring reverification:\n" + "\n".join(lines)
        if any(k in q for k in ("changed", "modified", "virtual", "change")):
            s = v.summary
            return (f"Virtual function changes:\n"
                    f"  Added:     {s.get('added', 0)}\n"
                    f"  Removed:   {s.get('removed', 0)}\n"
                    f"  Modified:  {s.get('modified', 0)}\n"
                    f"  Unchanged: {s.get('unchanged', 0)}")

    # Dead code
    if results.dead:
        d = results.dead
        if any(k in q for k in ("remove", "dead code", "dead")):
            items = [i for i in d.items if i.category == "dead_code"]
            lines = [f"  {i.name} in {i.file_path.split('/')[-1].split(chr(92))[-1]}:{i.line_number}" for i in items[:8]]
            return f"{d.dead_count} dead function(s) to remove:\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("deactivated", "justify")):
            items = [i for i in d.items if i.category == "deactivated_code"]
            lines = [f"  {i.name} — {i.coverage_impact}" for i in items[:6]]
            return f"{d.deactivated_count} deactivated item(s):\n" + ("\n".join(lines) or "  (none)")
        if any(k in q for k in ("total dead", "how much", "how many dead")):
            return f"Dead: {d.dead_count}  Deactivated: {d.deactivated_count}  Total: {d.dead_count + d.deactivated_count}"

    # Coupling
    if results.coupling:
        c = results.coupling
        if any(k in q for k in ("high risk", "highest risk", "most risk")):
            high = [(name, lru) for name, lru in c.lru_impacts.items() if lru.risk_level == "high"]
            if high:
                return "High-risk LRUs:\n" + "\n".join(f"  {n}: {len(l.control_coupling)} ctrl + {len(l.data_coupling)} data coupling items" for n, l in high)
            return "No high-risk LRUs found."
        if any(k in q for k in ("coupling", "lru", "risk")):
            lines = [f"  {n}: {l.risk_level.upper()} ({len(l.control_coupling)} ctrl, {len(l.data_coupling)} data)" for n, l in c.lru_impacts.items()]
            return "LRU coupling summary:\n" + ("\n".join(lines) or "  (no LRU documents provided)")

    # General DO-178C knowledge
    if "category 1" in q:
        return ("Category 1: comment or formatting change only.\n"
                "Requires regression tests only — no full reverification.")
    if "category 2" in q:
        return ("Category 2: signature, implementation, or existence change.\n"
                "Requires full reverification of affected functions and their callers.")
    if "dal" in q:
        return ("DAL levels:\n"
                "  A — Catastrophic failure. MC/DC coverage required.\n"
                "  B — Hazardous failure. Decision coverage required.\n"
                "  C — Major failure. Statement coverage required.\n"
                "  D — Minor failure. Basic coverage required.")
    if any(k in q for k in ("ready", "certif", "pass")):
        if results.standards:
            crit = results.standards.violations_by_severity.get("CRITICAL", 0)
            score = results.standards.compliance_score
            if crit == 0 and score >= 90:
                return f"Compliance score {score:.1f}%, 0 CRITICAL violations. Looks good for certification review."
            return (f"Compliance score {score:.1f}%, {crit} CRITICAL violation(s).\n"
                    f"Fix CRITICAL violations before submitting to DER.")
        return "Run analysis first to assess certification readiness."

    return ("I can answer questions about your analysis results.\nTry:\n"
            "  'how many critical violations'\n"
            "  'which functions need reverification'\n"
            "  'what dead code should be removed'\n"
            "  'which LRU has highest risk'\n"
            "  'what is the compliance score'")


# ─────────────────────────────────────────────────────────────────────────────
# Data holders (filled after analysis runs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResults:
    virtual: Any = None
    coupling: Any = None
    dead: Any = None
    standards: Any = None
    checklist: Any = None
    search: Any = None
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

            # RAG narrative enrichment
            from rag_engine.llm.ollama_client import OllamaClient
            from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
            llm_narratives: dict = {}
            llm = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)
            if cfg.ollama_enabled and llm.is_available():
                self._log("[bold]Generating RAG narratives via LLM...[/bold]")
                rag = RAGNarrativeGenerator(llm, search)
                rag_tasks = {
                    'virtual': (
                        "virtual function change reverification DO-178C category",
                        f"Virtual: added={virtual.summary.get('added',0)}, removed={virtual.summary.get('removed',0)}, "
                        f"modified={virtual.summary.get('modified',0)}, unchanged={virtual.summary.get('unchanged',0)}. "
                        f"Assess reverification scope per DO-178C §12."
                    ),
                    'coupling': (
                        "LRU control data coupling dependency interface",
                        f"Coupling for {len(coupling.lru_impacts)} LRUs. "
                        f"Risk levels: {[(k, v.risk_level) for k, v in coupling.lru_impacts.items()]}. "
                        f"Assess coupling impact on DAL {cfg.dal_level}."
                    ),
                    'dead_code': (
                        "dead code unreachable deactivated function coverage",
                        f"Dead code: {dead.dead_count} dead, {dead.deactivated_count} deactivated, "
                        f"total {dead.total_functions}. Coverage: {dead.structural_coverage_impact}. "
                        f"Assess DO-178C §6.4.2.2 compliance."
                    ),
                    'standards': (
                        "MISRA C++ DO-178C code standard naming complexity violation",
                        f"Standards: score={standards.compliance_score:.1f}%, "
                        f"severities={standards.violations_by_severity}. "
                        f"Assess DO-178C §5.1 and MISRA C++ status."
                    ),
                }
                for key, (query, summary) in rag_tasks.items():
                    try:
                        narrative = rag.generate(query, summary, k=8, max_tokens=400)
                        if narrative:
                            llm_narratives[key] = narrative
                            self._log(f"  [green]RAG narrative: {key}[/green]")
                    except Exception as exc:
                        self._log(f"  [yellow]LLM skipped {key}: {exc}[/yellow]")

            # Reports + checklist
            gen = ReportGenerator(cfg)
            gen.generate_all(virtual, coupling, dead, standards,
                             format=self._output_fmt,  # type: ignore[arg-type]
                             llm_narratives=llm_narratives)
            checklist = ChecklistFiller(cfg).fill_sqa_checklist(virtual, dead, standards)

            self._results.virtual = virtual
            self._results.coupling = coupling
            self._results.dead = dead
            self._results.standards = standards
            self._results.checklist = checklist
            self._results.search = search
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

    def action_chat(self) -> None:
        self.app.push_screen(ChatScreen(self._results))

    def action_rerun(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────────────
# Chat screen — Q&A about analysis results (Ollama or offline fallback)
# ─────────────────────────────────────────────────────────────────────────────

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

    def __init__(self, results: AnalysisResults) -> None:
        super().__init__()
        self._results = results
        self._ollama = None
        self._history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        yield Static("", id="chat-status")
        with Horizontal(id="input-row"):
            yield Input(placeholder="Ask about your analysis results…", id="chat-input")
            yield Button("Send", variant="primary", id="btn-send")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold cyan]DO-178C Analysis Assistant[/bold cyan]")
        log.write("Ask questions about your analysis results. Type your question and press Enter or click Send.")
        log.write("")

        from rag_engine.llm.ollama_client import OllamaClient
        from rag_engine.config import load_config
        cfg = load_config()
        self._ollama = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)

        if self._ollama.is_available():
            self.query_one("#chat-status", Static).update(
                f"[green]Ollama ready ({cfg.ollama_model})[/green]"
            )
            self._seed_context()
        else:
            self.query_one("#chat-status", Static).update(
                "[yellow]Ollama unavailable — using structured offline answers[/yellow]"
            )

    def _seed_context(self) -> None:
        r = self._results
        parts = ["You are a DO-178C certification assistant. Answer questions about these analysis results concisely.\n"]
        if r.standards:
            s = r.standards
            parts.append(f"Standards: compliance {s.compliance_score:.1f}%, "
                         f"CRITICAL={s.violations_by_severity.get('CRITICAL',0)}, "
                         f"MAJOR={s.violations_by_severity.get('MAJOR',0)}, "
                         f"total violations={len(s.violations)}")
        if r.virtual:
            v = r.virtual.summary
            parts.append(f"Virtual changes: added={v.get('added',0)}, removed={v.get('removed',0)}, "
                         f"modified={v.get('modified',0)}, unchanged={v.get('unchanged',0)}")
        if r.dead:
            parts.append(f"Dead code: dead={r.dead.dead_count}, deactivated={r.dead.deactivated_count}")
        if r.coupling:
            risks = {n: l.risk_level for n, l in r.coupling.lru_impacts.items()}
            parts.append(f"Coupling risk by LRU: {risks}")
        self._history = [{"role": "system", "content": "\n".join(parts)}]

    def _ask(self, question: str) -> str:
        if self._ollama and self._ollama.is_available() and self._results.search:
            from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
            rag = RAGNarrativeGenerator(self._ollama, self._results.search)
            analysis_ctx = "\n".join(
                f"{m['content']}" for m in self._history[:1]
            )
            answer = rag.answer_question(question, analysis_ctx, k=8, max_tokens=500)
            if answer:
                return answer
        if self._ollama and self._ollama.is_available():
            self._history.append({"role": "user", "content": question})
            ctx = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self._history[-6:])
            answer = self._ollama.generate(ctx, max_tokens=400)
            if answer:
                self._history.append({"role": "assistant", "content": answer})
                return answer
        return _offline_answer(question, self._results)

    def _submit(self) -> None:
        inp = self.query_one("#chat-input", Input)
        question = inp.value.strip()
        if not question:
            return
        inp.value = ""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]You:[/bold cyan] {question}")
        status = self.query_one("#chat-status", Static)
        status.update("[dim]Thinking…[/dim]")

        def _bg():
            answer = self._ask(question)
            self.app.call_from_thread(lambda: self._show_answer(answer))

        threading.Thread(target=_bg, daemon=True).start()

    def _show_answer(self, answer: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold green]Assistant:[/bold green] {answer}")
        log.write("")
        from rag_engine.config import load_config
        cfg = load_config()
        if self._ollama and self._ollama.is_available():
            self.query_one("#chat-status", Static).update(f"[green]Ollama ready ({cfg.ollama_model})[/green]")
        else:
            self.query_one("#chat-status", Static).update("[yellow]Offline mode[/yellow]")

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
        self._history = []
        self._seed_context()


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
