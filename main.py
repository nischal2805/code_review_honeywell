from __future__ import annotations
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')
import sys
from pathlib import Path
from typing import Any, List, Optional
import typer
from loguru import logger
from tqdm import tqdm

from rag_engine.config import Config, load_config
from rag_engine.features.function_effect import FunctionEffectAnalyzer, format_effect
from rag_engine.features.sensor_analysis import SensorFinding
from rag_engine.features.sensor_data_checker import check_samples, load_samples_csv, summarize
from rag_engine.features.sensor_template import load_sensor_templates
from rag_engine.features.virtual_analysis import format_virtual_function
from rag_engine.llm.offline_qa import answer_question, function_names, not_found_message
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
from rag_engine.pipeline import AnalysisResults, InputError, run_pipeline
from rag_engine.reporting.report_generator import ReportGenerator

app = typer.Typer(name='rag-engine', help='DO-178C Codebase Analysis Engine')

_CONFIG = typer.Option('config.yaml', help='Path to config.yaml')
_TEMPLATES = typer.Option(None, '--sensor-templates', '--templates',
                          help='Sensor template YAML (overrides sensor_templates_file in the config)')


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(1)


def _load(config: str, templates: Optional[str] = None) -> Config:
    cfg = load_config(config)
    if templates:
        if not Path(templates).expanduser().exists():
            raise _fail(f"Sensor templates file not found: {templates}")
        cfg.sensor_templates_file = str(Path(templates).expanduser().resolve())
    return cfg


def _model(cfg: Config, codebase: str, **kwargs: Any) -> AnalysisResults:
    """The analyses without semantic search or reports: what the query commands need."""
    return _run(cfg, codebase, use_semantic_search=False, write_reports=False, **kwargs)


def _run(cfg: Config, codebase: str, **kwargs: Any) -> AnalysisResults:
    try:
        return run_pipeline(cfg, codebase, **kwargs)
    except InputError as exc:   # anything else is a bug: let it show its traceback
        raise _fail(f"Error: {exc}")


@app.command()
def analyze(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    base_build: Optional[str] = typer.Option(None, help='Base build dir for virtual analysis diff'),
    lru_docs: Optional[str] = typer.Option(None, help='Dir with LRU PDF/DOCX documents'),
    standards_file: Optional[str] = typer.Option(None, help='Path to custom DAL standards YAML/JSON/PDF/DOCX file'),
    sensor_templates: Optional[str] = _TEMPLATES,
    config: str = _CONFIG,
    output_format: str = typer.Option('docx', help='Output format: docx or markdown'),
    semantic_search: bool = typer.Option(True, '--semantic-search/--no-semantic-search',
                                         help='Use the embedding model for coupling analysis (keyword matching without)'),
) -> None:
    """Run all DO-178C analyses and generate §11-compliant reports."""
    cfg = _load(config, sensor_templates)
    logger.info(f"RAG Engine — DAL {cfg.dal_level}, output: {cfg.output_dir}")
    if standards_file:
        if not Path(standards_file).expanduser().exists():
            raise _fail(f"Standards file not found: {standards_file}")
        cfg.standards_file = standards_file

    with tqdm(total=100, desc='Analysis') as bar:
        def progress(step: str, pct: int) -> None:
            bar.set_description(step)
            bar.update(pct - bar.n)

        results = _run(cfg, codebase, base_build=base_build, lru_docs=lru_docs, output_format=output_format,
                       progress=progress, use_semantic_search=semantic_search)

    typer.echo(f"\nReports in {cfg.output_dir}/")
    for p in results.report_paths:
        typer.echo(f"  {p}")

    critical = results.standards.violations_by_severity.get('CRITICAL', 0)
    if critical > 0:
        typer.echo(f"\nWARNING: {critical} CRITICAL violation(s)", err=True)
        raise typer.Exit(1)


@app.command()
def virtuals(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    config: str = _CONFIG,
) -> None:
    """List the virtual functions, their overrides, callers and class hierarchies."""
    cfg = _load(config)
    cfg.sensor_templates_file = ''
    result = _model(cfg, codebase).virtual
    if not result.inventory:
        typer.echo("No virtual functions found.")
    for info in result.inventory:
        typer.echo(format_virtual_function(info) + '\n')
    if result.hierarchy_trees:
        typer.echo("Class hierarchies:")
        for tree in result.hierarchy_trees.values():
            typer.echo(tree + '\n')


@app.command()
def effect(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    method: str = typer.Argument(..., help='Method name, e.g. updateTemperature or SensorManager::updateTemperature'),
    config: str = _CONFIG,
    output_format: str = typer.Option('markdown', help='Report format: docx or markdown'),
) -> None:
    """Show what a method does, what it touches and what is affected if it changes."""
    cfg = _load(config)
    cfg.sensor_templates_file = ''
    results = _model(cfg, codebase)
    effects = FunctionEffectAnalyzer(results.parse_results, results.graph).analyze(method)
    if not effects:
        raise _fail(not_found_message('Function', method, function_names(results)))
    typer.echo('\n\n'.join(format_effect(e) for e in effects))
    typer.echo(f"\nReport: {ReportGenerator(cfg).generate_function_effect(effects, output_format, query=method)}")  # type: ignore[arg-type]


def _sensor_table(findings: List[SensorFinding]) -> str:
    rows = [('Sensor', 'Class', 'Corrupt', 'Stale', 'Debounce')]
    for f in findings:
        rows.append((f.template.name, f.class_name or '(not found)',
                     *(f.checks[c].status if f.checks else '-' for c in ('corrupt', 'stale', 'debounce'))))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    return '\n'.join('  '.join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows)


@app.command()
def sensors(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    templates: Optional[str] = _TEMPLATES,
    config: str = _CONFIG,
) -> None:
    """Check each templated sensor for corrupt / stale / debounce handling in the code."""
    cfg = _load(config, templates)
    if not cfg.sensor_templates_file:
        raise _fail("No sensor templates: pass --templates or set sensor_templates_file in the config")
    result = _model(cfg, codebase).sensors
    typer.echo(_sensor_table(result.findings))
    for f in result.findings:
        for name, c in f.checks.items():
            if c.status in ('missing', 'mismatch'):
                found = f"; found {', '.join(f'{v:g}' for v in c.found)}" if c.found else ''
                typer.echo(f"  {f.template.name}.{name}: {c.status} (expected {c.expected}{found})")
    if result.unmapped_sensor_classes:
        typer.echo(f"Sensor classes without a template: {', '.join(result.unmapped_sensor_classes)}")


@app.command('sensor-check')
def sensor_check(
    samples_csv: str = typer.Argument(..., help='CSV with columns sensor,timestamp_ms,value'),
    templates: Optional[str] = _TEMPLATES,
    config: str = _CONFIG,
) -> None:
    """Replay recorded sensor samples through the templates' corrupt / stale / debounce rules."""
    cfg = _load(config, templates)
    if not cfg.sensor_templates_file:
        raise _fail("No sensor templates: pass --templates or set sensor_templates_file in the config")
    try:
        by_name = {t.name: t for t in load_sensor_templates(cfg.sensor_templates_file)}
        samples = load_samples_csv(samples_csv)
    except (ValueError, OSError) as exc:
        raise _fail(f"Error: {exc}")
    unknown = []
    for name, rows in samples.items():
        if name not in by_name:
            unknown.append(name)
            continue
        results = check_samples(by_name[name], rows)
        typer.echo(name)
        for r in results:
            accepted = '' if r.accepted_value is None else f"  accepted={r.accepted_value:.10g}"
            typer.echo(f"  t={r.timestamp_ms:.10g} ms  value={r.value:.10g}{accepted}  {', '.join(r.flags) or 'ok'}")
        counts = summarize(results)
        typer.echo(f"  Summary: {counts['total']} samples, corrupt {counts['corrupt']}, "
                   f"stale {counts['stale']}, debounce {counts['debounce']}\n")
    if unknown:
        raise _fail(f"No template for sensor(s) in the CSV: {', '.join(unknown)}")


@app.command()
def ask(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    question: str = typer.Argument(..., help='e.g. "What are the virtual functions implemented by TemperatureSensor?"'),
    sensor_templates: Optional[str] = _TEMPLATES,
    config: str = _CONFIG,
) -> None:
    """Answer a question from the static analysis; the local LLM (if enabled) only explains the facts."""
    cfg = _load(config, sensor_templates)
    results = _model(cfg, codebase)
    llm = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)
    narrator = RAGNarrativeGenerator(llm, None) if cfg.ollama_enabled and llm.is_available() else None
    typer.echo(answer_question(question, results, narrator))


@app.command()
def validate_config(config: str = typer.Option('config.yaml')) -> None:
    """Validate configuration file."""
    cfg = load_config(config)
    typer.echo(f"Config OK — DAL {cfg.dal_level}, output: {cfg.output_dir}")

if __name__ == '__main__':
    # Piped output on Windows defaults to cp1252, which cannot print the inheritance-tree characters.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
    app()
