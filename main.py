from __future__ import annotations
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')
from pathlib import Path
from typing import Optional
import typer
from loguru import logger
from tqdm import tqdm

from rag_engine.config import load_config
from rag_engine.core.embeddings import SemanticSearch
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.document_processor.code_reader import CodeReader
from rag_engine.document_processor.doc_reader import DocReader
from rag_engine.features.coupling_analysis import CouplingAnalyzer
from rag_engine.features.dead_code_detector import DeadCodeDetector
from rag_engine.features.standards_validator import StandardsValidator
from rag_engine.features.virtual_analysis import VirtualAnalyzer
from rag_engine.knowledge_base.index_manager import IndexManager
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.reporting.checklist_filler import ChecklistFiller
from rag_engine.reporting.report_generator import ReportGenerator

app = typer.Typer(name='rag-engine', help='DO-178C Codebase Analysis Engine')


@app.command()
def analyze(
    codebase: str = typer.Argument(..., help='Path to C++ codebase directory'),
    base_build: Optional[str] = typer.Option(None, help='Base build dir for virtual analysis diff'),
    lru_docs: Optional[str] = typer.Option(None, help='Dir with LRU PDF/DOCX documents'),
    config: str = typer.Option('config.yaml', help='Path to config.yaml'),
    output_format: str = typer.Option('docx', help='Output format: docx or markdown'),
) -> None:
    """Run all four DO-178C analyses and generate §11-compliant reports."""
    cfg = load_config(config)
    logger.info(f"RAG Engine — DAL {cfg.dal_level}, output: {cfg.output_dir}")

    reader = CodeReader(cache_dir=cfg.cache_dir)
    current_results = reader.read_directory(codebase)
    if not current_results:
        typer.echo(f"No C++ files found in {codebase}", err=True)
        raise typer.Exit(1)

    base_results = reader.read_directory(base_build) if base_build else current_results

    all_functions = reader.all_functions(current_results)
    gb = CallGraphBuilder(current_results)
    gb.build()

    search = SemanticSearch(model_name=cfg.embedding_model)
    search.index_functions(all_functions)

    lru_doc_texts = DocReader().read_lru_documents(lru_docs) if lru_docs else {}

    def run_virtual():
        return VirtualAnalyzer(base_results, current_results).analyze()

    def run_coupling():
        return CouplingAnalyzer(current_results, gb, search, lru_doc_texts).analyze()

    def run_dead():
        return DeadCodeDetector(gb, current_results, cfg.entry_points).analyze()

    def run_standards():
        return StandardsValidator(cfg, current_results).analyze()

    tasks = {'virtual': run_virtual, 'coupling': run_coupling,
             'dead_code': run_dead, 'standards': run_standards}
    results = {}

    for name, fn in tqdm(tasks.items(), desc='Analyses'):
        try:
            results[name] = fn()
            logger.info(f"  done: {name}")
        except Exception as exc:
            logger.error(f"  failed {name}: {exc}")
            raise

    llm = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)
    if cfg.ollama_enabled and llm.is_available():
        logger.info(f"LLM enrichment via {cfg.ollama_model}")

    gen = ReportGenerator(cfg)
    paths = gen.generate_all(results['virtual'], results['coupling'],
                             results['dead_code'], results['standards'],
                             format=output_format)  # type: ignore[arg-type]

    ChecklistFiller(cfg).fill_sqa_checklist(results['virtual'], results['dead_code'], results['standards'])

    typer.echo(f"\nReports in {cfg.output_dir}/")
    for p in paths:
        typer.echo(f"  {p}")
    typer.echo(f"  {cfg.output_dir}/sqa_checklist.md")

    critical = results['standards'].violations_by_severity.get('CRITICAL', 0)
    if critical > 0:
        typer.echo(f"\nWARNING: {critical} CRITICAL violation(s)", err=True)
        raise typer.Exit(1)


@app.command()
def validate_config(config: str = typer.Option('config.yaml')) -> None:
    """Validate configuration file."""
    cfg = load_config(config)
    typer.echo(f"Config OK — DAL {cfg.dal_level}, output: {cfg.output_dir}")


if __name__ == '__main__':
    app()
