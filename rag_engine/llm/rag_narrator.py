from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from loguru import logger

from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.models import FunctionDef

if TYPE_CHECKING:   # annotation only; importing it would load the embedding models
    from rag_engine.core.embeddings import SemanticSearch

_SYS = (
    "You are a DO-178C certification engineer reviewing avionics software. "
    "You have access to retrieved code context from a semantic search index. "
    "Provide a concise, actionable certification assessment (3-5 sentences). "
    "Reference specific functions, files, or line numbers from the context where relevant. "
    "Focus on certification risk, required actions, and DAL implications."
)
_RETRIEVAL_SENTENCE = "You have access to retrieved code context from a semantic search index. "


def _format_functions(functions: List[FunctionDef], max_funcs: int = 8) -> str:
    if not functions:
        return "No related functions retrieved from index."
    lines = []
    for fn in functions[:max_funcs]:
        params = ", ".join(f"{p.type_} {p.name}" for p in fn.parameters) if fn.parameters else ""
        sig = f"{fn.return_type} {fn.name}({params})"
        flags = []
        if fn.is_virtual:
            flags.append("virtual")
        if fn.is_static:
            flags.append("static")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"  • {sig}{flag_str}\n"
            f"    file: {fn.file_path}:{fn.line_number} | "
            f"CC={fn.cyclomatic_complexity} | lines={fn.line_count}"
        )
    return "\n".join(lines)


class RAGNarrativeGenerator:
    """Prompts the local LLM with retrieved code (when a semantic index exists) and deterministic facts."""

    def __init__(self, llm: OllamaClient, search: Optional[SemanticSearch]) -> None:
        self._llm = llm
        self._search = search

    def _system(self) -> str:
        if self._search is not None:
            return _SYS
        return _SYS.replace(_RETRIEVAL_SENTENCE, "You are given deterministic static-analysis facts. ")

    def _code_context(self, query: str, k: int, label: str = '') -> str:
        """The retrieved-code prompt section; empty without a semantic index (the facts are then the only context)."""
        if self._search is None:
            return ''
        related = self._search.find_related(query, k=k)
        logger.debug(f"RAG retrieve: '{query}' → {len(related)} functions")
        return (f"=== RETRIEVED CODE CONTEXT (FAISS semantic search{label}) ===\n"
                f"{_format_functions(related)}\n\n")

    def generate(self, retrieval_query: str, analysis_summary: str,
                 k: int = 8, max_tokens: int = 400) -> Optional[str]:
        prompt = (
            f"{self._system()}\n\n"
            f"{self._code_context(retrieval_query, k, f', query: {retrieval_query!r}')}"
            f"=== ANALYSIS RESULTS ===\n"
            f"{analysis_summary}\n\n"
            f"Provide your DO-178C certification assessment:"
        )
        result = self._llm.generate(prompt, max_tokens=max_tokens)
        return result if result else None

    def answer_question(self, question: str, analysis_context: str,
                        k: int = 8, max_tokens: int = 500) -> Optional[str]:
        indexed = self._search is not None
        prompt = (
            f"{self._system()}\n\n"
            f"{self._code_context(question, k)}"
            f"=== {'ANALYSIS SUMMARY' if indexed else 'STATIC-ANALYSIS FACTS'} ===\n"
            f"{analysis_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"Answer based on the {'retrieved code and analysis results' if indexed else 'static-analysis facts'}:"
        )
        return self._llm.generate(prompt, max_tokens=max_tokens)
