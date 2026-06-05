from __future__ import annotations
from typing import List, Optional
from loguru import logger

from rag_engine.core.embeddings import SemanticSearch
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.models import FunctionDef

_SYS = (
    "You are a DO-178C certification engineer reviewing avionics software. "
    "You have access to retrieved code context from a semantic search index. "
    "Provide a concise, actionable certification assessment (3-5 sentences). "
    "Reference specific functions, files, or line numbers from the context where relevant. "
    "Focus on certification risk, required actions, and DAL implications."
)


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
    def __init__(self, llm: OllamaClient, search: SemanticSearch) -> None:
        self._llm = llm
        self._search = search

    def generate(self, retrieval_query: str, analysis_summary: str,
                 k: int = 8, max_tokens: int = 400) -> Optional[str]:
        related = self._search.find_related(retrieval_query, k=k)
        context = _format_functions(related)
        prompt = (
            f"{_SYS}\n\n"
            f"=== RETRIEVED CODE CONTEXT (FAISS semantic search, query: '{retrieval_query}') ===\n"
            f"{context}\n\n"
            f"=== ANALYSIS RESULTS ===\n"
            f"{analysis_summary}\n\n"
            f"Provide your DO-178C certification assessment:"
        )
        logger.debug(f"RAG retrieve: '{retrieval_query}' → {len(related)} functions")
        result = self._llm.generate(prompt, max_tokens=max_tokens)
        return result if result else None

    def answer_question(self, question: str, analysis_context: str,
                        k: int = 8, max_tokens: int = 500) -> Optional[str]:
        related = self._search.find_related(question, k=k)
        context = _format_functions(related)
        prompt = (
            f"{_SYS}\n\n"
            f"=== RETRIEVED CODE CONTEXT (FAISS semantic search) ===\n"
            f"{context}\n\n"
            f"=== ANALYSIS SUMMARY ===\n"
            f"{analysis_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"Answer based on the retrieved code and analysis results:"
        )
        logger.debug(f"RAG chat: '{question}' → {len(related)} functions retrieved")
        return self._llm.generate(prompt, max_tokens=max_tokens)
