from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional
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

    def __init__(self, llm: OllamaClient, search: Optional[SemanticSearch], *,
                 standards: Any = None, code_index: Any = None) -> None:
        self._llm = llm
        self._search = search
        self._standards = standards        # StandardsDocument: hybrid-retrieved clauses of the uploaded standard
        self._code_index = code_index      # HybridRetriever over the codebase's functions

    @classmethod
    def for_results(cls, llm: OllamaClient, results: Any) -> 'RAGNarrativeGenerator':
        return cls(llm, getattr(results, 'search', None), standards=getattr(results, 'standards_doc', None),
                   code_index=getattr(results, 'code_index', None))

    def _system(self) -> str:
        base = _SYS if (self._search is not None or self._code_index is not None) else \
            _SYS.replace(_RETRIEVAL_SENTENCE, "You are given deterministic static-analysis facts. ")
        if self._standards is not None:
            base += (" Clauses retrieved from the project's coding standard are given with their source; quote the "
                     "clause and its source when you rely on it, and never invent a clause, limit or line number.")
        return base

    def _code_context(self, query: str, k: int, label: str = '') -> str:
        """The retrieved-code prompt section; empty without an index (the facts are then the only context)."""
        if self._code_index is not None:
            hits = self._code_index.search(query, k=k)
            logger.debug(f"RAG retrieve code: '{query}' → {len(hits)} chunks ({self._code_index.mode})")
            body = '\n'.join(f"  • {h.chunk.section} ({h.chunk.source}): {h.chunk.text[:300]}" for h in hits)
            return (f"=== RETRIEVED CODE CONTEXT ({self._code_index.mode}{label}) ===\n"
                    f"{body or 'No related functions retrieved.'}\n\n")
        if self._search is None:
            return ''
        related = self._search.find_related(query, k=k)
        logger.debug(f"RAG retrieve: '{query}' → {len(related)} functions")
        return (f"=== RETRIEVED CODE CONTEXT (FAISS semantic search{label}) ===\n"
                f"{_format_functions(related)}\n\n")

    def _standards_context(self, query: str, k: int = 4) -> str:
        if self._standards is None:
            return ''
        hits = self._standards.search(query, k=k)
        if not hits:
            return ''
        body = '\n'.join(f"  [{i}] {h.chunk.cite()}: {h.chunk.text[:600]}" for i, h in enumerate(hits, 1))
        return f"=== RETRIEVED CODING-STANDARD CLAUSES ({self._standards.retriever.mode}) ===\n{body}\n\n"

    def generate(self, retrieval_query: str, analysis_summary: str,
                 k: int = 8, max_tokens: int = 400) -> Optional[str]:
        prompt = (
            f"{self._system()}\n\n"
            f"{self._standards_context(retrieval_query)}"
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
            f"{self._standards_context(question)}"
            f"{self._code_context(question, k)}"
            f"=== {'ANALYSIS SUMMARY' if indexed else 'STATIC-ANALYSIS FACTS'} ===\n"
            f"{analysis_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"Answer based on the {'retrieved code and analysis results' if indexed else 'static-analysis facts'}:"
        )
        return self._llm.generate(prompt, max_tokens=max_tokens)
