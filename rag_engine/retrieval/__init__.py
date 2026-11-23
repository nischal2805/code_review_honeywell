"""Retrieval-augmented generation building blocks: document ingestion and chunking, BM25, dense embeddings
with a re-embedding cache, reciprocal-rank fusion and reranking. Everything degrades to lexical retrieval
when the embedding or reranking models are unavailable (fully offline installs)."""
from rag_engine.retrieval.chunking import Block, Chunk, chunk_blocks, chunk_functions, read_blocks, sentences
from rag_engine.retrieval.retriever import Hit, HybridRetriever

__all__ = ['Block', 'Chunk', 'Hit', 'HybridRetriever', 'chunk_blocks', 'chunk_functions', 'read_blocks', 'sentences']
