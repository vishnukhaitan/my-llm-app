"""RAG helpers. A2 load, A3 chunks, A4 embeddings, A5 retrieve."""

from src.rag.ask_rag import ask_rag
from src.rag.chunker import chunk_documents, chunk_text
from src.rag.corpus import clean_text, load_corpus, parse_document
from src.rag.embeddings import build_index, embed_batch, load_index, save_index
from src.rag.retrieve import cosine, retrieve

__all__ = [
    "ask_rag",
    "build_index",
    "chunk_documents",
    "chunk_text",
    "clean_text",
    "cosine",
    "embed_batch",
    "load_corpus",
    "load_index",
    "parse_document",
    "retrieve",
    "save_index",
]
