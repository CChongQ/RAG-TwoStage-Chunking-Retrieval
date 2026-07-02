"""Retrieval helpers shared by RAG pipelines."""

from __future__ import annotations

from typing import Any


def _raise_retrieval_error(exc: Exception) -> None:
    message = str(exc)
    if "expecting embedding with dimension" in message and "got" in message:
        raise ValueError(
            "Vector-store embedding dimension mismatch. Rebuild the vector store with the same embedding_model configured for this pipeline. "
            f"Original error: {message}"
        ) from exc
    raise exc


def retrieve_documents(retriever: Any, query: str, top_k: int | None = None) -> list[Any]:
    """Retrieve documents"""
    if top_k is not None and hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = top_k

    try:
        if hasattr(retriever, "invoke"):
            return list(retriever.invoke(query))
        if hasattr(retriever, "get_relevant_documents"):
            return list(retriever.get_relevant_documents(query))
    except Exception as exc:
        _raise_retrieval_error(exc)

    raise TypeError("Retriever must provide invoke() or get_relevant_documents().")


def documents_to_texts(documents: list[Any]) -> list[str]:
    """Convert retrieved document objects to page-content strings."""
    return [getattr(document, "page_content", str(document)) for document in documents]


def retrieve_contexts(retriever: Any, query: str, top_k: int | None = None) -> list[str]:
    """Retrieve documents and return their text content."""
    return documents_to_texts(retrieve_documents(retriever, query, top_k=top_k))
