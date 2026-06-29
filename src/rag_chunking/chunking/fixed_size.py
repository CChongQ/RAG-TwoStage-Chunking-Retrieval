"""Fixed-size chunking helpers for the baseline RAG pipeline."""

from __future__ import annotations

from typing import Any


DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 20


def split_fixed_size_documents(
    documents: list[Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    add_start_index: bool = True,
) -> list[Any]:
    """Split documents using fix size chunking """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "split_fixed_size_documents requires langchain-text-splitters."
        ) from exc

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    return text_splitter.split_documents(documents)