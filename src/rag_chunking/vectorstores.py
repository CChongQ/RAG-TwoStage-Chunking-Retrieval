"""Vector-store helpers shared by RAG pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_faiss_vector_store(documents: list[Any], embedding: Any | None = None) -> Any:
    """Create an in-memory FAISS vector store from LangChain documents."""
    try:
        from langchain.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "create_faiss_vector_store requires langchain and langchain-openai."
        ) from exc

    embedding_model = embedding or OpenAIEmbeddings()
    return FAISS.from_documents(documents, embedding=embedding_model)


def save_faiss_vector_store(
    documents: list[Any],
    folder_name: str | Path,
    embedding: Any | None = None,
) -> Any:
    """Create a FAISS vector store and save it locally."""
    vector_store = create_faiss_vector_store(documents, embedding=embedding)
    vector_store.save_local(str(folder_name))
    return vector_store


def create_chroma_vector_store(
    documents: list[Any],
    persist_directory: str | Path,
    embedding: Any | None = None,
) -> Any:
    """Create and persist a Chroma vector store from documents."""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "create_chroma_vector_store requires langchain-chroma and langchain-openai."
        ) from exc

    embedding_model = embedding or OpenAIEmbeddings()
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist_directory),
    )


def load_chroma_vector_store(
    persist_directory: str | Path,
    embedding: Any | None = None,
) -> Any:
    """Load a persisted Chroma vector store."""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "load_chroma_vector_store requires langchain-chroma and langchain-openai."
        ) from exc

    embedding_model = embedding or OpenAIEmbeddings()
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_model,
    )


def as_retriever(
    vector_store: Any,
    *,
    search_type: str = "similarity",
    top_k: int = 10,
    fetch_k: int | None = None,
) -> Any:
    """Create a retriever from a LangChain-compatible vector store."""
    search_kwargs = {"k": top_k}
    if fetch_k is not None:
        search_kwargs["fetch_k"] = fetch_k
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)