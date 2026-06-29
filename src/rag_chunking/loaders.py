"""Document-loading helpers shared by RAG pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rag_chunking.config import DATASET_DIR


MetadataFunc = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def question_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata callback for records with a top-level ``question_text`` field."""
    metadata["question_text"] = record.get("question_text")
    metadata["document_url"] = record.get("document_url")
    metadata["example_id"] = record.get("example_id")
    return metadata


def gold_answer_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata callback that also keeps the record's gold answer."""
    question_metadata(record, metadata)
    metadata["gold_answer"] = record.get("gold_answer")
    return metadata


def grouped_question_metadata(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata callback for grouped document records with a ``questions`` list."""
    metadata["document_url"] = record.get("document_url")
    metadata["question_texts"] = [
        question.get("question_text")
        for question in record.get("questions", [])
        if question.get("question_text")
    ]
    return metadata


def resolve_dataset_file(file_path: str | Path) -> Path:
    """Resolve relative dataset filenames against the repository dataset folder."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    return DATASET_DIR / path


def load_json_documents(
    file_path: str | Path,
    *,
    jq_schema: str = ".[]",
    content_key: str = "document_text",
    metadata_func: MetadataFunc = question_metadata,
) -> list[Any]:
    """Load JSON records as LangChain documents using ``JSONLoader``.

    LangChain is imported lazily so this module remains importable in a minimal
    test environment.
    """
    try:
        from langchain_community.document_loaders import JSONLoader
    except ImportError as exc:
        raise ImportError(
            "load_json_documents requires langchain-community and jq."
        ) from exc

    loader = JSONLoader(
        file_path=str(resolve_dataset_file(file_path)),
        jq_schema=jq_schema,
        content_key=content_key,
        metadata_func=metadata_func,
    )
    return loader.load()


def load_question_documents(file_path: str | Path) -> list[Any]:
    """Load final-test style records with question metadata."""
    return load_json_documents(file_path, metadata_func=question_metadata)


def load_gold_answer_documents(file_path: str | Path) -> list[Any]:
    """Load final-test style records with gold answer metadata."""
    return load_json_documents(file_path, metadata_func=gold_answer_metadata)


def load_grouped_question_documents(file_path: str | Path) -> list[Any]:
    """Load grouped document records with a list of question texts."""
    return load_json_documents(file_path, metadata_func=grouped_question_metadata)