"""Build the structure-based Level 1 Chroma vector store."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.structure_based import SectionRecord, StructureBasedChunker
from rag_chunking.config import PROJECT_ROOT
from rag_chunking.data_io import load_json
from rag_chunking.loaders import resolve_dataset_file
from rag_chunking.vectorstores import create_chroma_vector_store


@dataclass(frozen=True)
class L1VectorBuildConfig:
    """Configuration for building a structure-based L1 vector store."""

    dataset_file: str = "gold_test_file_30.json"
    vector_dir: str = "L1_vector_final"
    max_documents: int | None = None
    overwrite: bool = False


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_l1_sections(config: L1VectorBuildConfig | None = None) -> list[SectionRecord]:
    """Extract structure-based L1 sections from the configured dataset."""
    config = config or L1VectorBuildConfig()
    records: list[dict[str, Any]] = load_json(resolve_dataset_file(config.dataset_file))
    chunker = StructureBasedChunker()
    return chunker.chunk_documents(records, max_documents=config.max_documents)


def build_l1_vector_store(config: L1VectorBuildConfig | None = None) -> tuple[Any, list[SectionRecord]]:
    """Build and persist the L1 Chroma vector store."""
    config = config or L1VectorBuildConfig()
    vector_dir = _resolve_project_path(config.vector_dir)

    if vector_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Vector directory already exists: {vector_dir}. "
                "Pass overwrite=True or choose a different vector_dir."
            )
        shutil.rmtree(vector_dir)

    records: list[dict[str, Any]] = load_json(resolve_dataset_file(config.dataset_file))
    chunker = StructureBasedChunker()
    sections = chunker.chunk_documents(records, max_documents=config.max_documents)
    documents = chunker.to_langchain_documents(sections)
    vector_store = create_chroma_vector_store(documents, vector_dir)
    return vector_store, sections