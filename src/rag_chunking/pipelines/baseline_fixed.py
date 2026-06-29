"""Fixed-size baseline RAG pipeline"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.fixed_size import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_fixed_size_documents,
)
from rag_chunking.config import PROJECT_ROOT
from rag_chunking.generation import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generate_answer_from_contexts,
)
from rag_chunking.loaders import load_gold_answer_documents, load_question_documents
from rag_chunking.results import build_question_to_gold_answer_map, make_run_result, save_run_results
from rag_chunking.retrieval import retrieve_contexts
from rag_chunking.vectorstores import as_retriever, create_chroma_vector_store, load_chroma_vector_store


@dataclass(frozen=True)
class FixedBaselineConfig:
    """Configuration for the fixed-size baseline pipeline."""

    dataset_file: str = "gold_test_file_30.json"
    vector_dir: str = "Baseline_vector"
    output_path: str = "evaluation/run_results_baseline.json"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = 10
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    generation_model: str = DEFAULT_GENERATION_MODEL
    rebuild_vector_store: bool = True
    max_questions: int | None = None


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_questions(documents: list[Any]) -> list[str]:
    """Collect question texts from loaded dataset documents."""
    return [
        document.metadata["question_text"]
        for document in documents
        if "question_text" in getattr(document, "metadata", {})
    ]


def build_or_load_vector_store(
    documents: list[Any],
    config: FixedBaselineConfig,
) -> Any:
    """Build or load the baseline Chroma vector store."""
    vector_dir = _resolve_project_path(config.vector_dir)

    if config.rebuild_vector_store:
        if vector_dir.exists():
            shutil.rmtree(vector_dir)
        chunks = split_fixed_size_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        return create_chroma_vector_store(chunks, vector_dir)

    return load_chroma_vector_store(vector_dir)


def run_fixed_baseline(config: FixedBaselineConfig | None = None) -> list[dict[str, Any]]:
    """Run the fixed-size baseline and save results to JSON."""
    config = config or FixedBaselineConfig()

    documents = load_question_documents(config.dataset_file)
    gold_documents = load_gold_answer_documents(config.dataset_file)
    questions = collect_questions(documents)
    if config.max_questions is not None:
        questions = questions[: config.max_questions]

    question_to_gold = build_question_to_gold_answer_map(gold_documents)
    vector_store = build_or_load_vector_store(documents, config)
    retriever = as_retriever(vector_store, search_type="similarity", top_k=config.top_k)

    run_results = []
    for index, question in enumerate(questions, start=1):
        print(f"============={index} Question:{question}=============")
        contexts = retrieve_contexts(retriever, question)
        response = generate_answer_from_contexts(
            question,
            contexts,
            temperature=config.temperature,
            top_p=config.top_p,
            model=config.generation_model,
        )
        print("-------Final Answer:-------------")
        print(f"Question: {question}\n Response: {response}")

        run_results.append(
            make_run_result(
                question,
                contexts,
                response,
                question_to_gold.get(question.strip(), ""),
            )
        )

    output_path = _resolve_project_path(config.output_path)
    save_run_results(run_results, output_path)
    return run_results