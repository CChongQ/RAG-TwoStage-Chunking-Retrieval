"""Fixed-size baseline RAG pipeline."""

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
from rag_chunking.retrieval import documents_to_texts, retrieve_documents
from rag_chunking.vectorstores import (
    DEFAULT_OPENAI_EMBED_MODEL,
    as_retriever,
    create_chroma_vector_store,
    create_openai_embedding,
    load_chroma_vector_store,
)


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
    embedding_model: str = DEFAULT_OPENAI_EMBED_MODEL
    generation_model: str = DEFAULT_GENERATION_MODEL
    rebuild_vector_store: bool = True
    max_questions: int | None = None
    verbose: bool = True


def _log(message: str, config: FixedBaselineConfig) -> None:
    if config.verbose:
        print(f"[baseline] {message}")


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
    embedding = create_openai_embedding(config.embedding_model)

    if config.rebuild_vector_store:
        _log(
            f"Chunking {len(documents)} documents with chunk_size={config.chunk_size}, "
            f"chunk_overlap={config.chunk_overlap}",
            config,
        )
        if vector_dir.exists():
            _log(f"Removing existing vector store at {vector_dir}", config)
            shutil.rmtree(vector_dir)
        chunks = split_fixed_size_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        _log(f"Created {len(chunks)} fixed-size chunks", config)
        _log(
            f"Embedding chunks with {config.embedding_model} and saving Chroma vector store to {vector_dir}",
            config,
        )
        vector_store = create_chroma_vector_store(chunks, vector_dir, embedding=embedding)
        _log("Vector store build complete", config)
        return vector_store

    if not vector_dir.exists():
        raise FileNotFoundError(
            f"Baseline vector store does not exist: {vector_dir}. "
            "Run with --rebuild-vector-store or choose an existing --vector-dir."
        )
    _log(f"Loading existing Chroma vector store from {vector_dir}", config)
    return load_chroma_vector_store(vector_dir, embedding=embedding)


def run_fixed_baseline(config: FixedBaselineConfig | None = None) -> list[dict[str, Any]]:
    """Run the fixed-size baseline and save results to JSON."""
    config = config or FixedBaselineConfig()

    _log(f"Loading test documents from {config.dataset_file}", config)
    documents = load_question_documents(config.dataset_file)
    _log(f"Loaded {len(documents)} documents for retrieval", config)

    _log("Loading gold answers for result records", config)
    gold_documents = load_gold_answer_documents(config.dataset_file)
    question_to_gold = build_question_to_gold_answer_map(gold_documents)
    _log(f"Loaded gold answers for {len(question_to_gold)} questions", config)

    questions = collect_questions(documents)
    _log(f"Collected {len(questions)} questions", config)
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
        _log(f"Limiting run to first {len(questions)} questions", config)

    vector_store = build_or_load_vector_store(documents, config)
    _log(f"Creating similarity retriever with top_k={config.top_k}", config)
    retriever = as_retriever(vector_store, search_type="similarity", top_k=config.top_k)

    run_results = []
    for index, question in enumerate(questions, start=1):
        print(f"============={index} Question:{question}=============")
        _log("Retrieving relevant chunks from vector store", config)
        retrieved_documents = retrieve_documents(retriever, question)
        contexts = documents_to_texts(retrieved_documents)
        _log(f"Retrieved {len(contexts)} contexts", config)
        for context_index, context in enumerate(contexts, start=1):
            preview = context.replace("\n", " ")[:160]
            _log(f"Context {context_index}: {preview}", config)

        _log(f"Generating answer with {config.generation_model}", config)
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
    _log(f"Saving {len(run_results)} results to {output_path}", config)
    save_run_results(run_results, output_path)
    _log("Fixed-size baseline run complete", config)
    return run_results
