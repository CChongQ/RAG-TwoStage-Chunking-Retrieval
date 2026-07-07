"""Structure-based baseline RAG pipeline using the Level 1 vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.config import PROJECT_ROOT
from rag_chunking.generation import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generate_answer_from_contexts,
)
from rag_chunking.loaders import load_gold_answer_documents
from rag_chunking.results import build_question_to_gold_answer_map, make_run_result, save_run_results
from rag_chunking.retrieval import documents_to_texts, retrieve_documents
from rag_chunking.vectorstores import (
    DEFAULT_OPENAI_EMBED_MODEL,
    as_retriever,
    create_openai_embedding,
    load_chroma_vector_store,
)


@dataclass(frozen=True)
class StructureBaselineConfig:
    """Configuration for the structure-based baseline pipeline."""

    dataset_file: str = "gold_test_file_30.json"
    l1_vector_dir: str = "artifacts/vectorstores/L1_vector_final"
    output_path: str = "artifacts/run_results/run_results_baseline_L1.json"
    top_k: int = 2
    embedding_model: str = DEFAULT_OPENAI_EMBED_MODEL
    generation_model: str = DEFAULT_GENERATION_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_questions: int | None = None
    verbose: bool = True


def _log(message: str, config: StructureBaselineConfig) -> None:
    if config.verbose:
        print(f"[structure-baseline] {message}")


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_questions(documents: list[Any]) -> list[str]:
    """Collect query strings from loaded dataset documents."""
    return [
        document.metadata["question_text"]
        for document in documents
        if "question_text" in getattr(document, "metadata", {})
    ]


def run_structure_baseline(config: StructureBaselineConfig | None = None) -> list[dict[str, Any]]:
    """Run the structure-based baseline and save raw run results."""
    config = config or StructureBaselineConfig()

    _log(f"Loading test documents from {config.dataset_file}", config)
    documents = load_gold_answer_documents(config.dataset_file)
    questions = collect_questions(documents)
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
        _log(f"Limiting run to first {len(questions)} questions", config)
    question_to_gold = build_question_to_gold_answer_map(documents)

    # Load the prebuilt Level 1 structure-based vector store.
    embedding = create_openai_embedding(config.embedding_model)
    vector_store = load_chroma_vector_store(
        _resolve_project_path(config.l1_vector_dir),
        embedding=embedding,
    )
    retriever = as_retriever(vector_store, search_type="similarity", top_k=config.top_k)

    run_results = []
    for index, question in enumerate(questions, start=1):
        print(f"============={index} Question:{question}=============")
        # Retrieve structure-based sections directly and generate from them.
        retrieved_documents = retrieve_documents(retriever, question)
        contexts = documents_to_texts(retrieved_documents)
        _log(f"Retrieved {len(contexts)} structure-based contexts", config)

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

    # Save raw run results for later RAGAS evaluation.
    output_path = _resolve_project_path(config.output_path)
    _log(f"Saving {len(run_results)} results to {output_path}", config)
    save_run_results(run_results, output_path)
    _log("Structure-based baseline run complete", config)
    return run_results