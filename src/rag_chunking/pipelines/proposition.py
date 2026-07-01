"""Two-stage structure + proposition RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_chunking.chunking.proposition import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_L1_FETCH_K,
    DEFAULT_L1_TOP_K,
    DEFAULT_L2_MODEL,
    DEFAULT_RESULT_TOP_K,
    create_proposition_runnable,
    get_new_prop_doc,
)
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
from rag_chunking.vectorstores import as_retriever, create_openai_embedding, load_chroma_vector_store, save_faiss_vector_store


@dataclass(frozen=True)
class PropositionPipelineConfig:
    """Configuration for the two-stage proposition pipeline."""

    dataset_file: str = "gold_test_file_30.json"
    l1_vector_dir: str = "L1_vector_final"
    l2_vector_dir: str = "L2_vector"
    output_path: str | None = None
    l1_top_k: int = DEFAULT_L1_TOP_K
    l1_fetch_k: int = DEFAULT_L1_FETCH_K
    result_top_k: int = DEFAULT_RESULT_TOP_K
    l2_model: str = DEFAULT_L2_MODEL
    embedding_model: str = DEFAULT_EMBED_MODEL
    generation_model: str = DEFAULT_GENERATION_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    mode: str = "prod"
    max_questions: int | None = None


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_output_path(l2_model: str) -> Path:
    """Match the notebook's dated proposition output filename."""
    today = datetime.today().strftime("%Y-%m-%d")
    return PROJECT_ROOT / "evaluation" / f"run_results_proposition_{l2_model}_{today}.json"




def collect_questions(documents: list[Any]) -> list[str]:
    """Collect query strings from loaded dataset documents."""
    return [
        document.metadata["question_text"]
        for document in documents
        if "question_text" in getattr(document, "metadata", {})
    ]


def get_l2_retrieval_result(
    query: str,
    l1_retriever: Any,
    proposition_runnable: Any,
    config: PropositionPipelineConfig,
    *,
    embedding: Any | None = None,
) -> list[str]:
    """Complete L1 retrieval + L2 chunking and retrieval for one query."""
    # Step 1: Get L1 relevant sections based on the input question.
    l1_relevant_sections = retrieve_documents(l1_retriever, query)
    print(f"--- L1 Retrieved {len(l1_relevant_sections)} relavant sections")
    l1_relevant_sections_texts = documents_to_texts(l1_relevant_sections)

    # Step 2: Start L2 based on L1 result.
    # Run Proposition Model.
    print("Running Proposition...")
    l2_prop_results = [
        proposition_runnable.invoke({"document": text}) for text in l1_relevant_sections_texts
    ]
    if not l2_prop_results:
        return []

    # Put propositioned result into a new doc.
    proposition_docs = get_new_prop_doc(l1_relevant_sections, l2_prop_results)
    print(f"{len(proposition_docs)} proposition docs to be embeded")
    if config.mode == "test":
        for idx, each_doc in enumerate(proposition_docs, start=1):
            print(f"proposition {idx}: {each_doc.page_content}")

    if len(proposition_docs) <= 1:
        return []

    # Save L2 result to vector database.
    l2_vectorstore = save_faiss_vector_store(
        proposition_docs,
        _resolve_project_path(config.l2_vector_dir),
        embedding=embedding,
    )
    l2_retriever = as_retriever(
        l2_vectorstore,
        search_type="similarity",
        top_k=config.result_top_k,
    )

    # L2 Retrieval.
    l2_relevant_sections = retrieve_documents(l2_retriever, query)
    l2_relevant_sec_text = documents_to_texts(l2_relevant_sections)
    print(f"--L2 Retrieved {len(l2_relevant_sec_text)} relavant sections")
    return l2_relevant_sec_text


def run_proposition_pipeline(config: PropositionPipelineConfig | None = None) -> list[dict[str, Any]]:
    """Run the two-stage structure + proposition RAG pipeline."""
    config = config or PropositionPipelineConfig()

    documents = load_gold_answer_documents(config.dataset_file)
    questions = collect_questions(documents)
    if config.max_questions is not None:
        questions = questions[: config.max_questions]
    question_to_gold_map = build_question_to_gold_answer_map(documents)

    embeddings = create_openai_embedding(config.embedding_model)
    l1_vectorstore = load_chroma_vector_store(_resolve_project_path(config.l1_vector_dir), embedding=embeddings)

    # Set L1 retriever.
    l1_retriever = as_retriever(
        l1_vectorstore,
        search_type="mmr",
        top_k=config.l1_top_k,
        fetch_k=config.l1_fetch_k,
    )

    # L2 Model setup.
    proposition_runnable = create_proposition_runnable(config.l2_model)

    run_results = []
    for index, question in enumerate(questions, start=1):
        print(f"============={index} Question:{question}=============")
        relevant_content = get_l2_retrieval_result(
            question,
            l1_retriever,
            proposition_runnable,
            config,
            embedding=embeddings,
        )

        # Convert document into text before passing into the prompt.
        response = generate_answer_from_contexts(
            question,
            relevant_content,
            temperature=config.temperature,
            top_p=config.top_p,
            model=config.generation_model,
        )
        print("-------Final Answer:-------------")
        print(f"Question: {question}\n Response: {response}")

        # Save to output file.
        run_results.append(
            make_run_result(
                question,
                relevant_content,
                response,
                question_to_gold_map.get(question.strip(), ""),
            )
        )

        if config.mode == "test":
            break

    output_path = _resolve_project_path(config.output_path) if config.output_path else default_output_path(config.l2_model)
    save_run_results(run_results, output_path)
    return run_results