"""Two-stage structure + sentence-window RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.sentence_window import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K_2,
    DEFAULT_TOP_N,
    DEFAULT_WINDOW_SIZE,
    SENTENCE_WINDOW_PROMPT,
    create_customized_query_engine,
    create_node_index,
    source_nodes_to_contexts,
)
from rag_chunking.config import PROJECT_ROOT
from rag_chunking.loaders import load_gold_answer_documents
from rag_chunking.results import build_question_to_gold_answer_map, make_run_result, save_run_results
from rag_chunking.retrieval import retrieve_documents
from rag_chunking.vectorstores import as_retriever, load_chroma_vector_store


@dataclass(frozen=True)
class SentenceWindowConfig:
    """Configuration for the two-stage sentence-window pipeline."""

    dataset_file: str = "gold_test_file_30.json"
    l1_vector_dir: str = "L1_vector_final"
    node_index_dir: str = "L2_nodes_test"
    output_path: str = "evaluation/run_results_sentence_window.json"
    l1_top_k: int = 3
    l1_fetch_k: int = 10
    window_size: int = DEFAULT_WINDOW_SIZE
    top_n: int = DEFAULT_TOP_N
    top_k_2: int = DEFAULT_TOP_K_2
    rerank_model: str = DEFAULT_RERANK_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    embed_model: str = DEFAULT_EMBED_MODEL
    prompt_str: str = SENTENCE_WINDOW_PROMPT
    max_questions: int | None = None


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


def get_l1_relevant_sections(vector_store: Any, query: str, config: SentenceWindowConfig) -> list[Any]:
    """Run Level 1 retrieval and keep the top structural sections."""
    # Run retriever for the input query. The notebook fetched 10 from Chroma and
    # then sliced to the requested top_k sections.
    retriever = as_retriever(
        vector_store,
        search_type="similarity",
        top_k=config.l1_fetch_k,
    )
    return retrieve_documents(retriever, query)[: config.l1_top_k]


def run_sentence_window_pipeline(config: SentenceWindowConfig | None = None) -> list[dict[str, Any]]:
    """Run the two-stage structure + sentence-window RAG pipeline."""
    config = config or SentenceWindowConfig()

    # Load test set and collect gold answers.
    documents = load_gold_answer_documents(config.dataset_file)
    test_questions = collect_questions(documents)
    if config.max_questions is not None:
        test_questions = test_questions[: config.max_questions]
    question_to_gold_map = build_question_to_gold_answer_map(documents)

    # Load L1 vector store.
    l1_vectorstore = load_chroma_vector_store(_resolve_project_path(config.l1_vector_dir))

    all_results = []
    for index, query in enumerate(test_questions, start=1):
        print(f"============={index} Question:{query}=============")

        # Level 1 retrieval.
        relevant_sections = get_l1_relevant_sections(l1_vectorstore, query, config)

        # Level 2 chunking.
        node_index = create_node_index(
            relevant_sections,
            config.window_size,
            _resolve_project_path(config.node_index_dir),
            llm_model=config.llm_model,
            llm_temperature=config.llm_temperature,
            embed_model=config.embed_model,
        )

        # Level 2 query engine.
        sentence_window_engine = create_customized_query_engine(
            config.top_n,
            config.top_k_2,
            config.rerank_model,
            node_index,
            config.prompt_str,
        )

        # Level 2 retrieval and generation.
        response = sentence_window_engine.query(query)
        retrieved_contexts = source_nodes_to_contexts(response.source_nodes)
        print(f"Question: {query}")
        print(f"Response: {response.response}")
        for source_index, node in enumerate(response.source_nodes, start=1):
            print(f"No. {source_index} Source Node: {node.get_content()}")

        all_results.append(
            make_run_result(
                query,
                retrieved_contexts,
                response.response,
                question_to_gold_map.get(query.strip(), ""),
            )
        )

    save_run_results(all_results, _resolve_project_path(config.output_path))
    return all_results