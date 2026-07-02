"""Sentence-window Level 2 retrieval helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_WINDOW_SIZE = 3
DEFAULT_TOP_N = 10
DEFAULT_TOP_K_2 = 15
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


SENTENCE_WINDOW_PROMPT = (
    "Answer the question **directly and concisely** using only the provided context.\n"
    "- Do not repeat the question.\n"
    "- Do not include information not in the context.\n"
    "- If the answer is unclear or not found, say'I don't have the answer'.\n"
    "Question:{query_str}\n"
    "Relevant contents:{context_str}\n"
    "Answer: "
)


def create_node_index(
    documents_to_node: list[Any],
    window_size: int,
    folder_name: str | Path,
    *,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Any:
    """Create and persist a LlamaIndex sentence-window node index."""
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.node_parser import SentenceWindowNodeParser
        from llama_index.core.schema import Document as LlamaDocument
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
    except ImportError as exc:
        raise ImportError("create_node_index requires llama-index and llama-index OpenAI packages.") from exc

    doc_llama = [
        LlamaDocument(text=document.page_content, metadata=document.metadata)
        for document in documents_to_node
    ]

    # Level 2 chunking: split retrieved L1 sections into sentence windows.
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    Settings.llm = OpenAI(model=llm_model, temperature=llm_temperature)
    Settings.embed_model = OpenAIEmbedding(model=embed_model)
    Settings.node_parser = node_parser

    node_index = VectorStoreIndex.from_documents(doc_llama)
    node_index.storage_context.persist(persist_dir=str(folder_name))
    return node_index


def get_query_engine(
    top_n: int,
    top_k: int,
    rerank_model: str,
    node_index: Any,
) -> Any:
    """Create the Level 2 sentence-window query engine."""
    try:
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor
        from llama_index.core.postprocessor import SentenceTransformerRerank
    except ImportError as exc:
        raise ImportError("get_query_engine requires llama-index postprocessors.") from exc

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=top_n, model=rerank_model)

    # Level 2 query engine: retrieve sentence nodes, replace text with the
    # surrounding window metadata, then rerank the final contexts.
    return node_index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[postproc, rerank],
    )


def create_customized_query_engine(
    top_n: int,
    top_k: int,
    rerank_model: str,
    node_index: Any,
    prompt_str: str = SENTENCE_WINDOW_PROMPT,
) -> Any:
    """Create a query engine with the same concise-answer prompt as the notebook."""
    try:
        from llama_index.core import PromptTemplate
    except ImportError as exc:
        raise ImportError("create_customized_query_engine requires llama-index.") from exc

    customized_prompt = PromptTemplate(prompt_str)
    query_engine = get_query_engine(top_n, top_k, rerank_model, node_index)
    # Customize generation prompt using the same context-only answer style as
    # the proposition chunking notebook.
    query_engine.update_prompts({"response_synthesizer:text_qa_template": customized_prompt})
    return query_engine


def source_nodes_to_contexts(source_nodes: list[Any]) -> list[str]:
    """Extract text content from LlamaIndex response source nodes."""
    return [node.get_content() for node in source_nodes]