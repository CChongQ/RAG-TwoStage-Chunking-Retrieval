"""Proposition chunking helpers for the two-stage RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_L2_MODEL = "gpt-3.5-turbo"
DEFAULT_L1_TOP_K = 3
DEFAULT_L1_FETCH_K = 10
DEFAULT_RESULT_TOP_K = 10
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


PROPOSITION_SYSTEM_PROMPT = """
Decompose the Passage below into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in valid JSON.

Output format:
1. Present the results as a JSON object with a key `"sentences"`, and the value should be an array of strings.Return **only** a valid JSON object with the following format (no additional text or explanation):
{{
  "sentences": [
    "sentence1",
    "sentence2",
    "sentence3"
  ]
}}
"""


PROPOSITION_EXAMPLES = [
    {
        "document": "Title: ZEostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century.",
        "propositions": "['The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.', 'Georg Franck von Franckenau was a professor of medicine.', 'The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.']",
    }
]


@dataclass
class PropositionSentences:
    """Simple fallback shape for proposition results in tests and adapters."""

    sentences: list[str]


def get_sentences_model() -> type[Any]:
    """Create the Pydantic model expected by LangChain's output parser."""
    try:
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError("Proposition parsing requires pydantic.") from exc

    class Sentences(BaseModel):
        sentences: list[str]

    return Sentences


def create_proposition_prompt() -> Any:
    """Create the few-shot proposition prompt from the notebook."""
    try:
        from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
    except ImportError as exc:
        raise ImportError("create_proposition_prompt requires langchain-core.") from exc

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{document}"),
            ("ai", "{propositions}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=PROPOSITION_EXAMPLES,
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", PROPOSITION_SYSTEM_PROMPT),
            few_shot_prompt,
            ("human", "{document}"),
        ]
    )


def create_proposition_runnable(model: str = DEFAULT_L2_MODEL, *, api_key: str | None = None) -> Any:
    """Create the L2 proposition extraction runnable."""
    try:
        from langchain_core.output_parsers import PydanticOutputParser
    except ImportError:
        try:
            from langchain.output_parsers import PydanticOutputParser
        except ImportError as exc:
            raise ImportError(
                "create_proposition_runnable requires langchain-core or langchain."
            ) from exc

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "create_proposition_runnable requires langchain-openai or langchain-community."
            ) from exc

    parser = PydanticOutputParser(pydantic_object=get_sentences_model())
    prompt = create_proposition_prompt()
    llm = ChatOpenAI(model=model, openai_api_key=api_key)
    return prompt | llm | parser


def get_new_prop_doc(relevant_sections: list[Any], prop_results: list[Any]) -> list[Any]:
    """Put propositioned results into new LangChain documents."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError as exc:
            raise ImportError("get_new_prop_doc requires langchain-core or langchain.") from exc

    proposition_docs = []
    for original_doc, result in zip(relevant_sections, prop_results):
        source_metadata = getattr(original_doc, "metadata", {}) or {}
        for sentence in result.sentences:
            proposition_docs.append(
                Document(
                    page_content=sentence,
                    metadata={
                        "source_title": source_metadata.get("title") or source_metadata.get("Title"),
                        "source": source_metadata.get("source"),
                    },
                )
            )
    return proposition_docs


def proposition_results_to_documents(relevant_sections: list[Any], prop_results: list[Any]) -> list[Any]:
    """Backward-friendly alias for creating proposition documents."""
    return get_new_prop_doc(relevant_sections, prop_results)