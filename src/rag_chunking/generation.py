"""Answer-generation helpers shared by RAG pipelines."""

from __future__ import annotations

import os
from typing import Any

from rag_chunking.prompts import build_answer_prompt


DEFAULT_GENERATION_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9


def get_api_response(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    *,
    model: str = DEFAULT_GENERATION_MODEL,
) -> str:
    """Call the OpenAI chat-completions API and return message text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


def create_openai_client(api_key: str | None = None) -> Any:
    """Create an OpenAI client using an explicit key or ``OPENAI_API_KEY``."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("create_openai_client requires the openai package.") from exc

    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))


def generate_answer_from_contexts(
    question: str,
    contexts: list[str],
    *,
    client: Any | None = None,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    model: str = DEFAULT_GENERATION_MODEL,
) -> str:
    """Generate an answer with the shared context-only prompt."""
    openai_client = client or create_openai_client()
    user_prompt = build_answer_prompt(question, contexts)
    return get_api_response(
        openai_client,
        system_prompt,
        user_prompt,
        temperature,
        top_p,
        model=model,
    )