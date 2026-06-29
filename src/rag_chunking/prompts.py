"""Prompt templates shared by RAG pipelines."""

ANSWER_FROM_CONTEXT_TEMPLATE = """Answer the question **directly and concisely** using only the provided context.
- Do not repeat the question.
- Do not include information not in the context.
- If the answer is unclear or not found, say 'I don't have the answer.'

Question: {question}
Relevant contents:{context}
"""


def build_answer_prompt(question: str, contexts: list[str]) -> str:
    """Build the shared answer-generation prompt used by the notebooks."""
    return ANSWER_FROM_CONTEXT_TEMPLATE.format(
        question=question,
        context="\n\n".join(contexts),
    )