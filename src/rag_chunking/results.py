"""Helpers for building and saving pipeline run results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_chunking.data_io import save_json
from rag_chunking.schemas import GoldAnswer, RunResult


def build_question_to_gold_answer_map(documents: list[Any]) -> dict[str, GoldAnswer | str]:
    """Build a map from question text to answer from loaded documents."""

    question_to_gold_answer: dict[str, GoldAnswer | str] = {}
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        question = metadata.get("question_text")
        if question:
            question_to_gold_answer[question.strip()] = metadata.get("gold_answer", "")
    return question_to_gold_answer


def make_run_result(
    question: str,
    retrieved_contexts: list[str],
    response: str,
    gold_answer: GoldAnswer | str,
) -> RunResult:
    """Create a run-result record using the shared output schema."""
    return {
        "input_question": question,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
        "gold_answer": gold_answer,
    }


def save_run_results(run_results: list[RunResult], output_path: str | Path) -> Path:
    """Save pipeline run results as UTF-8 JSON."""
    return save_json(run_results, output_path)
