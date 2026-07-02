"""Gold-answer extraction helpers for Natural Questions records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_chunking.data_io import load_json, save_json


def clean_token(token: str) -> str:
    """Remove simple HTML-style marker tokens such as ``<P>`` and ``</Table>``."""
    return "" if token.startswith("<") and token.endswith(">") else token


def extract_gold_answer_for_question(
    tokens: list[str],
    annotations: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    """Extract long and short gold-answer text from token offsets."""
    annotation = annotations[0] if annotations else {}

    long_answer = None
    long_start = annotation.get("long_answer", {}).get("start_token", -1)
    long_end = annotation.get("long_answer", {}).get("end_token", -1)
    if long_start >= 0 and long_end > long_start:
        long_span_tokens = tokens[long_start:long_end]
        long_answer = " ".join(
            cleaned for token in long_span_tokens if (cleaned := clean_token(token))
        )

    short_answers = []
    for short_answer in annotation.get("short_answers", []):
        short_start = short_answer.get("start_token", -1)
        short_end = short_answer.get("end_token", -1)
        if short_start >= 0 and short_end > short_start:
            short_span_tokens = tokens[short_start:short_end]
            short_answers.append(
                " ".join(
                    cleaned
                    for token in short_span_tokens
                    if (cleaned := clean_token(token))
                )
            )

    return long_answer, short_answers


def add_gold_answers_to_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a ``gold_answer`` field to Natural Questions records."""
    for record in records:
        tokens = record["document_text"].split()
        long_answer, short_answers = extract_gold_answer_for_question(
            tokens,
            record.get("annotations", []),
        )
        record["gold_answer"] = {
            "long_answer": long_answer,
            "short_answers": short_answers,
        }
    return records


def get_all_gold_answers(input_path: str | Path, output_path: str | Path) -> Path:
    """Load records, add gold answers, and save the result."""
    records = load_json(input_path)
    return save_json(add_gold_answers_to_records(records), output_path)
