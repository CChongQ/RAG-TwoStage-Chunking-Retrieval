"""Shared data structures for pipeline inputs and outputs."""

from __future__ import annotations

from typing import Any, TypedDict


class GoldAnswer(TypedDict, total=False):
    """Gold answer attached to Natural Questions examples."""

    long_answer: str | None
    short_answers: list[str]


class RunResult(TypedDict):
    """Common output schema produced by RAG pipeline notebooks."""

    input_question: str
    retrieved_contexts: list[str]
    response: str
    gold_answer: GoldAnswer | str


class EvaluationScores(TypedDict, total=False):
    """RAGAS metric scores attached after evaluation."""

    Context_Precision: float | None
    Context_Recall: float | None
    Response_Relevancy: float | None
    Faithfulness: float | None


class EvaluatedRunResult(RunResult, total=False):
    """Run result after evaluation metrics have been added."""

    Evaluation: EvaluationScores


DocumentLike = Any