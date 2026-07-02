"""Analysis helpers for evaluated RAG pipeline outputs."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any

from rag_chunking.data_io import load_json
from rag_chunking.evaluation.ragas_metrics import METRICS


QUESTION_TYPE_UNKNOWN = "Unknown"


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Evaluation analysis requires pandas.") from exc
    return pd


def load_eval_metrics_all_methods(file_paths: dict[str, str | Path]) -> dict[str, Any]:
    """Load each method's ``Evaluation`` records as pandas DataFrames."""
    pd = _pd()
    results = {}
    for name, path in file_paths.items():
        data = load_json(path)
        metrics = [record["Evaluation"] for record in data]
        results[name] = pd.DataFrame(metrics)
    return results


def confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return mean and normal-approximation confidence half-width."""
    values = [value for value in data if value is not None]
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        return mean(values), 0.0

    center = mean(values)
    standard_error = stdev(values) / sqrt(len(values))
    z_score = NormalDist().inv_cdf((1 + confidence) / 2)
    return center, standard_error * z_score


def summarize_metrics(eval_result: dict[str, Any]) -> Any:
    """Create the notebook-style mean/std summary table."""
    pd = _pd()
    formatted_summary = []

    for name, df in eval_result.items():
        row = {"Method": name}
        for metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            row[metric] = f"{mean_val:.3f} (+/-{std_val:.3f})"
        formatted_summary.append(row)

    summary_df = pd.DataFrame(formatted_summary)
    return summary_df.set_index("Method") if not summary_df.empty else summary_df


def combine_metric_frames(eval_result: dict[str, Any]) -> Any:
    """Combine all method DataFrames into one long-form DataFrame."""
    pd = _pd()
    return pd.concat([df.assign(Method=name) for name, df in eval_result.items()])


def get_question_type(
    question: str,
    fact_based_questions: set[str] | list[str],
    broad_questions: set[str] | list[str],
) -> str:
    """Classify a question using explicit fact-based and broad question lists."""
    if question in set(fact_based_questions):
        return "Fact-Based"
    if question in set(broad_questions):
        return "Broad"
    return QUESTION_TYPE_UNKNOWN


def question_type_summary(
    file_paths: dict[str, str | Path],
    fact_based_questions: set[str] | list[str],
    broad_questions: set[str] | list[str],
) -> Any:
    """Group metric means by question type for each method."""
    pd = _pd()
    all_stats = []

    for method, path in file_paths.items():
        data = load_json(path)
        df = pd.DataFrame([record["Evaluation"] for record in data])
        questions = [record["input_question"] for record in data]
        df["input_question"] = questions
        df["question_type"] = df["input_question"].apply(
            lambda question: get_question_type(question, fact_based_questions, broad_questions)
        )

        # Group by question type and compute mean scores.
        grouped = df.groupby("question_type")[[metric for metric in METRICS if metric in df]].mean()
        for question_type, row in grouped.iterrows():
            row_dict = row.to_dict()
            row_dict["Method"] = method
            row_dict["Question_Type"] = question_type
            all_stats.append(row_dict)

    summary_df = pd.DataFrame(all_stats)
    if summary_df.empty:
        return summary_df

    columns = ["Method", "Question_Type", *[metric for metric in METRICS if metric in summary_df]]
    return summary_df[columns].round(3)


def split_question_type_summary(summary_df: Any) -> tuple[Any, Any]:
    """Extract fact-based and broad summary tables."""
    columns_order = ["Method", *[metric for metric in METRICS if metric in summary_df]]
    fact_based = summary_df[summary_df["Question_Type"] == "Fact-Based"]
    broad = summary_df[summary_df["Question_Type"] == "Broad"]
    return (
        fact_based.drop(columns=["Question_Type"]).reset_index(drop=True)[columns_order],
        broad.drop(columns=["Question_Type"]).reset_index(drop=True)[columns_order],
    )