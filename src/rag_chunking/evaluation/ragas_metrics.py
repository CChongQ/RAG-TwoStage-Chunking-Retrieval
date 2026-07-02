"""RAGAS scoring helpers for pipeline run-result JSON files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_chunking.config import PROJECT_ROOT
from rag_chunking.data_io import load_json, save_json
from rag_chunking.schemas import EvaluationScores


METRICS = ["Context_Precision", "Context_Recall", "Response_Relevancy", "Faithfulness"]


@dataclass(frozen=True)
class RagasEvaluationConfig:
    """Configuration for evaluating a run-result JSON file with RAGAS."""

    input_path: str
    output_path: str | None = None
    evaluator_model: str = "gpt-3.5-turbo"
    embedding_model: str | None = None


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_eval_output_path(input_path: str | Path) -> Path:
    """Match the notebook's dated eval output filename."""
    today = datetime.today().strftime("%Y-%m-%d")
    input_name = Path(input_path).name
    return PROJECT_ROOT / "evaluation" / f"eval_{today}_{input_name}"


def combine_gold_reference(gold_answer: Any) -> str:
    """Combine long and short answers as the RAGAS reference string."""
    if not isinstance(gold_answer, dict):
        return str(gold_answer or "")

    long_answer = gold_answer.get("long_answer") or ""
    short_answers = gold_answer.get("short_answers") or []
    return f"{long_answer} {' '.join(short_answers)}".strip()


def empty_scores() -> EvaluationScores:
    """Return the notebook's default empty metric record."""
    return {
        "Context_Precision": None,
        "Context_Recall": None,
        "Response_Relevancy": None,
        "Faithfulness": None,
    }


def create_ragas_wrappers(evaluator_model: str = "gpt-3.5-turbo", embedding_model: str | None = None) -> tuple[Any, Any]:
    """Create RAGAS LLM and embedding wrappers."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError("RAGAS evaluation requires ragas and langchain-openai dependencies.") from exc

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=evaluator_model))
    embeddings = OpenAIEmbeddings(model=embedding_model) if embedding_model else OpenAIEmbeddings()
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    return evaluator_llm, evaluator_embeddings


async def evaluate_all_metrics(
    user_input: str | None,
    response: str | None,
    reference: str | None,
    retrieved_contexts: list[str] | None,
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> EvaluationScores:
    """Evaluate all RAGAS metrics for one single-turn sample."""
    results = empty_scores()

    # Skip evaluation if required fields are missing.
    if not response or not retrieved_contexts:
        return results

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise ImportError("evaluate_all_metrics requires ragas.") from exc

    sample = SingleTurnSample(
        user_input=user_input or "",
        response=response or "",
        reference=reference or "",
        retrieved_contexts=retrieved_contexts,
    )

    # Run metrics only if their required inputs are present.
    if retrieved_contexts and reference:
        context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
        results["Context_Precision"] = round(await context_precision.single_turn_ascore(sample), 4)

        context_recall = LLMContextRecall(llm=evaluator_llm)
        results["Context_Recall"] = round(await context_recall.single_turn_ascore(sample), 4)

    if user_input and response:
        response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        results["Response_Relevancy"] = round(await response_relevancy.single_turn_ascore(sample), 4)

    if response and retrieved_contexts:
        faithfulness = Faithfulness(llm=evaluator_llm)
        results["Faithfulness"] = round(await faithfulness.single_turn_ascore(sample), 4)

    return results


async def evaluate_all(
    run_results: list[dict[str, Any]],
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> list[dict[str, Any]]:
    """Evaluate every run-result item and attach an ``Evaluation`` field."""
    for item in run_results:
        user_input = item.get("input_question")
        print(f"Evaluating question: {user_input}...")

        # Combine long and short answers as reference.
        combined_reference = combine_gold_reference(item.get("gold_answer", {}))
        evaluation = await evaluate_all_metrics(
            user_input,
            item.get("response"),
            combined_reference,
            item.get("retrieved_contexts"),
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )
        item["Evaluation"] = evaluation
    return run_results


async def evaluate_run_file(config: RagasEvaluationConfig) -> list[dict[str, Any]]:
    """Load a run-result file, evaluate it, and save evaluated results."""
    input_path = _resolve_project_path(config.input_path)
    output_path = _resolve_project_path(config.output_path) if config.output_path else default_eval_output_path(input_path)
    evaluator_llm, evaluator_embeddings = create_ragas_wrappers(
        config.evaluator_model,
        config.embedding_model,
    )
    run_results = load_json(input_path)
    eval_result = await evaluate_all(
        run_results,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
    )

    # Save evaluation result.
    save_json(eval_result, output_path, indent=4)
    print(f"Saved evaluated results to {output_path}")
    return eval_result