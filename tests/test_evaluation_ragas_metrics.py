import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.evaluation.ragas_metrics import (  # noqa: E402
    combine_gold_reference,
    default_eval_output_path,
    empty_scores,
    evaluate_all,
    evaluate_all_metrics,
)


class RagasMetricsTests(unittest.TestCase):
    def test_combine_gold_reference_joins_long_and_short_answers(self):
        gold = {"long_answer": "Long", "short_answers": ["Short1", "Short2"]}

        self.assertEqual(combine_gold_reference(gold), "Long Short1 Short2")

    def test_combine_gold_reference_handles_non_dict(self):
        self.assertEqual(combine_gold_reference("answer"), "answer")

    def test_empty_scores_uses_expected_metric_keys(self):
        self.assertEqual(
            empty_scores(),
            {
                "Context_Precision": None,
                "Context_Recall": None,
                "Response_Relevancy": None,
                "Faithfulness": None,
            },
        )

    def test_default_eval_output_path_matches_notebook_pattern(self):
        output = default_eval_output_path("evaluation/run_results_baseline.json")

        self.assertEqual(output.parent.name, "evaluation")
        self.assertTrue(output.name.startswith("eval_"))
        self.assertTrue(output.name.endswith("run_results_baseline.json"))

    def test_evaluate_all_metrics_skips_missing_response(self):
        result = asyncio.run(
            evaluate_all_metrics(
                "question",
                "",
                "reference",
                ["context"],
                evaluator_llm=object(),
                evaluator_embeddings=object(),
            )
        )

        self.assertEqual(result, empty_scores())

    @patch("builtins.print")
    def test_evaluate_all_attaches_scores_from_metric_function(self, _print):
        async def fake_evaluate_all_metrics(*args, **kwargs):
            return {"Context_Precision": 1.0}

        import rag_chunking.evaluation.ragas_metrics as metrics_module

        original = metrics_module.evaluate_all_metrics
        metrics_module.evaluate_all_metrics = fake_evaluate_all_metrics
        try:
            data = [
                {
                    "input_question": "q",
                    "response": "r",
                    "gold_answer": {"long_answer": "g", "short_answers": []},
                    "retrieved_contexts": ["c"],
                }
            ]
            result = asyncio.run(
                evaluate_all(data, evaluator_llm=object(), evaluator_embeddings=object())
            )
        finally:
            metrics_module.evaluate_all_metrics = original

        self.assertEqual(result[0]["Evaluation"], {"Context_Precision": 1.0})


if __name__ == "__main__":
    unittest.main()