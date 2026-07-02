import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.evaluation.analysis import (  # noqa: E402
    confidence_interval,
    get_question_type,
    load_eval_metrics_all_methods,
    question_type_summary,
    summarize_metrics,
)


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is not installed")
class EvaluationAnalysisTests(unittest.TestCase):
    def test_confidence_interval_single_value_has_zero_width(self):
        center, half_width = confidence_interval([0.5])

        self.assertEqual(center, 0.5)
        self.assertEqual(half_width, 0.0)

    def test_get_question_type(self):
        self.assertEqual(get_question_type("q1", {"q1"}, set()), "Fact-Based")
        self.assertEqual(get_question_type("q2", set(), {"q2"}), "Broad")
        self.assertEqual(get_question_type("q3", set(), set()), "Unknown")

    @patch("rag_chunking.evaluation.analysis.load_json")
    def test_load_eval_metrics_all_methods(self, load_json):
        load_json.return_value = [
            {"Evaluation": {"Context_Precision": 1.0, "Faithfulness": 0.5}}
        ]

        result = load_eval_metrics_all_methods({"Baseline": "path.json"})

        self.assertIn("Baseline", result)
        self.assertEqual(result["Baseline"].iloc[0]["Context_Precision"], 1.0)

    def test_summarize_metrics_formats_mean_and_std(self):
        import pandas as pd

        eval_result = {
            "A": pd.DataFrame(
                [
                    {"Context_Precision": 1.0, "Faithfulness": 0.5},
                    {"Context_Precision": 0.0, "Faithfulness": 1.0},
                ]
            )
        }

        summary = summarize_metrics(eval_result)

        self.assertIn("Context_Precision", summary.columns)
        self.assertTrue(summary.loc["A", "Context_Precision"].startswith("0.500"))

    @patch("rag_chunking.evaluation.analysis.load_json")
    def test_question_type_summary_groups_by_question_type(self, load_json):
        load_json.return_value = [
            {
                "input_question": "fact q",
                "Evaluation": {
                    "Context_Precision": 1.0,
                    "Context_Recall": 0.5,
                    "Response_Relevancy": 1.0,
                    "Faithfulness": 0.5,
                },
            }
        ]

        summary = question_type_summary({"A": "path.json"}, {"fact q"}, set())

        self.assertEqual(summary.iloc[0]["Question_Type"], "Fact-Based")
        self.assertEqual(summary.iloc[0]["Context_Precision"], 1.0)


if __name__ == "__main__":
    unittest.main()