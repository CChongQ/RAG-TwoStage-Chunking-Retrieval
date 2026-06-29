import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.data_io import get_n_sample_from_file, has_valid_annotation  # noqa: E402


class DataIoTests(unittest.TestCase):
    def test_has_valid_annotation_accepts_long_answer(self):
        entry = {
            "annotations": [
                {
                    "long_answer": {"start_token": 1, "end_token": 3},
                    "short_answers": [],
                }
            ]
        }

        self.assertTrue(has_valid_annotation(entry))

    def test_has_valid_annotation_rejects_missing_spans(self):
        entry = {
            "annotations": [
                {
                    "long_answer": {"start_token": -1, "end_token": -1},
                    "short_answers": [],
                }
            ]
        }

        self.assertFalse(has_valid_annotation(entry))

    def test_sample_from_file_filters_long_gold_answers(self):
        data = [
            {"id": 1, "gold_answer": {"long_answer": "short"}},
            {"id": 2, "gold_answer": {"long_answer": "x" * 10}},
        ]

        with patch("rag_chunking.data_io.load_json", return_value=data), patch(
            "rag_chunking.data_io.save_json", return_value=Path("output.json")
        ) as save_json:
            output_path = get_n_sample_from_file(
                "input.json",
                dataset_size=1,
                output_filename="output.json",
                max_gold_length=5,
                seed=7,
            )

        self.assertEqual(output_path, Path("output.json"))
        save_json.assert_called_once_with(
            [{"id": 1, "gold_answer": {"long_answer": "short"}}],
            "output.json",
        )


if __name__ == "__main__":
    unittest.main()
