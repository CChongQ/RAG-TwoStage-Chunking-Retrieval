import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.gold_answers import (  # noqa: E402
    add_gold_answers_to_records,
    clean_token,
    extract_gold_answer_for_question,
)


class GoldAnswerTests(unittest.TestCase):
    def test_clean_token_removes_html_markers(self):
        self.assertEqual(clean_token("<P>"), "")
        self.assertEqual(clean_token("answer"), "answer")

    def test_extract_gold_answer_for_question(self):
        tokens = ["<P>", "The", "answer", "is", "Paris", "</P>"]
        annotations = [
            {
                "long_answer": {"start_token": 0, "end_token": 6},
                "short_answers": [{"start_token": 4, "end_token": 5}],
            }
        ]

        long_answer, short_answers = extract_gold_answer_for_question(tokens, annotations)

        self.assertEqual(long_answer, "The answer is Paris")
        self.assertEqual(short_answers, ["Paris"])

    def test_add_gold_answers_to_records(self):
        records = [
            {
                "document_text": "<P> The answer is Paris </P>",
                "annotations": [
                    {
                        "long_answer": {"start_token": 0, "end_token": 6},
                        "short_answers": [{"start_token": 4, "end_token": 5}],
                    }
                ],
            }
        ]

        result = add_gold_answers_to_records(records)

        self.assertEqual(
            result[0]["gold_answer"],
            {"long_answer": "The answer is Paris", "short_answers": ["Paris"]},
        )


if __name__ == "__main__":
    unittest.main()
