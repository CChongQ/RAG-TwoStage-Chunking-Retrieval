import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.prompts import build_answer_prompt  # noqa: E402
from rag_chunking.results import (  # noqa: E402
    build_question_to_gold_answer_map,
    make_run_result,
)
from rag_chunking.retrieval import documents_to_texts, retrieve_contexts  # noqa: E402


class FakeDocument:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class InvokeRetriever:
    def __init__(self):
        self.search_kwargs = {}

    def invoke(self, query):
        return [FakeDocument(f"context for {query}")]


class SharedUtilityTests(unittest.TestCase):
    def test_build_answer_prompt_contains_question_and_contexts(self):
        prompt = build_answer_prompt("Who?", ["first", "second"])

        self.assertIn("Question: Who?", prompt)
        self.assertIn("first\n\nsecond", prompt)
        self.assertIn("I don't have the answer", prompt)

    def test_retrieve_contexts_uses_retriever_and_top_k(self):
        retriever = InvokeRetriever()

        contexts = retrieve_contexts(retriever, "query", top_k=3)

        self.assertEqual(contexts, ["context for query"])
        self.assertEqual(retriever.search_kwargs["k"], 3)

    def test_documents_to_texts_accepts_plain_objects(self):
        self.assertEqual(documents_to_texts([FakeDocument("a"), "b"]), ["a", "b"])

    def test_build_question_to_gold_answer_map(self):
        documents = [
            FakeDocument(metadata={"question_text": " q1 ", "gold_answer": {"long_answer": "a1"}}),
            FakeDocument(metadata={"question_text": "q2", "gold_answer": "a2"}),
        ]

        result = build_question_to_gold_answer_map(documents)

        self.assertEqual(result["q1"], {"long_answer": "a1"})
        self.assertEqual(result["q2"], "a2")

    def test_make_run_result_uses_shared_schema_keys(self):
        result = make_run_result("q", ["ctx"], "answer", {"short_answers": ["answer"]})

        self.assertEqual(
            set(result),
            {"input_question", "retrieved_contexts", "response", "gold_answer"},
        )


if __name__ == "__main__":
    unittest.main()