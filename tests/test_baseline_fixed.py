import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.pipelines.baseline_fixed import (  # noqa: E402
    FixedBaselineConfig,
    build_or_load_vector_store,
    collect_questions,
)


class FakeDocument:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}


class BaselineFixedTests(unittest.TestCase):
    def test_collect_questions_reads_metadata(self):
        documents = [
            FakeDocument({"question_text": "q1"}),
            FakeDocument({}),
            FakeDocument({"question_text": "q2"}),
        ]

        self.assertEqual(collect_questions(documents), ["q1", "q2"])

    @patch("rag_chunking.pipelines.baseline_fixed.create_chroma_vector_store")
    @patch("rag_chunking.pipelines.baseline_fixed.split_fixed_size_documents")
    def test_build_vector_store_uses_fixed_chunk_settings(self, split_docs, create_store):
        split_docs.return_value = ["chunk"]
        create_store.return_value = "vector-store"
        config = FixedBaselineConfig(
            vector_dir="tests/nonexistent_baseline_vector",
            chunk_size=123,
            chunk_overlap=7,
            rebuild_vector_store=True,
        )

        result = build_or_load_vector_store(["doc"], config)

        self.assertEqual(result, "vector-store")
        split_docs.assert_called_once_with(
            ["doc"],
            chunk_size=123,
            chunk_overlap=7,
            add_start_index=True,
        )
        self.assertEqual(create_store.call_args.args[0], ["chunk"])

    @patch("rag_chunking.pipelines.baseline_fixed.load_chroma_vector_store")
    def test_build_vector_store_can_reuse_existing_store(self, load_store):
        load_store.return_value = "existing-store"
        config = FixedBaselineConfig(vector_dir="Baseline_vector", rebuild_vector_store=False)

        result = build_or_load_vector_store([], config)

        self.assertEqual(result, "existing-store")


if __name__ == "__main__":
    unittest.main()