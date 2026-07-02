import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.chunking.sentence_window import source_nodes_to_contexts  # noqa: E402
from rag_chunking.pipelines.sentence_window import (  # noqa: E402
    SentenceWindowConfig,
    collect_questions,
    get_l1_relevant_sections,
    run_sentence_window_pipeline,
)


class FakeDocument:
    def __init__(self, text="", metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class FakeRetriever:
    def __init__(self):
        self.search_kwargs = {}

    def invoke(self, query):
        return [FakeDocument(f"section {i}") for i in range(5)]


class FakeVectorStore:
    def __init__(self):
        self.retriever = FakeRetriever()

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.retriever.search_kwargs = dict(search_kwargs)
        return self.retriever


class FakeSourceNode:
    def __init__(self, text):
        self.text = text

    def get_content(self):
        return self.text


class FakeResponse:
    response = "answer"
    source_nodes = [FakeSourceNode("ctx1"), FakeSourceNode("ctx2")]


class FakeQueryEngine:
    def query(self, query):
        return FakeResponse()


class SentenceWindowTests(unittest.TestCase):
    def test_source_nodes_to_contexts(self):
        self.assertEqual(source_nodes_to_contexts([FakeSourceNode("a")]), ["a"])

    def test_collect_questions_reads_metadata(self):
        documents = [FakeDocument(metadata={"question_text": "q1"}), FakeDocument(metadata={})]

        self.assertEqual(collect_questions(documents), ["q1"])

    def test_l1_retrieval_fetches_then_slices_top_sections(self):
        vector_store = FakeVectorStore()
        config = SentenceWindowConfig(l1_fetch_k=10, l1_top_k=3)

        sections = get_l1_relevant_sections(vector_store, "query", config)

        self.assertEqual(len(sections), 3)
        self.assertEqual(vector_store.search_type, "similarity")
        self.assertEqual(vector_store.retriever.search_kwargs["k"], 10)

    @patch("builtins.print")
    @patch("rag_chunking.pipelines.sentence_window.save_run_results")
    @patch("rag_chunking.pipelines.sentence_window.create_customized_query_engine")
    @patch("rag_chunking.pipelines.sentence_window.create_node_index")
    @patch("rag_chunking.pipelines.sentence_window.load_chroma_vector_store")
    @patch("rag_chunking.pipelines.sentence_window.load_gold_answer_documents")
    def test_run_sentence_window_pipeline_wires_two_stage_flow(
        self,
        load_documents,
        load_vector_store,
        create_node_index,
        create_engine,
        save_results,
        _print,
    ):
        load_documents.return_value = [
            FakeDocument(metadata={"question_text": "q", "gold_answer": {"long_answer": "gold"}})
        ]
        load_vector_store.return_value = FakeVectorStore()
        create_node_index.return_value = "node-index"
        create_engine.return_value = FakeQueryEngine()
        save_results.return_value = Path("out.json")
        config = SentenceWindowConfig(max_questions=1, output_path="evaluation/out.json")

        results = run_sentence_window_pipeline(config)

        self.assertEqual(results[0]["input_question"], "q")
        self.assertEqual(results[0]["retrieved_contexts"], ["ctx1", "ctx2"])
        self.assertEqual(results[0]["response"], "answer")
        self.assertEqual(results[0]["gold_answer"], {"long_answer": "gold"})
        create_node_index.assert_called_once()
        create_engine.assert_called_once()
        save_results.assert_called_once()


if __name__ == "__main__":
    unittest.main()