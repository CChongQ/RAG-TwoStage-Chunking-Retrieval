import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.chunking.proposition import PropositionSentences  # noqa: E402
from rag_chunking.pipelines.proposition import (  # noqa: E402
    PropositionPipelineConfig,
    collect_questions,
    default_output_path,
    get_l2_retrieval_result,
    run_proposition_pipeline,
)


class FakeDocument:
    def __init__(self, text="", metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.search_kwargs = {}

    def invoke(self, query):
        return self.docs


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = dict(search_kwargs)
        return FakeRetriever(self.docs)


class FakeRunnable:
    def invoke(self, payload):
        return PropositionSentences(sentences=[payload["document"] + " prop", "extra prop"])


class PropositionPipelineTests(unittest.TestCase):
    def test_collect_questions_reads_metadata(self):
        docs = [FakeDocument(metadata={"question_text": "q1"}), FakeDocument(metadata={})]

        self.assertEqual(collect_questions(docs), ["q1"])

    def test_default_output_path_matches_notebook_pattern(self):
        output = default_output_path("gpt-3.5-turbo")

        self.assertEqual(output.parent.name, "evaluation")
        self.assertIn("run_results_proposition_gpt-3.5-turbo_", output.name)

    @patch("builtins.print")
    def test_get_l2_retrieval_result_returns_empty_without_propositions(self, _print):
        config = PropositionPipelineConfig()
        retriever = FakeRetriever([])

        result = get_l2_retrieval_result("q", retriever, FakeRunnable(), config)

        self.assertEqual(result, [])

    @patch("builtins.print")
    @patch("rag_chunking.pipelines.proposition.save_faiss_vector_store")
    @patch("rag_chunking.pipelines.proposition.get_new_prop_doc")
    def test_get_l2_retrieval_result_embeds_and_retrieves_props(self, get_docs, save_store, _print):
        get_docs.return_value = [FakeDocument("p1"), FakeDocument("p2")]
        save_store.return_value = FakeVectorStore([FakeDocument("selected")])
        config = PropositionPipelineConfig(result_top_k=5)
        l1_docs = [FakeDocument("section one"), FakeDocument("section two")]
        retriever = FakeRetriever(l1_docs)

        result = get_l2_retrieval_result("q", retriever, FakeRunnable(), config, embedding="emb")

        self.assertEqual(result, ["selected"])
        save_store.assert_called_once()
        self.assertEqual(save_store.call_args.kwargs["embedding"], "emb")
        self.assertEqual(save_store.return_value.search_kwargs["k"], 5)

    @patch("builtins.print")
    @patch("rag_chunking.pipelines.proposition.save_run_results")
    @patch("rag_chunking.pipelines.proposition.generate_answer_from_contexts")
    @patch("rag_chunking.pipelines.proposition.get_l2_retrieval_result")
    @patch("rag_chunking.pipelines.proposition.create_proposition_runnable")
    @patch("rag_chunking.pipelines.proposition.load_chroma_vector_store")
    @patch("rag_chunking.pipelines.proposition.create_openai_embedding")
    @patch("rag_chunking.pipelines.proposition.load_gold_answer_documents")
    def test_run_proposition_pipeline_wires_generation_and_saving(
        self,
        load_docs,
        create_embedding,
        load_vector,
        create_runnable,
        get_l2,
        generate_answer,
        save_results,
        _print,
    ):
        load_docs.return_value = [
            FakeDocument(metadata={"question_text": "q", "gold_answer": {"long_answer": "gold"}})
        ]
        create_embedding.return_value = "embedding"
        load_vector.return_value = FakeVectorStore([])
        create_runnable.return_value = FakeRunnable()
        get_l2.return_value = ["ctx"]
        generate_answer.return_value = "answer"
        save_results.return_value = Path("out.json")
        config = PropositionPipelineConfig(output_path="evaluation/out.json", max_questions=1)

        results = run_proposition_pipeline(config)

        self.assertEqual(results[0]["input_question"], "q")
        self.assertEqual(results[0]["retrieved_contexts"], ["ctx"])
        self.assertEqual(results[0]["response"], "answer")
        self.assertEqual(results[0]["gold_answer"], {"long_answer": "gold"})
        generate_answer.assert_called_once()
        save_results.assert_called_once()


if __name__ == "__main__":
    unittest.main()