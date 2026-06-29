import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chunking.chunking.structure_based import StructureBasedChunker  # noqa: E402
from rag_chunking.pipelines.build_l1_vectorstore import (  # noqa: E402
    L1VectorBuildConfig,
    build_l1_sections,
    build_l1_vector_store,
)


class StructureBasedChunkerTests(unittest.TestCase):
    def test_extract_sections_from_headings_and_paragraphs(self):
        html = """
        <h1>Movie</h1>
        <p>Intro text [1].</p>
        <h2>Cast</h2>
        <p>Alice starred.</p>
        <h3>Reception</h3>
        <p>Reviews were positive.</p>
        """
        sections = StructureBasedChunker().extract_sections(html)

        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]["section"]["title"], "Movie")
        self.assertEqual(sections[0]["section"]["content"], "Intro text .")
        self.assertEqual(sections[1]["section"]["title"], "Movie > Cast")
        self.assertEqual(sections[2]["section"]["title"], "Movie > Cast > Reception")

    def test_extract_sections_falls_back_to_full_document(self):
        sections = StructureBasedChunker().extract_sections("plain text only")

        self.assertEqual(sections[0]["section"], {"title": "Full Document", "content": "plain text only"})

    def test_chunk_documents_assigns_stable_document_ids(self):
        records = [{"document_text": "<h1>Title</h1><p>Content.</p>"}]

        sections = StructureBasedChunker().chunk_documents(records)

        self.assertEqual(sections[0]["document_id"], "Doc_0_Sec_1")

    @patch("rag_chunking.pipelines.build_l1_vectorstore.load_json")
    def test_build_l1_sections_loads_dataset_records(self, load_json):
        load_json.return_value = [{"document_text": "<h1>Title</h1><p>Content.</p>"}]

        sections = build_l1_sections(L1VectorBuildConfig(max_documents=1))

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["section"]["title"], "Title")

    @patch("rag_chunking.pipelines.build_l1_vectorstore.create_chroma_vector_store")
    @patch("rag_chunking.chunking.structure_based.StructureBasedChunker.to_langchain_documents")
    @patch("rag_chunking.pipelines.build_l1_vectorstore.load_json")
    def test_build_l1_vector_store_wires_sections_to_chroma(self, load_json, to_docs, create_store):
        load_json.return_value = [{"document_text": "<h1>Title</h1><p>Content.</p>"}]
        to_docs.return_value = ["doc"]
        create_store.return_value = "vector-store"
        config = L1VectorBuildConfig(vector_dir="tests/nonexistent_l1_vector", overwrite=False)

        vector_store, sections = build_l1_vector_store(config)

        self.assertEqual(vector_store, "vector-store")
        self.assertEqual(len(sections), 1)
        create_store.assert_called_once()
        self.assertEqual(create_store.call_args.args[0], ["doc"])


if __name__ == "__main__":
    unittest.main()