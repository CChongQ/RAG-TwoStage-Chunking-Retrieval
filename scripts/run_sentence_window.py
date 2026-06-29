"""Run the two-stage structure + sentence-window RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.pipelines.sentence_window import (  # noqa: E402
    SentenceWindowConfig,
    run_sentence_window_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sentence-window RAG pipeline.")
    parser.add_argument("--dataset-file", default="gold_test_file_30.json")
    parser.add_argument("--l1-vector-dir", default="L1_vector_final")
    parser.add_argument("--node-index-dir", default="L2_nodes_test")
    parser.add_argument("--output-path", default="evaluation/run_results_sentence_window.json")
    parser.add_argument("--l1-top-k", type=int, default=3)
    parser.add_argument("--l1-fetch-k", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k-2", type=int, default=15)
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SentenceWindowConfig(
        dataset_file=args.dataset_file,
        l1_vector_dir=args.l1_vector_dir,
        node_index_dir=args.node_index_dir,
        output_path=args.output_path,
        l1_top_k=args.l1_top_k,
        l1_fetch_k=args.l1_fetch_k,
        window_size=args.window_size,
        top_n=args.top_n,
        top_k_2=args.top_k_2,
        rerank_model=args.rerank_model,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        embed_model=args.embed_model,
        max_questions=args.max_questions,
    )
    run_results = run_sentence_window_pipeline(config)
    print(f"Saved {len(run_results)} results to {args.output_path}")


if __name__ == "__main__":
    main()