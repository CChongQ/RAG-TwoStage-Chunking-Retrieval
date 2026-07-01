"""Run the two-stage structure + sentence-window RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.pipelines.sentence_window import (  # noqa: E402
    SentenceWindowConfig,
    run_sentence_window_pipeline,
)


DEFAULT_CONFIG = "configs/sentence_window.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sentence-window RAG pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--l1-vector-dir", default=None)
    parser.add_argument("--node-index-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--l1-top-k", type=int, default=None)
    parser.add_argument("--l1-fetch-k", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--top-k-2", type=int, default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--prompt-str", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataclass_config(
        SentenceWindowConfig,
        args.config,
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
        prompt_str=args.prompt_str,
        max_questions=args.max_questions,
    )
    run_results = run_sentence_window_pipeline(config)
    print(f"Saved {len(run_results)} results to {config.output_path}")


if __name__ == "__main__":
    main()
