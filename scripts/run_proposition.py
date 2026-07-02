"""Run the two-stage structure + proposition RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.pipelines.proposition import (  # noqa: E402
    PropositionPipelineConfig,
    run_proposition_pipeline,
)


DEFAULT_CONFIG = "configs/proposition.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposition RAG pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--l1-vector-dir", default=None)
    parser.add_argument("--l2-vector-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--l1-top-k", type=int, default=None)
    parser.add_argument("--l1-fetch-k", type=int, default=None)
    parser.add_argument("--result-top-k", type=int, default=None)
    parser.add_argument("--l2-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--mode", choices=["prod", "test"], default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataclass_config(
        PropositionPipelineConfig,
        args.config,
        dataset_file=args.dataset_file,
        l1_vector_dir=args.l1_vector_dir,
        l2_vector_dir=args.l2_vector_dir,
        output_path=args.output_path,
        l1_top_k=args.l1_top_k,
        l1_fetch_k=args.l1_fetch_k,
        result_top_k=args.result_top_k,
        l2_model=args.l2_model,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
        max_questions=args.max_questions,
    )
    run_results = run_proposition_pipeline(config)
    print(f"Saved {len(run_results)} proposition results")


if __name__ == "__main__":
    main()
