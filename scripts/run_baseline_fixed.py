"""Run the fixed-size baseline RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.pipelines.baseline_fixed import FixedBaselineConfig, run_fixed_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed-size baseline RAG pipeline.")
    parser.add_argument("--dataset-file", default="gold_test_file_30.json")
    parser.add_argument("--vector-dir", default="Baseline_vector")
    parser.add_argument("--output-path", default="evaluation/run_results_baseline.json")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--generation-model", default="gpt-4o")
    parser.add_argument(
        "--reuse-vector-store",
        action="store_true",
        help="Load an existing Chroma vector store instead of rebuilding it.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FixedBaselineConfig(
        dataset_file=args.dataset_file,
        vector_dir=args.vector_dir,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_model=args.generation_model,
        rebuild_vector_store=not args.reuse_vector_store,
        max_questions=args.max_questions,
    )
    run_results = run_fixed_baseline(config)
    print(f"Saved {len(run_results)} results to {args.output_path}")


if __name__ == "__main__":
    main()