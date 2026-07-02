"""Run the fixed-size baseline RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.pipelines.baseline_fixed import FixedBaselineConfig, run_fixed_baseline  # noqa: E402


DEFAULT_CONFIG = "configs/baseline_fixed.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed-size baseline RAG pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--vector-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--generation-model", default=None)
    parser.add_argument(
        "--rebuild-vector-store",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to rebuild the Chroma vector store.",
    )
    parser.add_argument(
        "--reuse-vector-store",
        action="store_false",
        dest="rebuild_vector_store",
        help="Load an existing Chroma vector store instead of rebuilding it.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial runs.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable pipeline logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataclass_config(
        FixedBaselineConfig,
        args.config,
        dataset_file=args.dataset_file,
        vector_dir=args.vector_dir,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        rebuild_vector_store=args.rebuild_vector_store,
        max_questions=args.max_questions,
        verbose=args.verbose,
    )
    run_results = run_fixed_baseline(config)
    print(f"Saved {len(run_results)} results to {config.output_path}")


if __name__ == "__main__":
    main()
