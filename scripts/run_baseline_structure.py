"""Run the structure-based baseline RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.pipelines.baseline_structure import (  # noqa: E402
    StructureBaselineConfig,
    run_structure_baseline,
)


DEFAULT_CONFIG = "configs/baseline_structure.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the structure-based baseline RAG pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--l1-vector-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
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
        StructureBaselineConfig,
        args.config,
        dataset_file=args.dataset_file,
        l1_vector_dir=args.l1_vector_dir,
        output_path=args.output_path,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_questions=args.max_questions,
        verbose=args.verbose,
    )
    run_results = run_structure_baseline(config)
    print(f"Saved {len(run_results)} results to {config.output_path}")


if __name__ == "__main__":
    main()