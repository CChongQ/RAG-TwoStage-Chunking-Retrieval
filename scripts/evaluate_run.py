"""Evaluate a RAG run-result JSON file with RAGAS metrics."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.evaluation.ragas_metrics import RagasEvaluationConfig, evaluate_run_file  # noqa: E402


DEFAULT_CONFIG = "configs/evaluation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RAG run-result JSON file with RAGAS.")
    parser.add_argument("input_path", nargs="?", default=None, help="Run-result JSON file to evaluate.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataclass_config(
        RagasEvaluationConfig,
        args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
    )
    asyncio.run(evaluate_run_file(config))


if __name__ == "__main__":
    main()
