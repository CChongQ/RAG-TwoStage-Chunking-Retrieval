"""Analyze evaluated RAG result JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.evaluation.analysis import load_eval_metrics_all_methods, summarize_metrics  # noqa: E402


def parse_method_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method or not path:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    return method, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print mean/std summaries for evaluated RAG result files.")
    parser.add_argument(
        "--method-file",
        action="append",
        type=parse_method_path,
        required=True,
        help="Method/file pair, for example Baseline=evaluation/eval_baseline.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_paths = dict(args.method_file)
    eval_result = load_eval_metrics_all_methods(file_paths)
    print(summarize_metrics(eval_result).to_string())


if __name__ == "__main__":
    main()