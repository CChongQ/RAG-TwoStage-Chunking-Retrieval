"""Analyze evaluated RAG result JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_yaml  # noqa: E402
from rag_chunking.evaluation.analysis import load_eval_metrics_all_methods, summarize_metrics  # noqa: E402


DEFAULT_CONFIG = "configs/analysis.yaml"


def parse_method_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method or not path:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    return method, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print mean/std summaries for evaluated RAG result files.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument(
        "--method-file",
        action="append",
        type=parse_method_path,
        help="Method/file pair, for example Baseline=evaluation/eval_baseline.json.",
    )
    return parser.parse_args()


def load_analysis_config(path: str | Path, method_files: list[tuple[str, str]] | None) -> dict[str, Any]:
    config = load_yaml(path)
    unknown_keys = sorted(set(config) - {"method_files", "fact_based_questions", "broad_questions"})
    if unknown_keys:
        raise ValueError(f"Unknown analysis config key(s): {', '.join(unknown_keys)}")

    configured_files = dict(config.get("method_files") or {})
    if method_files:
        configured_files.update(method_files)
    if not configured_files:
        raise ValueError("At least one method file is required in YAML or --method-file.")
    return configured_files


def main() -> None:
    args = parse_args()
    file_paths = load_analysis_config(args.config, args.method_file)
    eval_result = load_eval_metrics_all_methods(file_paths)
    print(summarize_metrics(eval_result).to_string())


if __name__ == "__main__":
    main()
