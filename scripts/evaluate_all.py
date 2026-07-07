"""Evaluate multiple RAG run-result JSON files with RAGAS metrics."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_yaml, resolve_project_path  # noqa: E402
from rag_chunking.data_io import load_json, save_json  # noqa: E402
from rag_chunking.evaluation.ragas_metrics import create_ragas_wrappers, evaluate_all  # noqa: E402


DEFAULT_CONFIG = "configs/evaluate_all.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/evaluations"
DEFAULT_EVALUATOR_MODEL = "gpt-3.5-turbo"


def parse_method_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method or not path:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    return method, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple RAG run-result files with RAGAS.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument(
        "--run-file",
        action="append",
        type=parse_method_path,
        help="Method/file pair, for example Baseline=artifacts/run_results/run_results_baseline.json.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser.parse_args()


def _slugify_method(method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", method.strip())
    return slug.strip("_") or "method"


def default_batch_output_path(method: str, input_path: str | Path, output_dir: str | Path) -> Path:
    today = datetime.today().strftime("%Y-%m-%d")
    input_name = Path(input_path).name
    method_slug = _slugify_method(method)
    return resolve_project_path(output_dir) / f"eval_{today}_{method_slug}_{input_name}"


def load_batch_config(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    # Load YAML defaults, then apply CLI overrides.
    
    config = load_yaml(path)
    unknown_keys = sorted(set(config) - {"run_files", "output_dir", "evaluator_model", "embedding_model"})
    if unknown_keys:
        raise ValueError(f"Unknown batch evaluation config key(s): {', '.join(unknown_keys)}")

    run_files = dict(config.get("run_files") or {})
    if args.run_file:
        run_files.update(args.run_file)
    if not run_files:
        raise ValueError("At least one run file is required in YAML or --run-file.")

    return {
        "run_files": run_files,
        "output_dir": args.output_dir or config.get("output_dir") or DEFAULT_OUTPUT_DIR,
        "evaluator_model": args.evaluator_model or config.get("evaluator_model") or DEFAULT_EVALUATOR_MODEL,
        "embedding_model": args.embedding_model if args.embedding_model is not None else config.get("embedding_model"),
    }


async def evaluate_many_run_files(
    run_files: dict[str, str],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
    embedding_model: str | None = None,
) -> dict[str, Path]:
    
    evaluator_llm, evaluator_embeddings = create_ragas_wrappers(evaluator_model, embedding_model)
    saved_paths: dict[str, Path] = {}

    # Evaluate each strategy output independently.
    for method, input_path_value in run_files.items():
        input_path = resolve_project_path(input_path_value)
        output_path = default_batch_output_path(method, input_path, output_dir)
        print(f"Evaluating {method}: {input_path}")
        
        run_results = load_json(input_path)
        evaluated_results = await evaluate_all(
            run_results,
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )
        
        # Save scored results separately
        save_json(evaluated_results, output_path, indent=4)
        saved_paths[method] = output_path
        print(f"Saved {method} evaluated results to {output_path}")

    return saved_paths


def main() -> None:
    args = parse_args()
    config = load_batch_config(args.config, args)
    asyncio.run(evaluate_many_run_files(**config))


if __name__ == "__main__":
    main()