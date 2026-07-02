"""Build the structure-based L1 Chroma vector store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.config import load_dataclass_config  # noqa: E402
from rag_chunking.pipelines.build_l1_vectorstore import (  # noqa: E402
    L1VectorBuildConfig,
    build_l1_vector_store,
)


DEFAULT_CONFIG = "configs/build_l1_vectorstore.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the structure-based L1 vector store.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file to load.")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--vector-dir", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to replace the vector directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataclass_config(
        L1VectorBuildConfig,
        args.config,
        dataset_file=args.dataset_file,
        vector_dir=args.vector_dir,
        embedding_model=args.embedding_model,
        max_documents=args.max_documents,
        overwrite=args.overwrite,
    )
    _, sections = build_l1_vector_store(config)
    print(f"Saved {len(sections)} structure-based sections to {config.vector_dir}")


if __name__ == "__main__":
    main()
