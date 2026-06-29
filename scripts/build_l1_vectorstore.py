"""Build the structure-based L1 Chroma vector store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.pipelines.build_l1_vectorstore import (  # noqa: E402
    L1VectorBuildConfig,
    build_l1_vector_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the structure-based L1 vector store.")
    parser.add_argument("--dataset-file", default="gold_test_file_30.json")
    parser.add_argument("--vector-dir", default="L1_vector_final")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the vector directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = L1VectorBuildConfig(
        dataset_file=args.dataset_file,
        vector_dir=args.vector_dir,
        max_documents=args.max_documents,
        overwrite=args.overwrite,
    )
    _, sections = build_l1_vector_store(config)
    print(f"Saved {len(sections)} structure-based sections to {args.vector_dir}")


if __name__ == "__main__":
    main()