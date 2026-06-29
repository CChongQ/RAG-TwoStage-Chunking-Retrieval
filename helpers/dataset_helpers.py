"""Compatibility wrapper for dataset helper functions.

The import-safe implementations live in ``rag_chunking.data_io``. This module
keeps old notebook/script imports working during the gradual migration.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.data_io import (  # noqa: E402,F401
    NQ_DATASET_PATH as NQ_dataset_path,
    count_each_wiki_occurence,
    count_each_wiki_occurrence,
    dataset_path,
    format_test_dataset,
    get_N_sample_from_file,
    get_first_n_elements,
    get_n_sample_from_file,
    get_test_dataset,
    has_valid_annotation,
    load_json,
    save_json,
)


if __name__ == "__main__":
    print(
        "Dataset helper functions are import-safe now. "
        "Call a specific function from rag_chunking.data_io or this wrapper."
    )
