"""Compatibility wrapper for gold-answer helper functions.

The import-safe implementations live in ``rag_chunking.gold_answers``. This
module keeps old notebook/script imports working during the gradual migration.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chunking.gold_answers import (  # noqa: E402,F401
    add_gold_answers_to_records,
    clean_token,
    extract_gold_answer_for_question,
    get_all_gold_answers,
)


if __name__ == "__main__":
    print(
        "Gold-answer helper functions are import-safe now. "
        "Call a specific function from rag_chunking.gold_answers or this wrapper."
    )
