"""Project-level path helpers and configuration placeholders."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

