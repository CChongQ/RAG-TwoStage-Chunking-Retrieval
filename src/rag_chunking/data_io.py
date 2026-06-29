"""Input/output helpers for Natural Questions datasets and run results."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rag_chunking.config import DATASET_DIR


NQ_DATASET_PATH = DATASET_DIR / "simplified-nq-train.jsonl"


def load_json(path: str | Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Save data as UTF-8 JSON and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)
    return output_path


def dataset_path(filename: str | Path) -> Path:
    """Resolve a filename relative to the repository dataset directory."""
    path = Path(filename)
    if path.is_absolute():
        return path
    return DATASET_DIR / path


def get_first_n_elements(
    input_file: str | Path,
    output_file_name: str | Path,
    n: int = 10,
) -> Path:
    """Extract the first ``n`` JSONL records and save them as a JSON list."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = dataset_path(output_file_name)
    with input_path.open("r", encoding="utf-8") as file:
        data = [json.loads(line) for _, line in zip(range(n), file)]

    return save_json(data, output_path, indent=4)


def count_each_wiki_occurrence(
    input_file: str | Path,
    output_file: str | Path | None = None,
) -> Counter[str]:
    """Count how many questions are associated with each document URL."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    data = load_json(input_path)
    url_counter: Counter[str] = Counter()
    for entry in data:
        url_counter[entry["document_url"]] += 1

    if output_file is not None:
        save_json(dict(url_counter), output_file, indent=4)

    return url_counter


def count_each_wiki_occurence(
    input_file: str | Path,
    output_file: str | Path | None = None,
) -> Counter[str]:
    """Backward-compatible alias for the original misspelled helper name."""
    return count_each_wiki_occurrence(input_file, output_file)


def format_test_dataset(
    url_count_filename: str | Path,
    input_file: str | Path,
    output_file: str | Path,
    top_n_urls: int = 10,
) -> Path:
    """Group question records by the most frequent document URLs."""
    url_counts = load_json(url_count_filename)
    top_urls = set(sorted(url_counts, key=url_counts.get, reverse=True)[:top_n_urls])
    doc_data = load_json(input_file)

    organized_data: dict[str, dict[str, Any]] = {}
    for question in doc_data:
        doc_url = question["document_url"]
        if doc_url not in top_urls:
            continue

        if doc_url not in organized_data:
            organized_data[doc_url] = {
                "document_url": doc_url,
                "document_text": question["document_text"],
                "questions": [],
            }

        organized_data[doc_url]["questions"].append(
            {
                "question_text": question["question_text"],
                "long_answer_candidates": question.get("long_answer_candidates", []),
                "annotations": question.get("annotations", []),
                "example_id": question["example_id"],
            }
        )

    return save_json(list(organized_data.values()), dataset_path(output_file), indent=4)


def has_valid_annotation(entry: dict[str, Any]) -> bool:
    """Return True when a Natural Questions entry has a valid gold answer span."""
    for annotation in entry.get("annotations", []):
        long_answer = annotation.get("long_answer", {})
        short_answers = annotation.get("short_answers", [])

        valid_long = (
            long_answer.get("start_token", -1) != -1
            and long_answer.get("end_token", -1) != -1
        )
        valid_short = any(
            "start_token" in answer and "end_token" in answer
            for answer in short_answers
        )

        if valid_long or valid_short:
            return True
    return False


def get_test_dataset(
    filename: str | Path,
    output_filename: str | Path,
    dataset_size: int,
    *,
    seed: int | None = None,
) -> Path:
    """Sample valid entries with unique document URLs."""
    data = load_json(filename)
    valid_samples = [entry for entry in data if has_valid_annotation(entry)]

    unique_doc_map: dict[str, dict[str, Any]] = {}
    for sample in valid_samples:
        doc_url = sample.get("document_url")
        if doc_url not in unique_doc_map:
            unique_doc_map[doc_url] = sample

    unique_samples = list(unique_doc_map.values())
    if len(unique_samples) < dataset_size:
        raise ValueError("Not enough unique document_url entries with valid annotations.")

    rng = random.Random(seed)
    sampled = rng.sample(unique_samples, dataset_size)
    return save_json(sampled, output_filename)


def get_n_sample_from_file(
    input_filename: str | Path,
    dataset_size: int,
    output_filename: str | Path,
    max_gold_length: int = 2000,
    *,
    seed: int | None = None,
) -> Path:
    """Sample records whose long gold answer length is below a threshold."""
    data = load_json(input_filename)
    filtered_data = [
        item
        for item in data
        if len(item.get("gold_answer", {}).get("long_answer", "")) <= max_gold_length
    ]

    if len(filtered_data) < dataset_size:
        raise ValueError(
            f"Only {len(filtered_data)} items in the file, but {dataset_size} were requested."
        )

    rng = random.Random(seed)
    sampled_data = rng.sample(filtered_data, dataset_size)
    return save_json(sampled_data, output_filename)


def get_N_sample_from_file(
    in_filename: str | Path,
    dataset_size: int,
    output_filename: str | Path,
    max_gold_length: int = 2000,
) -> Path:
    """Backward-compatible alias for the original helper name."""
    return get_n_sample_from_file(
        in_filename,
        dataset_size,
        output_filename,
        max_gold_length=max_gold_length,
    )
