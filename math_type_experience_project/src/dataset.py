from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.io_utils import read_csv, read_json, read_jsonl
from src.schema import ProblemRecord

QUESTION_KEYS = ["question", "problem", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "final_answer", "output", "target", "label"]
SOLUTION_KEYS = ["solution", "rationale", "explanation", "reasoning", "steps"]
DATASET_KEYS = ["dataset", "source"]
SPLIT_KEYS = ["split", "subset"]
SUBJECT_KEYS = ["subject", "topic", "domain", "category"]
DIFFICULTY_KEYS = ["difficulty", "level"]


def _first_value(raw: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return str(raw[key])
    return default


def _infer_split(name: str) -> str:
    lower = name.lower()
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    if "dev" in lower or "valid" in lower or "val" in lower:
        return "dev"
    return "train"


def _infer_dataset_name(path: Path) -> str:
    base = path.stem.lower()
    base = re.sub(r"_(train|test|dev|valid|val)$", "", base)
    return base


def _normalize_record(raw: Dict[str, Any], path: Path, idx: int) -> ProblemRecord | None:
    question = _first_value(raw, QUESTION_KEYS)
    answer = _first_value(raw, ANSWER_KEYS)
    if not question or not answer:
        return None
    solution = _first_value(raw, SOLUTION_KEYS)
    dataset = _first_value(raw, DATASET_KEYS, _infer_dataset_name(path))
    split = _first_value(raw, SPLIT_KEYS, _infer_split(path.name))
    subject = _first_value(raw, SUBJECT_KEYS, "unknown")
    difficulty = _first_value(raw, DIFFICULTY_KEYS, "unknown")
    record_id = str(raw.get("id", f"{path.stem}_{idx}"))
    metadata = {
        k: v
        for k, v in raw.items()
        if k
        not in set(
            QUESTION_KEYS
            + ANSWER_KEYS
            + SOLUTION_KEYS
            + DATASET_KEYS
            + SPLIT_KEYS
            + SUBJECT_KEYS
            + DIFFICULTY_KEYS
            + ["id"]
        )
    }
    return ProblemRecord(
        id=record_id,
        question=question.strip(),
        answer=answer.strip(),
        solution=solution.strip(),
        dataset=dataset.strip(),
        split=split.strip().lower(),
        subject=subject.strip(),
        difficulty=difficulty.strip(),
        metadata=metadata,
    )


def discover_dataset_files(dataset_root: str | Path) -> List[Path]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".csv"}
    ]
    files = [p for p in files if not p.name.startswith(".") and "README" not in p.name.upper()]
    if not files:
        raise FileNotFoundError(f"No dataset files found under: {root}")

    non_demo = [p for p in files if "demo" not in p.name.lower()]
    return sorted(non_demo if non_demo else files)


def _load_raw_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        return [obj]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_records(dataset_root: str | Path) -> Tuple[List[ProblemRecord], List[ProblemRecord]]:
    train_records: List[ProblemRecord] = []
    eval_records: List[ProblemRecord] = []
    for path in discover_dataset_files(dataset_root):
        rows = _load_raw_rows(path)
        for idx, raw in enumerate(rows):
            record = _normalize_record(raw, path, idx)
            if record is None:
                continue
            if record.split in {"test", "dev", "valid", "val"}:
                eval_records.append(record)
            else:
                train_records.append(record)
    return train_records, eval_records
