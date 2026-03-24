from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_records
from src.io_utils import ensure_dir, write_json, write_jsonl
from src.kb_builder import KnowledgeBaseBuilder
from src.llm_backends import build_backend
from src.retriever import HybridRetriever


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config_path = resolve_path(PROJECT_ROOT, args.config)
    cfg = load_config(config_path)
    dataset_root = resolve_path(PROJECT_ROOT, cfg["paths"]["dataset_root"])
    output_root = ensure_dir(resolve_path(PROJECT_ROOT, cfg["paths"]["output_root"]))

    train_records, eval_records = load_records(dataset_root)
    backend = build_backend(cfg["llm"])
    builder = KnowledgeBaseBuilder(
        backend=backend,
        min_validation_score=float(cfg["kb"].get("min_validation_score", 0.55)),
        use_validation=bool(cfg["kb"].get("use_validation", True)),
        use_advanced_generation=bool(cfg["kb"].get("use_advanced_generation", True)),
    )
    kb_entries = builder.build(train_records)
    write_jsonl(output_root / "kb_entries.jsonl", [entry.to_dict() for entry in kb_entries])
    taxonomy = builder.summarize_taxonomy(kb_entries)
    taxonomy.update(
        {
            "num_train_records": len(train_records),
            "num_eval_records": len(eval_records),
            "dataset_root": str(dataset_root),
        }
    )
    write_json(output_root / "type_taxonomy.json", taxonomy)

    retriever = HybridRetriever(
        alpha_type=float(cfg["retrieval"].get("alpha_type", 0.45)),
        alpha_question=float(cfg["retrieval"].get("alpha_question", 0.35)),
        alpha_experience=float(cfg["retrieval"].get("alpha_experience", 0.10)),
        alpha_quality=float(cfg["retrieval"].get("alpha_quality", 0.10)),
    )
    retriever.fit(kb_entries)
    retriever.save(output_root / "retriever.pkl")
    print(f"[build_kb] train_records={len(train_records)} eval_records={len(eval_records)} kb_entries={len(kb_entries)}")
    print(f"[build_kb] outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
