from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_records
from src.evaluation import evaluate_modes
from src.io_utils import ensure_dir
from src.llm_backends import build_backend
from src.pipeline import MathKBPipeline
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

    _, eval_records = load_records(dataset_root)
    backend = build_backend(cfg["llm"])
    retriever = HybridRetriever.load(output_root / "retriever.pkl")
    pipeline = MathKBPipeline(
        backend=backend,
        retriever=retriever,
        top_k=int(cfg["retrieval"].get("top_k", 4)),
        use_quality=bool(cfg["retrieval"].get("use_quality", True)),
        refine=bool(cfg["retrieval"].get("refine", True)),
    )
    summaries = evaluate_modes(
        records=eval_records,
        pipeline=pipeline,
        modes=list(cfg["evaluation"].get("modes", ["zero_shot", "type_only", "experience_only", "full"])),
        output_dir=output_root,
        max_eval_samples=cfg["evaluation"].get("max_eval_samples"),
    )
    print("[evaluate] summary:")
    for mode, summary in summaries.items():
        print(mode, summary)


if __name__ == "__main__":
    main()
