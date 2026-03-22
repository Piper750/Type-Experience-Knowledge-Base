from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_records
from src.evaluation import evaluate_modes
from src.io_utils import ensure_dir, write_csv, write_json
from src.kb_builder import KnowledgeBaseBuilder
from src.llm_backends import build_backend
from src.pipeline import MathKBPipeline
from src.retriever import HybridRetriever


ABLATIONS = {
    "zero_shot": {"mode": ["zero_shot"]},
    "type_only": {"mode": ["type_only"]},
    "experience_only": {"mode": ["experience_only"]},
    "full": {"mode": ["full"]},
    "no_validation": {"mode": ["full"], "kb": {"use_validation": False}},
    "no_refinement": {"mode": ["full"], "retrieval": {"refine": False}},
    "no_advanced_generation": {"mode": ["full"], "kb": {"use_advanced_generation": False}},
}


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
    base_cfg = load_config(config_path)
    dataset_root = resolve_path(PROJECT_ROOT, base_cfg["paths"]["dataset_root"])
    ablation_root = ensure_dir(resolve_path(PROJECT_ROOT, base_cfg["paths"]["output_root"]) / "ablation")
    train_records, eval_records = load_records(dataset_root)
    results = []

    for ablation_name, patch in ABLATIONS.items():
        cfg = copy.deepcopy(base_cfg)
        for section_name, section_patch in patch.items():
            if section_name == "mode":
                cfg.setdefault("evaluation", {})["modes"] = section_patch
            else:
                cfg.setdefault(section_name, {}).update(section_patch)

        backend = build_backend(cfg["llm"])
        builder = KnowledgeBaseBuilder(
            backend=backend,
            min_validation_score=float(cfg["kb"].get("min_validation_score", 0.55)),
            use_validation=bool(cfg["kb"].get("use_validation", True)),
            use_advanced_generation=bool(cfg["kb"].get("use_advanced_generation", True)),
        )
        kb_entries = builder.build(train_records)
        retriever = HybridRetriever(
            alpha_type=float(cfg["retrieval"].get("alpha_type", 0.45)),
            alpha_question=float(cfg["retrieval"].get("alpha_question", 0.35)),
            alpha_experience=float(cfg["retrieval"].get("alpha_experience", 0.10)),
            alpha_quality=float(cfg["retrieval"].get("alpha_quality", 0.10)),
        )
        retriever.fit(kb_entries)
        run_dir = ensure_dir(ablation_root / ablation_name)
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
            modes=list(cfg["evaluation"].get("modes", ["full"])),
            output_dir=run_dir,
            max_eval_samples=cfg["evaluation"].get("max_eval_samples"),
        )
        first_mode = list(summaries.keys())[0]
        results.append(
            {
                "ablation": ablation_name,
                "mode": first_mode,
                "accuracy": summaries[first_mode]["accuracy"],
                "avg_latency_sec": summaries[first_mode]["avg_latency_sec"],
                "num_samples": summaries[first_mode]["num_samples"],
                "kb_entries": len(kb_entries),
            }
        )
        print(f"[ablation] {ablation_name}: {results[-1]}")

    write_json(ablation_root / "ablation_summary.json", results)
    write_csv(ablation_root / "ablation_summary.csv", results)


if __name__ == "__main__":
    main()
