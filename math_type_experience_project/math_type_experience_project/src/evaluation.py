from __future__ import annotations

import time
from typing import Dict, Iterable, List

from src.heuristics import normalize_answer
from src.io_utils import write_csv, write_json, write_jsonl
from src.schema import ProblemRecord


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def evaluate_modes(
    records: Iterable[ProblemRecord],
    pipeline,
    modes: List[str],
    output_dir,
    max_eval_samples: int | None = None,
) -> Dict[str, Dict[str, float]]:
    records = list(records)
    if max_eval_samples is not None:
        records = records[:max_eval_samples]

    all_summaries: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        predictions = []
        correct = 0
        latencies = []
        for record in records:
            start = time.perf_counter()
            result = pipeline.predict(record.question, mode=mode)
            latency = time.perf_counter() - start
            latencies.append(latency)
            is_correct = exact_match(str(result["answer"]), record.answer)
            correct += int(is_correct)
            predictions.append(
                {
                    "id": record.id,
                    "dataset": record.dataset,
                    "subject": record.subject,
                    "difficulty": record.difficulty,
                    "question": record.question,
                    "gold_answer": record.answer,
                    "pred_answer": result["answer"],
                    "correct": is_correct,
                    "mode": mode,
                    "latency_sec": round(latency, 6),
                    "query_fine_type": result["query_info"]["fine_type"],
                    "num_retrieved": len(result["retrieved_items"]),
                }
            )
        accuracy = correct / max(len(records), 1)
        summary = {
            "num_samples": len(records),
            "num_correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_latency_sec": round(sum(latencies) / max(len(latencies), 1), 6),
        }
        all_summaries[mode] = summary
        write_jsonl(output_dir / f"predictions_{mode}.jsonl", predictions)
        write_csv(output_dir / f"predictions_{mode}.csv", predictions)

    write_json(output_dir / "summary.json", all_summaries)
    write_csv(
        output_dir / "summary.csv",
        [dict(mode=mode, **summary) for mode, summary in all_summaries.items()],
    )
    return all_summaries
