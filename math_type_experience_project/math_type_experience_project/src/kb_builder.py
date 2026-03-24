from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, List

from src.heuristics import abstract_problem, normalize_answer
from src.schema import AbstractInfo, ExperienceInfo, KBEntry, ProblemRecord

EXPECTED_KEYWORDS = {
    "linear_equation": ["方程", "x", "等式"],
    "rectangle_perimeter": ["周长", "长", "宽"],
    "rectangle_area": ["面积", "长", "宽"],
    "triangle_area": ["面积", "底", "高"],
    "remainder": ["余数", "除数"],
    "combinations": ["组合", "选取"],
    "percentage": ["百分", "比例"],
    "average": ["平均", "总和"],
    "ratio_scale": ["比例", "份"],
    "gcd": ["公因数"],
    "lcm": ["公倍数"],
    "consecutive_integers": ["连续整数", "方程"],
    "arithmetic_total_cost": ["单价", "总价"],
}


class KnowledgeBaseBuilder:
    def __init__(
        self,
        backend: Any,
        min_validation_score: float = 0.55,
        use_validation: bool = True,
        use_advanced_generation: bool = True,
    ) -> None:
        self.backend = backend
        self.min_validation_score = min_validation_score
        self.use_validation = use_validation
        self.use_advanced_generation = use_advanced_generation

    def _validate(
        self,
        record: ProblemRecord,
        abstract_info: AbstractInfo,
        experience_info: ExperienceInfo,
    ) -> Dict[str, Any]:
        answer_valid = 1.0 if normalize_answer(record.answer) else 0.0

        inferred_from_summary = abstract_problem(
            experience_info.summary + " " + " ".join(experience_info.strategy_steps)
        )
        type_consistency = 1.0 if inferred_from_summary.coarse_type == abstract_info.coarse_type else 0.5

        step_count = len(experience_info.strategy_steps)
        if 2 <= step_count <= 6:
            step_quality = 1.0
        elif 1 <= step_count <= 8:
            step_quality = 0.7
        else:
            step_quality = 0.3

        expected = EXPECTED_KEYWORDS.get(abstract_info.fine_type, [])
        combined_text = " ".join(
            experience_info.strategy_steps
            + experience_info.key_principles
            + experience_info.formulas
            + experience_info.pitfalls
            + [experience_info.summary]
        )
        if expected:
            hits = sum(1 for token in expected if token in combined_text)
            authority_match = hits / len(expected)
        else:
            authority_match = 0.8

        final_score = round(mean([answer_valid, type_consistency, step_quality, authority_match]), 4)
        return {
            "answer_valid": answer_valid,
            "type_consistency": type_consistency,
            "step_quality": step_quality,
            "authority_match": authority_match,
            "final_score": final_score,
        }

    def _deduplicate(self, entries: List[KBEntry]) -> List[KBEntry]:
        best_by_key: Dict[str, KBEntry] = {}
        for entry in entries:
            key = f"{entry.abstract_info.fine_type}::{entry.question.strip().lower()}"
            if key not in best_by_key:
                best_by_key[key] = entry
                continue
            old_score = float(best_by_key[key].validation.get("final_score", 0.0))
            new_score = float(entry.validation.get("final_score", 0.0))
            if new_score > old_score:
                best_by_key[key] = entry
        return list(best_by_key.values())

    def build(self, train_records: List[ProblemRecord]) -> List[KBEntry]:
        entries: List[KBEntry] = []
        for record in train_records:
            abstract_info = self.backend.abstract_problem(record.question)
            if not self.use_advanced_generation:
                abstract_info.skills = abstract_info.skills[:1]
                abstract_info.template = abstract_info.fine_type

            experience_info = self.backend.generate_experience(record, abstract_info)
            if not self.use_advanced_generation:
                experience_info.pitfalls = []
                experience_info.formulas = experience_info.formulas[:1]
                experience_info.summary = experience_info.summary.split("。", 1)[0]

            validation = self._validate(record, abstract_info, experience_info)
            if self.use_validation and validation["final_score"] < self.min_validation_score:
                continue

            entries.append(
                KBEntry(
                    problem_id=record.id,
                    dataset=record.dataset,
                    split=record.split,
                    question=record.question,
                    answer=record.answer,
                    solution=record.solution,
                    subject=record.subject,
                    difficulty=record.difficulty,
                    abstract_info=abstract_info,
                    experience_info=experience_info,
                    validation=validation,
                )
            )
        return self._deduplicate(entries)

    @staticmethod
    def summarize_taxonomy(entries: List[KBEntry]) -> Dict[str, Any]:
        coarse = Counter(entry.abstract_info.coarse_type for entry in entries)
        fine = Counter(entry.abstract_info.fine_type for entry in entries)
        return {
            "num_entries": len(entries),
            "coarse_type_counts": dict(coarse),
            "fine_type_counts": dict(fine),
        }
