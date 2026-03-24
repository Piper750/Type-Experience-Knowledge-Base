from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ProblemRecord:
    id: str
    question: str
    answer: str
    solution: str = ""
    dataset: str = "unknown"
    split: str = "train"
    subject: str = "unknown"
    difficulty: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AbstractInfo:
    coarse_type: str
    fine_type: str
    skills: List[str] = field(default_factory=list)
    template: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceInfo:
    strategy_steps: List[str] = field(default_factory=list)
    key_principles: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    pitfalls: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KBEntry:
    problem_id: str
    dataset: str
    split: str
    question: str
    answer: str
    solution: str
    subject: str
    difficulty: str
    abstract_info: AbstractInfo
    experience_info: ExperienceInfo
    validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class RetrievedItem:
    entry: KBEntry
    score: float
    type_score: float
    question_score: float
    experience_score: float
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": self.score,
            "type_score": self.type_score,
            "question_score": self.question_score,
            "experience_score": self.experience_score,
            "quality_score": self.quality_score,
        }
