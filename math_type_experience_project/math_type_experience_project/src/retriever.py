from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schema import KBEntry, RetrievedItem


class HybridRetriever:
    def __init__(
        self,
        alpha_type: float = 0.45,
        alpha_question: float = 0.35,
        alpha_experience: float = 0.10,
        alpha_quality: float = 0.10,
    ) -> None:
        self.alpha_type = alpha_type
        self.alpha_question = alpha_question
        self.alpha_experience = alpha_experience
        self.alpha_quality = alpha_quality
        self.entries: List[KBEntry] = []
        self.type_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.question_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.experience_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.type_matrix = None
        self.question_matrix = None
        self.experience_matrix = None

    def fit(self, entries: List[KBEntry]) -> None:
        self.entries = entries
        type_texts = [
            f"{e.abstract_info.coarse_type} {e.abstract_info.fine_type} {' '.join(e.abstract_info.skills)} {e.abstract_info.template}"
            for e in entries
        ]
        question_texts = [e.question for e in entries]
        experience_texts = [
            " ".join(
                e.experience_info.strategy_steps
                + e.experience_info.key_principles
                + e.experience_info.formulas
                + e.experience_info.pitfalls
                + [e.experience_info.summary]
            )
            for e in entries
        ]
        self.type_matrix = self.type_vectorizer.fit_transform(type_texts)
        self.question_matrix = self.question_vectorizer.fit_transform(question_texts)
        self.experience_matrix = self.experience_vectorizer.fit_transform(experience_texts)

    def save(self, path: str | Path) -> None:
        state = {
            "entries": self.entries,
            "alpha_type": self.alpha_type,
            "alpha_question": self.alpha_question,
            "alpha_experience": self.alpha_experience,
            "alpha_quality": self.alpha_quality,
            "type_vectorizer": self.type_vectorizer,
            "question_vectorizer": self.question_vectorizer,
            "experience_vectorizer": self.experience_vectorizer,
            "type_matrix": self.type_matrix,
            "question_matrix": self.question_matrix,
            "experience_matrix": self.experience_matrix,
        }
        with Path(path).open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetriever":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        obj = cls(
            alpha_type=state["alpha_type"],
            alpha_question=state["alpha_question"],
            alpha_experience=state["alpha_experience"],
            alpha_quality=state["alpha_quality"],
        )
        obj.entries = state["entries"]
        obj.type_vectorizer = state["type_vectorizer"]
        obj.question_vectorizer = state["question_vectorizer"]
        obj.experience_vectorizer = state["experience_vectorizer"]
        obj.type_matrix = state["type_matrix"]
        obj.question_matrix = state["question_matrix"]
        obj.experience_matrix = state["experience_matrix"]
        return obj

    def retrieve(
        self,
        query_type_text: str,
        query_question_text: str,
        query_experience_text: str,
        top_k: int = 4,
        use_quality: bool = True,
        refine: bool = True,
    ) -> List[RetrievedItem]:
        if not self.entries:
            return []
        q_type = self.type_vectorizer.transform([query_type_text])
        q_question = self.question_vectorizer.transform([query_question_text])
        q_experience = self.experience_vectorizer.transform([query_experience_text])

        type_scores = cosine_similarity(q_type, self.type_matrix)[0]
        question_scores = cosine_similarity(q_question, self.question_matrix)[0]
        experience_scores = cosine_similarity(q_experience, self.experience_matrix)[0]
        quality_scores = np.array(
            [float(entry.validation.get("final_score", 0.0)) for entry in self.entries],
            dtype=float,
        )
        if not use_quality:
            quality_scores = np.zeros_like(quality_scores)

        scores = (
            self.alpha_type * type_scores
            + self.alpha_question * question_scores
            + self.alpha_experience * experience_scores
            + self.alpha_quality * quality_scores
        )

        candidate_ids = list(np.argsort(scores)[::-1][: max(top_k * 3, top_k)])
        candidates: List[RetrievedItem] = []
        for idx in candidate_ids:
            candidates.append(
                RetrievedItem(
                    entry=self.entries[idx],
                    score=float(scores[idx]),
                    type_score=float(type_scores[idx]),
                    question_score=float(question_scores[idx]),
                    experience_score=float(experience_scores[idx]),
                    quality_score=float(quality_scores[idx]),
                )
            )

        if not refine:
            return candidates[:top_k]

        refined: List[RetrievedItem] = []
        seen_fine_types = set()
        for item in candidates:
            fine_type = item.entry.abstract_info.fine_type
            if fine_type in seen_fine_types:
                continue
            refined.append(item)
            seen_fine_types.add(fine_type)
            if len(refined) >= top_k:
                break

        if len(refined) < top_k:
            selected_problem_ids = {item.entry.problem_id for item in refined}
            for item in candidates:
                if item.entry.problem_id in selected_problem_ids:
                    continue
                refined.append(item)
                selected_problem_ids.add(item.entry.problem_id)
                if len(refined) >= top_k:
                    break
        return refined[:top_k]
