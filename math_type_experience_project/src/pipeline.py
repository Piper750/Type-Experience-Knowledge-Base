from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from src.heuristics import abstract_problem
from src.retriever import HybridRetriever
from src.schema import RetrievedItem


class MathKBPipeline:
    def __init__(
        self,
        backend,
        retriever: HybridRetriever,
        top_k: int = 4,
        use_quality: bool = True,
        refine: bool = True,
    ) -> None:
        self.backend = backend
        self.retriever = retriever
        self.top_k = top_k
        self.use_quality = use_quality
        self.refine = refine

    def _build_context(self, query_info, retrieved_items: List[RetrievedItem], mode: str) -> str:
        if mode == "zero_shot":
            return ""
        blocks: List[str] = [
            f"[Current Type]\ncoarse_type: {query_info.coarse_type}\nfine_type: {query_info.fine_type}\nskills: {', '.join(query_info.skills)}\ntemplate: {query_info.template}"
        ]
        if mode == "type_only":
            for rank, item in enumerate(retrieved_items, start=1):
                blocks.append(
                    f"[Retrieved Type {rank}]\nfine_type: {item.entry.abstract_info.fine_type}\ncoarse_type: {item.entry.abstract_info.coarse_type}\ntemplate: {item.entry.abstract_info.template}"
                )
            return "\n\n".join(blocks)

        if mode in {"experience_only", "full"}:
            for rank, item in enumerate(retrieved_items, start=1):
                exp = item.entry.experience_info
                pieces = [
                    f"[Retrieved Experience {rank}]\nscore: {item.score:.4f}",
                    f"type: {item.entry.abstract_info.fine_type}",
                    f"summary: {exp.summary}",
                    f"steps: {' | '.join(exp.strategy_steps)}",
                    f"pitfalls: {' | '.join(exp.pitfalls)}",
                    f"principles: {' | '.join(exp.key_principles)}",
                    f"formulas: {' | '.join(exp.formulas)}",
                ]
                if mode == "experience_only":
                    blocks.append("\n".join(pieces[0:2] + pieces[2:]))
                else:
                    blocks.append("\n".join(pieces))
        return "\n\n".join(blocks)

    def predict(self, question: str, mode: str = "full") -> Dict[str, object]:
        query_info = self.backend.abstract_problem(question)
        query_type_text = f"{query_info.coarse_type} {query_info.fine_type} {' '.join(query_info.skills)} {query_info.template}"
        query_experience_text = query_info.template + " " + " ".join(query_info.skills)

        retrieved_items: List[RetrievedItem] = []
        if mode != "zero_shot":
            retrieved_items = self.retriever.retrieve(
                query_type_text=query_type_text,
                query_question_text=question,
                query_experience_text=query_experience_text,
                top_k=self.top_k,
                use_quality=self.use_quality,
                refine=self.refine,
            )

        context = self._build_context(query_info, retrieved_items, mode)
        answer = self.backend.solve(question, context=context, mode=mode)
        return {
            "answer": answer,
            "query_info": asdict(query_info),
            "context": context,
            "retrieved_items": [item.to_dict() for item in retrieved_items],
        }
