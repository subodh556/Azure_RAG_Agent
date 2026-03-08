"""
EvalScores dataclass — holds all four LLM-judge scores and their reasoning.

Score scale : 1 (very poor) → 5 (excellent)
Evaluators  : Relevance · Groundedness · Coherence · Fluency
              (Azure AI Evaluation SDK, judge = GPT-4o mini)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalScores:
    relevance:           float = 0.0
    groundedness:        float = 0.0
    coherence:           float = 0.0
    fluency:             float = 0.0
    relevance_reason:    str   = ""
    groundedness_reason: str   = ""
    coherence_reason:    str   = ""
    fluency_reason:      str   = ""

    @property
    def overall(self) -> float:
        """Mean of all four scores, rounded to 2 dp."""
        return round(
            (self.relevance + self.groundedness + self.coherence + self.fluency) / 4,
            2,
        )

    @property
    def verdict(self) -> str:
        s = self.overall
        if s >= 4.0:
            return "PASS"
        if s >= 2.5:
            return "PARTIAL"
        return "FAIL"

    @property
    def verdict_color(self) -> str:
        return {
            "PASS":    "#22c55e",
            "PARTIAL": "#fbbf24",
            "FAIL":    "#ef4444",
        }[self.verdict]
