"""
Azure AI Evaluation Service.

Wraps the four Azure AI Evaluation SDK LLM-judge evaluators.

Evaluators
──────────
RelevanceEvaluator    — Does the response address the query?
                        Inputs: query, response, context
GroundednessEvaluator — Is every claim supported by retrieved context?
                        Inputs: query, response, context
CoherenceEvaluator    — Is the response logically structured?
                        Inputs: query, response
FluencyEvaluator      — Is the response grammatically correct / natural?
                        Inputs: query, response

All four use AzureOpenAIModelConfiguration to call GPT-4o mini as the
judge model (same deployment used for RAG generation).
Each returns a 1–5 integer score and an optional reason string.

Install: pip install azure-ai-evaluation==0.3.3
"""
from __future__ import annotations

from config import AzureConfig
from evaluation.scores import EvalScores

try:
    from azure.ai.evaluation import (
        AzureOpenAIModelConfiguration,
        RelevanceEvaluator,
        GroundednessEvaluator,
        CoherenceEvaluator,
        FluencyEvaluator,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class AzureEvaluationService:
    """
    Runs all four evaluators against a (query, response, context) triple.

    Usage
    -----
    svc    = AzureEvaluationService(cfg)
    scores = svc.run(query="...", response="...", context="...")
    """

    def __init__(self, cfg: AzureConfig) -> None:
        if not _AVAILABLE:
            raise RuntimeError(
                "azure-ai-evaluation is not installed.\n"
                "Run: pip install azure-ai-evaluation==0.3.3"
            )
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint   = cfg.openai_endpoint,
            api_key          = cfg.openai_api_key,
            azure_deployment = cfg.chat_deployment,
            api_version      = "2024-06-01",
        )
        self._relevance    = RelevanceEvaluator(model_config)
        self._groundedness = GroundednessEvaluator(model_config)
        self._coherence    = CoherenceEvaluator(model_config)
        self._fluency      = FluencyEvaluator(model_config)

    def run(self, query: str, response: str, context: str) -> EvalScores:
        """
        Run all four evaluators.  Individual failures are caught so the
        others still complete.

        Parameters
        ----------
        query    : The original user question
        response : The GPT-4o mini generated RAG answer
        context  : Retrieved chunks concatenated as plain text
        """

        def _safe(evaluator, **kwargs) -> dict:
            try:
                return evaluator(**kwargs) or {}
            except Exception:
                return {}

        rel = _safe(self._relevance,
                    query=query, response=response, context=context)
        gnd = _safe(self._groundedness,
                    query=query, response=response, context=context)
        coh = _safe(self._coherence,
                    query=query, response=response)
        flu = _safe(self._fluency,
                    query=query, response=response)

        return EvalScores(
            relevance           = float(rel.get("relevance",    0) or 0),
            groundedness        = float(gnd.get("groundedness", 0) or 0),
            coherence           = float(coh.get("coherence",    0) or 0),
            fluency             = float(flu.get("fluency",      0) or 0),
            relevance_reason    = rel.get("relevance_reason",    ""),
            groundedness_reason = gnd.get("groundedness_reason", ""),
            coherence_reason    = coh.get("coherence_reason",    ""),
            fluency_reason      = flu.get("fluency_reason",      ""),
        )
