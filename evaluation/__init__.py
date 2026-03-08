"""evaluation — Azure AI Evaluation SDK pipeline (Relevance · Groundedness · Coherence · Fluency)."""
from .scores    import EvalScores
from .evaluator import AzureEvaluationService

__all__ = ["EvalScores", "AzureEvaluationService"]
