"""
Cost Model — Azure RAG stack pricing (USD, 2025).

Pricing references
──────────────────
  text-embedding-3-small : $0.02  / 1M tokens
  GPT-4o mini input      : $0.15  / 1M tokens
  GPT-4o mini output     : $0.60  / 1M tokens
  Azure AI Search S1     : $250   / month / search-unit
  Semantic ranker        : $1.00  / 1,000 queries (above free 1K/month tier)
"""
from __future__ import annotations

from models import CostEstimate


class CostModel:

    EMBED_PER_TOK     = 0.02  / 1_000_000
    GPT4O_IN_PER_TOK  = 0.15  / 1_000_000
    GPT4O_OUT_PER_TOK = 0.60  / 1_000_000
    SEARCH_PER_QUERY  = 0.000_028
    SEMANTIC_PER_Q    = 0.001

    # ── Monthly breakdown (baseline: 1,000 queries / day) ─────────────────────

    MONTHLY_TABLE: list[tuple[str, str, str]] = [
        ("Azure AI Search S1 (2 replicas)",          "$500.00", "$16.67"),
        ("Azure OpenAI text-embedding-3-small",      "$6.00",   "$0.20"),
        ("Azure Search semantic ranker",             "$30.00",  "$1.00"),
        ("Azure OpenAI GPT-4o mini – input tokens",  "$18.00",  "$0.60"),
        ("Azure OpenAI GPT-4o mini – output tokens", "$7.20",   "$0.24"),
        ("Azure Blob Storage (500 GB docs)",         "$9.00",   "$0.30"),
        ("Log Analytics (2 GB/day ingestion)",       "$165.60", "$5.52"),
        ("Azure Key Vault (secrets + CMK)",          "$0.30",   "$0.01"),
        ("Private Endpoints × 2",                   "$14.40",  "$0.48"),
        ("Azure Monitor dashboards",                 "$5.00",   "$0.17"),
    ]

    # ── Optimisation strategies ───────────────────────────────────────────────

    OPTIMIZATIONS: list[tuple[str, str, str]] = [
        ("Embedding cache (Redis)",           "~40%", "MD5(query) → TTL 1h; already implemented"),
        ("text-embedding-3-small vs ADA-002", "~80%", "$0.02 vs $0.10 per 1M tokens"),
        ("Semantic ranker gating",            "~35%", "Skip when RRF top-1 score > 0.5"),
        ("Azure OpenAI Batch API",            "~50%", "Off-peak indexing at 50% discount"),
        ("Tiered index storage",              "~25%", "Hot/cold partitioning on doc age"),
        ("PTU (provisioned throughput)",      "~30%", "Commit to PTU for predictable volume"),
        ("Vector quantization on HNSW",       "~60%", "int8 quantization reduces index size"),
    ]

    # ── Per-query estimator ───────────────────────────────────────────────────

    @classmethod
    def estimate(
        cls,
        in_tok:   int,
        out_tok:  int,
        semantic: bool = True,
    ) -> CostEstimate:
        """Return a per-query CostEstimate for the given token counts."""
        return CostEstimate(
            embed_usd      = 20 * cls.EMBED_PER_TOK,
            search_usd     = cls.SEARCH_PER_QUERY,
            llm_input_usd  = in_tok  * cls.GPT4O_IN_PER_TOK,
            llm_output_usd = out_tok * cls.GPT4O_OUT_PER_TOK,
            semantic_usd   = cls.SEMANTIC_PER_Q if semantic else 0.0,
        )
