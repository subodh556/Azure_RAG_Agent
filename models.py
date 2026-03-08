"""
Shared data models used across all modules.

Document   → raw ingested content
Chunk      → one indexable unit (stored in Azure AI Search)
RetrievalResult → output of a hybrid search call
ChatMessage     → one assistant turn (answer + chunks)
CostEstimate    → per-query cost breakdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Document:
    doc_id:   str
    name:     str
    content:  str
    source:   str  = ""
    metadata: dict = field(default_factory=dict)
    added_at: str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Chunk:
    """
    One indexable unit.

    to_search_doc()        → serialise for Azure AI Search upload
    from_search_result()   → deserialise from Azure AI Search response
    """
    chunk_id:       str
    doc_id:         str
    doc_name:       str
    chunk_index:    int
    page_num:       int         = 1
    text:           str         = ""
    vector:         list[float] = field(default_factory=list)
    score:          float       = 0.0   # @search.score  (RRF hybrid)
    reranker_score: float       = 0.0   # @search.reranker_score (semantic)

    def to_search_doc(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "doc_name":    self.doc_name,
            "chunk_index": self.chunk_index,
            "page_num":    self.page_num,
            "text":        self.text,
            "text_vector": self.vector,
        }

    @classmethod
    def from_search_result(cls, r: dict) -> "Chunk":
        return cls(
            chunk_id       = r["chunk_id"],
            doc_id         = r["doc_id"],
            doc_name       = r["doc_name"],
            chunk_index    = r["chunk_index"],
            page_num       = r.get("page_num", 1),
            text           = r["text"],
            score          = r.get("@search.score", 0.0),
            reranker_score = r.get("@search.reranker_score", 0.0),
        )


@dataclass
class RetrievalResult:
    chunks:          list[Chunk]
    query:           str
    semantic_ranked: bool
    latency_ms:      int


@dataclass
class ChatMessage:
    role:          str
    content:       str
    chunks:        list[Chunk] = field(default_factory=list)
    ts:            str         = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )
    input_tokens:  int = 0
    output_tokens: int = 0


@dataclass
class CostEstimate:
    embed_usd:      float
    search_usd:     float
    llm_input_usd:  float
    llm_output_usd: float
    semantic_usd:   float

    @property
    def total(self) -> float:
        return (
            self.embed_usd
            + self.search_usd
            + self.llm_input_usd
            + self.llm_output_usd
            + self.semantic_usd
        )
