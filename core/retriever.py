"""
Azure AI Search retriever — hybrid BM25 + HNSW + optional semantic reranker.

Over-fetch strategy
───────────────────
Requests (top_k + 5) results from Azure, deduplicates by chunk_id
client-side, re-sorts by the best available score, then trims to exactly
top_k.  This guarantees the caller always gets the correct number of
unique, best-ranked chunks even when the same chunk surfaces via both
the BM25 and vector pathways.

Security trimming
─────────────────
filter_doc_ids → OData $filter: "doc_id in ('id1','id2')"
In production these doc_ids are derived from the caller's AAD group claims.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery

from config import AzureConfig
from models import Chunk, RetrievalResult
from core.embeddings import EmbeddingService


class AzureSearchRetriever:

    MAX_RETRIES = 3

    def __init__(self, cfg: AzureConfig, embed_svc: EmbeddingService) -> None:
        self.cfg    = cfg
        self.embed  = embed_svc
        self.client = SearchClient(
            endpoint   = cfg.search_endpoint,
            index_name = cfg.index_name,
            credential = AzureKeyCredential(cfg.search_api_key),
        )

    # ── Document management ───────────────────────────────────────────────────

    def doc_exists(self, doc_id: str) -> int:
        """Return chunk count already indexed for doc_id (0 = not present)."""
        try:
            raw = self.client.search(
                search_text = "*",
                filter      = f"doc_id eq '{doc_id}'",
                select      = ["chunk_id"],
                top         = 1,
            )
            return sum(1 for _ in raw)
        except Exception:
            return 0

    def upload_chunks(self, chunks: list[Chunk]) -> None:
        """
        Upsert chunks in batches of 1,000.
        Idempotent: re-indexing the same doc_id updates rather than duplicates
        because chunk_ids are deterministic.
        """
        BATCH = 1000
        docs  = [c.to_search_doc() for c in chunks]
        for i in range(0, len(docs), BATCH):
            results = self.client.merge_or_upload_documents(
                documents=docs[i: i + BATCH]
            )
            failed = [r.key for r in results if not r.succeeded]
            if failed:
                raise RuntimeError(f"Upload failed for chunk_ids: {failed}")

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        raw = self.client.search(
            search_text = "*",
            filter      = f"doc_id eq '{doc_id}'",
            select      = ["chunk_id"],
            top         = 10000,
        )
        to_delete = [{"chunk_id": r["chunk_id"]} for r in raw]
        if to_delete:
            self.client.delete_documents(documents=to_delete)
        return len(to_delete)

    # ── Core search ───────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query:          str,
        top_k:          int = 5,
        filter_doc_ids: Optional[list[str]] = None,
        use_semantic:   bool = True,
    ) -> RetrievalResult:
        """Hybrid search: BM25 + HNSW vector + optional semantic reranker."""
        t0 = time.monotonic()

        odata_filter: Optional[str] = None
        if filter_doc_ids:
            quoted       = [f"'{d}'" for d in filter_doc_ids]
            odata_filter = "doc_id in (" + ", ".join(quoted) + ")"

        query_vector = self.embed.embed_single(query)
        fetch        = top_k + 5

        vector_query = VectorizedQuery(
            vector              = query_vector,
            k_nearest_neighbors = fetch * 2,
            fields              = "text_vector",
            exhaustive          = False,
        )

        search_kwargs: dict[str, Any] = {
            "search_text":    query,
            "vector_queries": [vector_query],
            "select":         ["chunk_id", "doc_id", "doc_name",
                               "chunk_index", "page_num", "text"],
            "top":            fetch,
            "filter":         odata_filter,
            "query_type":     (
                QueryType.SEMANTIC
                if (use_semantic and self.cfg.use_semantic_rank)
                else QueryType.FULL
            ),
        }
        if use_semantic and self.cfg.use_semantic_rank:
            search_kwargs["semantic_configuration_name"] = self.cfg.semantic_config
            search_kwargs["query_caption"] = "extractive"
            search_kwargs["query_answer"]  = "extractive"

        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = self.client.search(**search_kwargs)

                seen:   set[str]    = set()
                chunks: list[Chunk] = []
                for r in raw:
                    cid = r["chunk_id"]
                    if cid not in seen:
                        seen.add(cid)
                        chunks.append(Chunk.from_search_result(dict(r)))

                # Re-sort by best available score after client-side dedup
                if use_semantic and self.cfg.use_semantic_rank:
                    chunks.sort(key=lambda c: c.reranker_score, reverse=True)
                else:
                    chunks.sort(key=lambda c: c.score, reverse=True)

                chunks = chunks[:top_k]
                elapsed = int((time.monotonic() - t0) * 1000)

                return RetrievalResult(
                    chunks          = chunks,
                    query           = query,
                    semantic_ranked = (use_semantic and self.cfg.use_semantic_rank),
                    latency_ms      = elapsed,
                )

            except (HttpResponseError, ServiceRequestError) as e:
                last_exc = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Azure AI Search query failed after {self.MAX_RETRIES} "
            f"retries: {last_exc}"
        )

    def text_only_search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """BM25-only fallback when the embedding service is unavailable."""
        raw = self.client.search(
            search_text = query,
            select      = ["chunk_id", "doc_id", "doc_name",
                           "chunk_index", "page_num", "text"],
            top         = top_k,
        )
        return [Chunk.from_search_result(dict(r)) for r in raw]
