"""
RAG Agent — central orchestrator for all six capabilities.

Composes all modules into a single facade:
  add_document / remove_document / list_documents   → document lifecycle
  rag_query                                         → RAG Chat
  summarise        (→ summary.Summariser)           → Capability 2
  compare          (→ compare.Comparator)           → Capability 3
  evaluate         (→ evaluation.AzureEvaluationService) → Capability 4
  clear_history / get_history                       → conversation state

Both EmbeddingService and GenerationService share one AzureOpenAI client
pointed at the same Azure OpenAI resource but use separate deployments:
  embed_deployment  → text-embedding-3-small
  chat_deployment   → gpt-4o-mini
"""
from __future__ import annotations

from typing import Optional

from openai import AzureOpenAI

from config  import AzureConfig
from models  import ChatMessage, Document, RetrievalResult

from core.index_manager import AzureSearchIndexManager
from core.embeddings    import EmbeddingService
from core.chunker       import DocumentChunker
from core.retriever     import AzureSearchRetriever
from core.pdf_extractor import PDFExtractor          # noqa: F401  (re-exported convenience)

from generation.prompts import PromptBuilder
from generation.service  import GenerationService

from summary.summariser   import Summariser
from compare.comparator   import Comparator
from evaluation.evaluator import AzureEvaluationService
from evaluation.scores    import EvalScores


class RAGAgent:
    """
    Multi-capability RAG orchestrator — 100 % Azure stack.

    Instantiation
    ─────────────
    cfg   = AzureConfig.from_env()
    agent = RAGAgent(cfg)
    agent.bootstrap_index()     # creates / updates Azure AI Search index schema
    """

    def __init__(self, cfg: AzureConfig) -> None:
        self.cfg = cfg

        openai_client = AzureOpenAI(
            azure_endpoint = cfg.openai_endpoint,
            api_key        = cfg.openai_api_key,
            api_version    = "2024-06-01",
        )

        self.embed_svc = EmbeddingService(cfg, openai_client)
        self.index_mgr = AzureSearchIndexManager(cfg)
        self.retriever = AzureSearchRetriever(cfg, self.embed_svc)
        self.gen_svc   = GenerationService(openai_client, cfg.chat_deployment)
        self.chunker   = DocumentChunker(max_words=200, min_words=30)

        # Capability helpers — each owns its domain logic
        self._summariser = Summariser(self.retriever.client, self.gen_svc)
        self._comparator = Comparator(self.retriever.client, self.gen_svc)

        self.docs: dict[str, Document] = {}

    def bootstrap_index(self) -> None:
        """Create or update the Azure AI Search index schema."""
        self.index_mgr.create_or_update_index()

    # ── Document lifecycle ────────────────────────────────────────────────────

    def add_document(self, doc: Document, force: bool = False) -> tuple[int, bool]:
        """
        Chunk → embed → upsert into Azure AI Search.

        Returns (chunk_count, was_skipped).
        Idempotent when force=False: skips if the doc is already indexed.
        Overlap is applied only to embedding inputs — stored chunk.text
        stays clean (no overlap prefix in the index or citation tooltips).
        """
        already_indexed = self.retriever.doc_exists(doc.doc_id) > 0
        if already_indexed and not force:
            self.docs[doc.doc_id] = doc
            return 0, True
        if already_indexed:
            self.retriever.delete_by_doc_id(doc.doc_id)

        self.docs[doc.doc_id] = doc

        chunks       = self.chunker.chunk(doc)
        embed_inputs = self.chunker.embed_texts(chunks, overlap_words=20)
        vecs         = self.embed_svc.embed_batch(embed_inputs)

        for c, v in zip(chunks, vecs):
            c.vector = v

        self.retriever.upload_chunks(chunks)
        return len(chunks), False

    def remove_document(self, doc_id: str) -> int:
        n = self.retriever.delete_by_doc_id(doc_id)
        self.docs.pop(doc_id, None)
        return n

    def list_documents(self) -> list[Document]:
        return list(self.docs.values())

    # ── Capability 1: RAG Chat ────────────────────────────────────────────────

    def rag_query(
        self,
        query:          str,
        top_k:          int = 5,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> ChatMessage:
        """
        Full RAG pipeline for one user query.

        1. Retrieve top_k chunks via hybrid_search()
        2. Build self-contained user message: SOURCES + QUESTION
        3. Generate answer (GPT-4o mini) with sliding-window history
        4. Store clean Q&A in history — no dangling [N] references
        """
        retrieval = self.retriever.hybrid_search(
            query          = query,
            top_k          = top_k,
            filter_doc_ids = filter_doc_ids,
            use_semantic   = self.cfg.use_semantic_rank,
        )

        if not retrieval.chunks:
            return ChatMessage(
                role    = "assistant",
                content = "I could not find an answer in the indexed documents.",
                chunks  = [],
            )

        text, in_t, out_t = self.gen_svc.generate(
            full_user_msg = PromptBuilder.rag_user_message(query, retrieval),
            system        = PromptBuilder.rag_system(),
            question      = query,
        )

        return ChatMessage(
            role          = "assistant",
            content       = text,
            chunks        = retrieval.chunks,
            input_tokens  = in_t,
            output_tokens = out_t,
        )

    # ── Capability 2: Summarise ───────────────────────────────────────────────

    def summarise(self, doc_id: str) -> str:
        if doc_id not in self.docs:
            raise ValueError(f"Document '{doc_id}' not found.")
        return self._summariser.run(self.docs[doc_id])

    # ── Capability 3: Cross-document Compare ──────────────────────────────────

    def compare(self, doc_id_a: str, doc_id_b: str) -> str:
        for did in (doc_id_a, doc_id_b):
            if did not in self.docs:
                raise ValueError(f"Document '{did}' not found.")
        return self._comparator.run(self.docs[doc_id_a], self.docs[doc_id_b])

    # ── Capability 4: Evaluation Pipeline ────────────────────────────────────

    def evaluate(
        self, query: str, top_k: int = 5
    ) -> tuple[RetrievalResult, str, EvalScores]:
        retrieval = self.retriever.hybrid_search(query, top_k=top_k)

        user_msg = PromptBuilder.rag_user_message(query, retrieval)
        answer, _, _ = self.gen_svc.generate_once(
            user_msg, PromptBuilder.rag_system(), max_tokens=1024
        )

        context = "\n\n".join(
            f"[Source {i}: {c.doc_name} p.{c.page_num}]\n{c.text}"
            for i, c in enumerate(retrieval.chunks, 1)
        )

        scores = AzureEvaluationService(self.cfg).run(
            query=query, response=answer, context=context
        )
        return retrieval, answer, scores

    # ── History ───────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self.gen_svc.clear_history()

    def get_history(self) -> list[dict]:
        return list(self.gen_svc.history)
