"""
Cross-Document Comparison capability.

Pipeline
────────
1. Fetch all chunks for Document A (ordered by chunk_index).
2. Fetch all chunks for Document B (ordered by chunk_index).
3. Build the comparison prompt via PromptBuilder.compare_system() which
   embeds both full texts and prescribes a structured output format.
4. Call GenerationService.generate_once() — stateless, no history.
5. Return the structured markdown comparison.

The comparison prompt enforces six sections:
  Overview / Common Themes / Key Differences /
  Unique to A / Unique to B / Synthesis & Recommendation
"""
from __future__ import annotations

from azure.search.documents import SearchClient

from models import Document
from generation.prompts import PromptBuilder
from generation.service  import GenerationService


class Comparator:
    """
    Compares two documents that are already indexed in Azure AI Search.

    Parameters
    ----------
    search_client : SearchClient  (from AzureSearchRetriever.client)
    gen_svc       : GenerationService
    """

    def __init__(
        self,
        search_client: SearchClient,
        gen_svc:       GenerationService,
    ) -> None:
        self._client  = search_client
        self._gen_svc = gen_svc

    def _fetch_full_text(self, doc_id: str) -> str:
        raw = self._client.search(
            search_text = "*",
            filter      = f"doc_id eq '{doc_id}'",
            select      = ["text", "chunk_index"],
            top         = 1000,
            order_by    = ["chunk_index asc"],
        )
        return "\n\n".join(r["text"] for r in raw)

    def run(self, doc_a: Document, doc_b: Document) -> str:
        """
        Compare two documents and return a structured markdown analysis.

        Raises
        ------
        ValueError  – if doc_a and doc_b have the same doc_id
        """
        if doc_a.doc_id == doc_b.doc_id:
            raise ValueError("Select two different documents to compare.")

        text_a = self._fetch_full_text(doc_a.doc_id)
        text_b = self._fetch_full_text(doc_b.doc_id)

        system_prompt = PromptBuilder.compare_system(
            doc_a = doc_a.name,
            doc_b = doc_b.name,
            text_a = text_a,
            text_b = text_b,
        )

        answer, _, _ = self._gen_svc.generate_once(
            user_msg   = (
                f'Compare "{doc_a.name}" and "{doc_b.name}" in detail '
                f"using the documents provided in the system prompt."
            ),
            system     = system_prompt,
            max_tokens = 1500,
        )
        return answer
