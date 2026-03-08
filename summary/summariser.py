"""
Document Summarisation capability.

Pipeline
────────
1. Fetch all chunks for the requested doc_id from Azure AI Search
   (ordered by chunk_index so the model reads them in original order).
2. Concatenate chunk texts into a single full-document string.
3. Call GenerationService.generate_once() with PromptBuilder.summary_system().
4. Return the structured markdown summary.

The summary prompt enforces five sections:
  Executive Summary / Key Topics / Main Arguments / Notable Details /
  Suggested Follow-up Questions
"""
from __future__ import annotations

from azure.search.documents import SearchClient

from models import Document
from generation.prompts import PromptBuilder
from generation.service  import GenerationService


class Summariser:
    """
    Summarises a single document that has already been indexed in Azure AI Search.

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

    def run(self, doc: Document) -> str:
        """
        Fetch all chunks for `doc` and return a structured AI summary.

        Raises
        ------
        ValueError  – if no chunks are found for doc.doc_id
        """
        raw = self._client.search(
            search_text = "*",
            filter      = f"doc_id eq '{doc.doc_id}'",
            select      = ["text", "chunk_index"],
            top         = 1000,
            order_by    = ["chunk_index asc"],
        )
        chunks_text = [r["text"] for r in raw]
        if not chunks_text:
            raise ValueError(
                f'No indexed chunks found for document "{doc.name}" '
                f'(doc_id={doc.doc_id}).'
            )

        full_text = "\n\n".join(chunks_text)
        user_msg  = f'Summarise the document titled "{doc.name}":\n\n{full_text}'

        answer, _, _ = self._gen_svc.generate_once(
            user_msg   = user_msg,
            system     = PromptBuilder.summary_system(),
            max_tokens = 1200,
        )
        return answer
