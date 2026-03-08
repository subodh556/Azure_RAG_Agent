"""
Page-aware, sentence-boundary document chunker.

Design principles
─────────────────
1. Page numbers are exact — derived from metadata["page_texts"] written by
   PDFExtractor, not from inline [Page N] markers.
2. Stored chunk text is CLEAN — no overlap prefix.  Overlap is added only
   inside embed_texts() when building embedding inputs, giving the vector
   encoder wider context without polluting stored text or citation tooltips.
3. Sentence-boundary splitting — paragraphs that exceed max_words are split
   on .!? boundaries so chunks never cut mid-sentence.
4. Short-segment merging is page-safe — segments are only merged when both
   share the same page number.
5. Deterministic chunk IDs: "{doc_id}-chunk-{i:04d}" — safe for upsert.
"""
from __future__ import annotations

import re

from models import Chunk, Document


class DocumentChunker:

    _SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, max_words: int = 200, min_words: int = 30) -> None:
        self.max_words = max_words
        self.min_words = min_words

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._SENT_SPLIT.split(text.strip()) if s.strip()]

    def _page_to_segments(
        self, page_num: int, page_text: str
    ) -> list[tuple[int, str]]:
        """Split one page's text into (page_num, segment) pairs."""
        segments: list[tuple[int, str]] = []
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', page_text) if p.strip()]

        for para in paragraphs:
            if len(para.split()) <= self.max_words:
                segments.append((page_num, para))
                continue

            # Split over-long paragraph at sentence boundaries
            current_sents: list[str] = []
            current_count = 0
            for sent in self._sentences(para):
                wc = len(sent.split())
                if current_count + wc > self.max_words and current_sents:
                    segments.append((page_num, " ".join(current_sents)))
                    current_sents = [sent]
                    current_count = wc
                else:
                    current_sents.append(sent)
                    current_count += wc
            if current_sents:
                segments.append((page_num, " ".join(current_sents)))

        return segments

    def _merge_short(
        self, segments: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """
        Merge short tail segments into their predecessor — same page only.
        Cross-page merging is forbidden to protect page attribution.
        """
        merged: list[tuple[int, str]] = []
        for page_num, seg in segments:
            if (
                merged
                and len(seg.split()) < self.min_words
                and merged[-1][0] == page_num
            ):
                prev_page, prev_text = merged[-1]
                merged[-1] = (prev_page, prev_text + " " + seg)
            else:
                merged.append((page_num, seg))
        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Convert a Document into a flat list of Chunk objects.

        Reads page-level text from doc.metadata["page_texts"] (set by
        PDFExtractor).  Falls back to doc.content as a single page for
        plain-text documents or unit tests.
        """
        page_entries: list[dict] = doc.metadata.get("page_texts") or []
        raw_pages: list[tuple[int, str]] = (
            [(e["page"], e["text"]) for e in page_entries]
            if page_entries
            else [(1, doc.content)]
        )

        raw_segments: list[tuple[int, str]] = []
        for page_num, page_text in raw_pages:
            raw_segments.extend(self._page_to_segments(page_num, page_text))

        final_segments = self._merge_short(raw_segments)

        return [
            Chunk(
                chunk_id    = f"{doc.doc_id}-chunk-{i:04d}",
                doc_id      = doc.doc_id,
                doc_name    = doc.name,
                chunk_index = i,
                page_num    = page_num,
                text        = text.strip(),
            )
            for i, (page_num, text) in enumerate(final_segments)
        ]

    def embed_texts(self, chunks: list[Chunk], overlap_words: int = 20) -> list[str]:
        """
        Build text strings for EmbeddingService.embed_batch().

        Prepends the tail of the previous chunk (overlap) so the vector
        encoder has wider cross-boundary context.  This overlap is NEVER
        stored in chunk.text — it exists only in the embedding input.
        """
        texts: list[str] = []
        for i, c in enumerate(chunks):
            if i > 0 and overlap_words > 0:
                tail = " ".join(chunks[i - 1].text.split()[-overlap_words:])
                texts.append(tail + " " + c.text)
            else:
                texts.append(c.text)
        return texts
