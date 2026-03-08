"""
PDF text extractor using PyMuPDF (fitz).

Extraction strategy
───────────────────
• page.get_text("blocks") → sorted by reading order (top→bottom, left→right)
• Text blocks only (type 0); image blocks (type 1) are skipped
• Blocks shorter than MIN_BLOCK_CHARS are discarded (page numbers / noise)
• Pages with no extractable text are silently skipped (scanned images)

Page tracking
─────────────
Each page's text is stored in metadata["page_texts"] as a list of
{"page": N, "text": "..."} dicts.  Document.content is the same text
concatenated WITHOUT any [Page N] markers — keeping content clean and
page tracking reliable for DocumentChunker.
"""
from __future__ import annotations

import pathlib
import re
from datetime import datetime, timezone

from models import Document

try:
    import fitz          # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False


class PDFExtractor:

    MIN_BLOCK_CHARS = 20
    MIN_DOC_WORDS   = 50

    @staticmethod
    def _clean_page(text: str) -> str:
        text = text.replace("\u00ad", "")        # remove soft hyphen
        text = re.sub(r"[^\S\n]+", " ", text)    # collapse horizontal whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)   # max one blank line
        return text.strip()

    @classmethod
    def from_bytes(cls, file_bytes: bytes, filename: str) -> Document:
        """
        Convert raw PDF bytes into a Document ready for indexing.

        Returns
        -------
        Document with:
          content              = full plain text (no page markers)
          metadata["page_texts"] = list of {"page": int, "text": str}

        Raises
        ------
        RuntimeError     – PyMuPDF not installed
        PermissionError  – PDF is password-protected
        ValueError       – extracted text too short (scanned/image-only PDF)
        """
        if not _PYMUPDF_AVAILABLE:
            raise RuntimeError(
                "PyMuPDF not installed. Run: pip install pymupdf==1.24.0"
            )

        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        if pdf.needs_pass:
            pdf.close()
            raise PermissionError(
                f'"{filename}" is password-protected and cannot be read.'
            )

        raw_meta   = pdf.metadata or {}
        page_texts: list[dict] = []

        for page_num, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks")
            text_blocks = sorted(
                [b for b in blocks
                 if b[6] == 0 and len(b[4].strip()) >= cls.MIN_BLOCK_CHARS],
                key=lambda b: (round(b[1] / 10) * 10, b[0]),
            )
            page_body = "\n\n".join(b[4].strip() for b in text_blocks)
            page_body = cls._clean_page(page_body)
            if page_body:
                page_texts.append({"page": page_num, "text": page_body})

        pdf.close()

        full_text = "\n\n".join(p["text"] for p in page_texts)

        if len(full_text.split()) < cls.MIN_DOC_WORDS:
            raise ValueError(
                f'"{filename}" yielded too little text after extraction. '
                "It may be a scanned / image-only PDF."
            )

        stem     = pathlib.Path(filename).stem
        doc_name = raw_meta.get("title", "").strip() or stem
        doc_id   = "pdf-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")

        return Document(
            doc_id   = doc_id,
            name     = doc_name,
            content  = full_text,
            source   = filename,
            metadata = {
                "pages":        len(page_texts),
                "title":        raw_meta.get("title", "").strip(),
                "author":       raw_meta.get("author", "").strip(),
                "file_size_kb": round(len(file_bytes) / 1024, 1),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "page_texts":   page_texts,   # consumed by DocumentChunker
            },
        )
