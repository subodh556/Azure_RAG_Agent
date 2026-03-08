"""
Prompt templates for all RAG capabilities.

System prompts are STABLE (no retrieved chunks) so they can be reused
across turns without poisoning the conversation history window.

Each user message is self-contained: it carries the SOURCES block +
the QUESTION for that specific turn.
"""
from __future__ import annotations

import textwrap

from models import RetrievalResult


class PromptBuilder:

    # ── RAG Chat ──────────────────────────────────────────────────────────────

    @staticmethod
    def rag_system() -> str:
        """Stable citation-enforcing system prompt — contains no sources."""
        return textwrap.dedent("""\
            You are a precise research assistant. Answer the user's question
            using ONLY the numbered source chunks provided in the user message.

            CITATION RULES — follow exactly:
            1. Cite INLINE: place the source number in square brackets
               immediately after the sentence or phrase it supports.
               Example: "The index uses BM25 for full-text search [1] and
               HNSW for dense vector retrieval [2]."
            2. Multiple sources for one claim are fine: [1][3]
            3. Use markdown formatting (bold, bullets, numbered lists) where
               it helps clarity.
            4. Do NOT add a References section or repeat sources at the end.
            5. If the answer cannot be found in the provided sources, reply
               exactly (no other text):
               "I could not find an answer in the indexed documents."
            6. Never fabricate facts not present in the sources.
        """)

    @staticmethod
    def rag_user_message(query: str, retrieval: RetrievalResult) -> str:
        """
        Build the self-contained user-turn message for a RAG query.

        Format
        ──────
        SOURCES (N chunks · mode · latency):

        [1] "Document Name" | page P
        <chunk text>

        [2] "Document Name" | page P
        <chunk text>

        ---

        QUESTION: <query>
        """
        mode = (
            "Hybrid BM25+HNSW"
            + ("+Semantic" if retrieval.semantic_ranked else "")
            + f" · {retrieval.latency_ms} ms"
        )
        parts: list[str] = [
            f"SOURCES ({len(retrieval.chunks)} chunks · {mode}):",
            "",
        ]
        for i, c in enumerate(retrieval.chunks, 1):
            parts.append(f'[{i}] "{c.doc_name}" | page {c.page_num}')
            parts.append(c.text.strip())
            parts.append("")

        parts += ["---", "", f"QUESTION: {query}"]
        return "\n".join(parts)

    # ── Summarise ─────────────────────────────────────────────────────────────

    @staticmethod
    def summary_system() -> str:
        return textwrap.dedent("""\
            You are a senior document analyst. Produce a structured summary
            with these exact sections:

            ## Executive Summary
            (2-3 sentences: core subject, purpose, key takeaway)

            ## Key Topics
            - (4-6 bullet points)

            ## Main Arguments / Findings
            1. (3-6 numbered items with detail)

            ## Notable Details
            (unique or critical specifics worth highlighting)

            ## Suggested Follow-up Questions
            1.
            2.
            3.
        """)

    # ── Cross-document Compare ────────────────────────────────────────────────

    @staticmethod
    def compare_system(doc_a: str, doc_b: str, text_a: str, text_b: str) -> str:
        return textwrap.dedent(f"""\
            You are a cross-document analysis specialist. Compare the two
            documents below using these exact sections:

            ## Overview
            One-sentence description of each document.

            ## Common Themes
            Shared topics, approaches, or conclusions.

            ## Key Differences
            Where they diverge in scope, methodology, or depth.

            ## Unique to "{doc_a}"
            Topics or claims that appear only in Document A.

            ## Unique to "{doc_b}"
            Topics or claims that appear only in Document B.

            ## Synthesis & Recommendation
            What a practitioner learns from reading both documents together.

            ---
            DOCUMENT A — "{doc_a}":
            {text_a}

            ---
            DOCUMENT B — "{doc_b}":
            {text_b}
        """)
