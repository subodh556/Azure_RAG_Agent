"""
Azure OpenAI GPT-4o mini generation service with sliding-window history.

Multi-turn history design
─────────────────────────
The key insight: what gets STORED in history must be different from what
gets SENT to the API.

  Sent to API (each turn):
    user      → full SOURCES block + question  (self-contained, current context)
    system    → stable citation-enforcing system prompt

  Stored in history (persisted across turns):
    user      → clean question only            (the `question` param)
    assistant → answer with ALL [N] stripped   (via _strip_citations)

Without this separation, on turn 2 the model would see its previous answer
containing [1][2] citation markers in history — but no sources labelled [1][2]
exist in that context.  The model cannot reconcile the dangling brackets and
either hallucinates or produces empty output.

_strip_citations() removes all [N] markers before storing, so every
subsequent turn sees plain coherent Q&A with no dangling references.
"""
from __future__ import annotations

import re

from openai import AzureOpenAI


class GenerationService:

    MAX_TURNS = 8

    def __init__(self, client: AzureOpenAI, chat_deployment: str) -> None:
        self.client          = client
        self.chat_deployment = chat_deployment
        self.history: list[dict] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_citations(text: str) -> str:
        """
        Remove all [N] citation markers from a response before storing in history.

        "The model uses BM25 [1] and HNSW [2]."
        → "The model uses BM25 and HNSW."
        """
        cleaned = re.sub(r'\s*\[\d+\]', '', text)
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned.strip()

    def _call(
        self,
        messages:    list[dict],
        system:      str,
        max_tokens:  int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Core API call shared by generate() and generate_once()."""
        all_msgs = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model       = self.chat_deployment,
            messages    = all_msgs,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        text       = response.choices[0].message.content or ""
        in_tokens  = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
        return text, in_tokens, out_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        full_user_msg: str,
        system:        str,
        question:      str,
        max_tokens:    int   = 1200,
        temperature:   float = 0.2,
    ) -> tuple[str, int, int]:
        """
        Generate a RAG answer with sliding-window conversation history.

        Parameters
        ----------
        full_user_msg : Complete message for this turn (SOURCES + QUESTION).
        system        : Stable system prompt — no sources.
        question      : Plain question string — this is what gets stored in
                        history, not the multi-KB sources block.
        """
        history_window = self.history[-(self.MAX_TURNS * 2):]
        messages       = history_window + [{"role": "user", "content": full_user_msg}]

        text, in_tok, out_tok = self._call(messages, system, max_tokens, temperature)

        # Store CLEAN history: plain question + citation-stripped answer
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": self._strip_citations(text)})

        return text, in_tok, out_tok

    def generate_once(
        self,
        user_msg:   str,
        system:     str,
        max_tokens: int = 1200,
    ) -> tuple[str, int, int]:
        """
        Stateless single-shot generation — no history read or written.
        Used for summarisation, comparison, and evaluation.
        """
        return self._call(
            [{"role": "user", "content": user_msg}],
            system, max_tokens, temperature=0.2,
        )

    def clear_history(self) -> None:
        self.history.clear()
