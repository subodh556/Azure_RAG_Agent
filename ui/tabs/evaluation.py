"""Tab 4: Azure AI Evaluation Pipeline."""
from __future__ import annotations

import streamlit as st

from agent         import RAGAgent
from ui.helpers    import render_answer
from ui.styles     import AZ_GREEN, AZ_YELLOW, AZ_RED

try:
    from azure.ai.evaluation import RelevanceEvaluator  # noqa: F401
    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False


def render(agent: RAGAgent) -> None:
    st.markdown("### 🧪 Azure AI Evaluation Pipeline")
    st.caption(
        "Azure AI Evaluation SDK · Relevance · Groundedness · Coherence · Fluency "
        "· All scored 1–5 by GPT-4o mini as judge · Powered by `azure-ai-evaluation`"
    )

    if not _EVAL_AVAILABLE:
        st.error(
            "**`azure-ai-evaluation` is not installed.**\n\n"
            "Run: `pip install azure-ai-evaluation==0.3.3` and restart the app."
        )
        return

    presets = [
        "How does Azure AI Search handle security and encryption?",
        "What chunking strategies improve RAG retrieval quality?",
        "Explain hybrid search: BM25 plus vector with RRF fusion.",
        "What are common production RAG failure modes?",
        "— Custom query —",
    ]

    selected_preset = st.selectbox("Preset query", presets, key="eval_preset")
    eval_query = (
        st.text_input(
            "Enter evaluation query",
            placeholder="Type your query here…",
            key="eval_custom",
        )
        if selected_preset == "— Custom query —"
        else selected_preset
    )

    top_k_eval = st.slider(
        "Chunks to retrieve (top-k)", 1, 10, agent.cfg.top_k, key="eval_topk"
    )

    if st.button("▶ Run Azure AI Evaluation", key="eval_btn"):
        if not eval_query.strip():
            st.warning("Please enter or select a query.")
            return
        if not agent.list_documents():
            st.warning("No documents indexed. Upload PDFs first.")
            return

        with st.spinner(
            "Retrieving from Azure AI Search → generating RAG answer → "
            "running Azure AI Evaluation SDK evaluators…"
        ):
            try:
                retrieval, answer, scores = agent.evaluate(
                    eval_query.strip(), top_k=top_k_eval
                )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return

        # Retrieval metrics
        st.divider()
        st.markdown("**🔍 Retrieval — Azure AI Search**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks retrieved", len(retrieval.chunks))
        m2.metric("Latency", f"{retrieval.latency_ms} ms")
        m3.metric("Search mode",
                  "Hybrid + Semantic" if retrieval.semantic_ranked else "Hybrid")
        top_score = max((c.score for c in retrieval.chunks), default=0)
        m4.metric("Top RRF score", f"{top_score:.4f}")

        with st.expander(
            f"Retrieved chunks ({len(retrieval.chunks)}) from Azure AI Search",
            expanded=False,
        ):
            from ui.tabs.rag_chat import _render_chunk_cards
            _render_chunk_cards(retrieval.chunks)

        # Generated answer
        st.divider()
        st.markdown("**🤖 Generated RAG Answer (what is being evaluated)**")
        st.markdown(
            f'<div class="assistant-bubble">'
            f'{render_answer(answer, retrieval.chunks)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Scores
        st.divider()
        st.markdown("**📊 Azure AI Evaluation SDK — LLM-Judge Scores (1–5)**")

        vc = scores.verdict_color
        st.markdown(
            f'<div style="background:{vc}22;border:2px solid {vc};'
            f'border-radius:10px;padding:14px 20px;margin-bottom:16px">'
            f'<span style="font-size:22px;font-weight:800;color:{vc}">'
            f'{scores.verdict}</span>'
            f'<span style="color:#e2e8f0;font-size:14px;margin-left:16px">'
            f'Overall score: <b>{scores.overall} / 5.0</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        def _card(col, label: str, score: float, reason: str, icon: str) -> None:
            color = AZ_GREEN if score >= 4.0 else AZ_YELLOW if score >= 2.5 else AZ_RED
            col.markdown(
                f'<div class="cost-card">'
                f'<div style="font-size:22px">{icon}</div>'
                f'<div class="cost-total" style="color:{color}">{score:.1f}</div>'
                f'<div class="cost-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if reason:
                col.caption(reason)

        _card(c1, "Relevance",    scores.relevance,    scores.relevance_reason,    "🎯")
        _card(c2, "Groundedness", scores.groundedness, scores.groundedness_reason, "⚓")
        _card(c3, "Coherence",    scores.coherence,    scores.coherence_reason,    "🔗")
        _card(c4, "Fluency",      scores.fluency,      scores.fluency_reason,      "✍️")

        st.divider()
        st.markdown("**📖 Evaluator Definitions**")
        import pandas as pd
        st.dataframe(
            pd.DataFrame([
                ("🎯 Relevance",    "RelevanceEvaluator",
                 "Does the response directly address what the query is asking?",
                 "query + response + context"),
                ("⚓ Groundedness",  "GroundednessEvaluator",
                 "Is every claim in the response supported by the retrieved context?",
                 "query + response + context"),
                ("🔗 Coherence",    "CoherenceEvaluator",
                 "Is the response logically structured and easy to follow?",
                 "query + response"),
                ("✍️ Fluency",      "FluencyEvaluator",
                 "Is the response grammatically correct and natural to read?",
                 "query + response"),
            ], columns=["Metric", "SDK Class", "Definition", "Inputs"]).set_index("Metric"),
            use_container_width=True,
        )
        st.caption(
            "All evaluators use GPT-4o mini as the judge via "
            "`AzureOpenAIModelConfiguration`. Scores: **1–5 integer scale**. "
            "Source: `azure-ai-evaluation` SDK — "
            "[docs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-sdk)"
        )
