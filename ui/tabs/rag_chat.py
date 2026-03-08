"""Tab 1: RAG Chat — hybrid search + GPT-4o mini + inline citations."""
from __future__ import annotations

import streamlit as st

from agent         import RAGAgent
from cost_model    import CostModel
from ui.helpers    import render_answer, score_bar_html
from ui.styles     import AZ_LBLUE, AZ_GRAY, AZ_YELLOW


def render(agent: RAGAgent) -> None:
    st.markdown("### 💬 RAG Chat")
    st.caption(
        "Hybrid search (BM25 + HNSW vector + Semantic Reranker) · "
        "GPT-4o mini generation · inline citations · follow-up context"
    )

    # ── Render conversation history ───────────────────────────────────────────
    for role, content, chunks in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {content}</div>',
                unsafe_allow_html=True,
            )
        else:
            rendered = render_answer(content, chunks) if content else "<em>(no response)</em>"
            st.markdown(
                f'<div class="assistant-bubble">🤖 {rendered}</div>',
                unsafe_allow_html=True,
            )
            if chunks:
                with st.expander(
                    f"🔍 Retrieved context — {len(chunks)} chunks from Azure AI Search",
                    expanded=False,
                ):
                    _render_chunk_cards(chunks)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.divider()
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Ask a question about your documents",
            placeholder      = "e.g. What are the main security controls discussed?",
            label_visibility = "collapsed",
            key              = "rag_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True, key="rag_send")

    # Prevent double-submit on rapid clicks
    last_q = st.session_state.get("_last_submitted_query", "")
    if send and query.strip() and query.strip() != last_q:
        st.session_state["_last_submitted_query"] = query.strip()

        with st.spinner("Embedding query → Azure AI Search → GPT-4o mini generating…"):
            try:
                msg = agent.rag_query(query.strip(), top_k=agent.cfg.top_k)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.session_state.chat_history.append(("user",      query.strip(), []))
        st.session_state.chat_history.append(("assistant", msg.content,   msg.chunks))

        if msg.input_tokens or msg.output_tokens:
            est = CostModel.estimate(
                msg.input_tokens, msg.output_tokens,
                semantic=agent.cfg.use_semantic_rank,
            )
            st.caption(
                f"💰 Cost — embed: ${est.embed_usd:.5f} · "
                f"search: ${est.search_usd:.5f} · "
                f"llm: ${est.llm_input_usd + est.llm_output_usd:.5f} · "
                f"**total: ${est.total:.5f}**"
            )

        st.rerun()


def _render_chunk_cards(chunks: list) -> None:
    for i, c in enumerate(chunks, 1):
        composite = max(c.score, c.reranker_score)
        bar       = score_bar_html(composite)
        sem_str   = (
            f' · sem <b style="color:{AZ_YELLOW}">{c.reranker_score:.3f}</b>'
            if c.reranker_score else ""
        )
        num_badge = (
            f'<span class="cite-num" '
            f'style="font-size:11px;width:20px;height:20px">{i}</span>'
        )
        st.markdown(
            f'<div class="chunk-card">'
            f'{num_badge} '
            f'<b style="color:{AZ_LBLUE}">{c.doc_name}</b> '
            f'<span style="color:{AZ_GRAY}">p.{c.page_num}</span><br>'
            f'{bar} '
            f'<span class="chunk-score">rrf {c.score:.3f}</span>{sem_str}<br>'
            f'<span style="color:{AZ_GRAY};font-size:12px">{c.text[:200]}…</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
