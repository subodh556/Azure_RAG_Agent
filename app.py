#!/usr/bin/env python3
"""
Azure AI Search RAG Agent — Streamlit entry point.

Run
───
    streamlit run app.py

The app module tree is flat — all sub-packages live alongside app.py so
Python resolves imports without any path manipulation.
"""

import streamlit as st

# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title            = "Azure AI Search RAG Agent",
    page_icon             = "🔷",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load everything AFTER set_page_config ────────────────────────────────────
from ui.styles  import inject_css, AZ_BLUE, AZ_GRAY
from ui.session import init_session, bootstrap_agent
from ui.sidebar import render_sidebar

from ui.tabs import rag_chat, summarise, compare, evaluation, security, cost


def main() -> None:
    inject_css()
    init_session()

    # Page header
    st.markdown(
        f'<h1 style="color:{AZ_BLUE};margin-bottom:0">🔷 Azure AI Search RAG Agent</h1>'
        f'<p style="color:{AZ_GRAY};margin-top:4px">'
        f"BM25 + HNSW Vector + Semantic Reranker · GPT-4o mini · "
        f"PDF ingestion via PyMuPDF"
        f"</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Bootstrap agent (cached)
    with st.spinner("Connecting to Azure AI Search and preparing index…"):
        agent, log, error = bootstrap_agent()

    with st.expander("🚀 Startup log", expanded=(error is not None)):
        for line in log:
            st.markdown(line)
        if error:
            st.error(f"**Startup failed:** {error}")

    if error or agent is None:
        st.stop()

    # Sidebar renders first (populates agent.docs from the upload panel)
    render_sidebar(agent)

    # Show "no documents" banner only when the index is truly empty
    try:
        index_empty = agent.index_mgr.get_doc_count() == 0
    except Exception:
        index_empty = not agent.list_documents()

    if index_empty:
        st.info(
            "**No documents indexed yet.** "
            "Use the **Upload PDF Documents** panel in the sidebar to add PDFs. "
            "They will be extracted, chunked, embedded, and indexed automatically.",
            icon="📤",
        )

    # Six capability tabs
    tabs = st.tabs([
        "💬 RAG Chat",
        "📋 Summarise",
        "⚖️ Compare",
        "🧪 Eval Pipeline",
        "🔒 Security",
        "💰 Cost Model",
    ])

    with tabs[0]: rag_chat.render(agent)
    with tabs[1]: summarise.render(agent)
    with tabs[2]: compare.render(agent)
    with tabs[3]: evaluation.render(agent)
    with tabs[4]: security.render()
    with tabs[5]: cost.render(agent)


if __name__ == "__main__":
    main()
