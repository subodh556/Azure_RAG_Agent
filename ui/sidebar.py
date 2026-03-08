"""
Streamlit sidebar: document upload panel + index statistics + document list.
"""
from __future__ import annotations

import streamlit as st

from agent            import RAGAgent
from core.pdf_extractor import PDFExtractor
from ui.styles        import AZ_LBLUE, AZ_GRAY


def render_sidebar(agent: RAGAgent) -> None:
    """Render the full sidebar including the upload panel."""
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:12px 0">'
            f'<span style="font-size:28px">🔷</span><br>'
            f'<b style="color:{AZ_LBLUE};font-size:15px">Azure AI Search</b><br>'
            f'<span style="color:{AZ_GRAY};font-size:12px">RAG Agent · GPT-4o mini</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**📤 Upload PDF Documents**")
        st.caption(
            "Select one or more PDFs. Each file is extracted in memory, "
            "chunked, embedded via Azure OpenAI, and indexed in Azure AI Search."
        )

        uploaded_files = st.file_uploader(
            label                 = "Choose PDF files",
            type                  = ["pdf"],
            accept_multiple_files = True,
            label_visibility      = "collapsed",
            key                   = "pdf_uploader",
        )

        if st.button(
            "📥 Index Uploaded PDFs",
            use_container_width = True,
            disabled            = not uploaded_files,
            key                 = "upload_btn",
        ):
            _process_uploads(agent, uploaded_files)

        # ── Index statistics ──────────────────────────────────────────────────
        try:
            chunk_count = agent.index_mgr.get_doc_count()
        except Exception:
            pass

        docs = agent.list_documents()

        # ── Indexed document list ─────────────────────────────────────────────
        if docs:
            st.markdown("**📄 Indexed Documents**")
            for doc in docs:
                pages = doc.metadata.get("pages", "?")
                size  = doc.metadata.get("file_size_kb", "?")
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<b style="color:{AZ_LBLUE}">{doc.name}</b><br>'
                    f'<span style="color:{AZ_GRAY};font-size:11px">'
                    f'{pages} pages · {doc.word_count:,} words · {size} KB'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            agent.clear_history()
            st.success("History cleared.")

        st.divider()
        st.markdown(
            f'<div style="color:{AZ_GRAY};font-size:11px;text-align:center">'
            f'azure-search-documents SDK<br>'
            f'openai SDK · PyMuPDF · Streamlit'
            f'</div>',
            unsafe_allow_html=True,
        )


def _process_uploads(agent: RAGAgent, uploaded_files: list) -> None:
    if not uploaded_files:
        return

    results  = {"ok": [], "skip": [], "err": []}
    progress = st.progress(0, text="Processing uploads…")
    total    = len(uploaded_files)

    for i, uf in enumerate(uploaded_files):
        progress.progress(i / total, text=f"Processing {uf.name}…")
        try:
            doc = PDFExtractor.from_bytes(uf.read(), uf.name)
            n, skipped = agent.add_document(doc)
            if skipped:
                results["skip"].append(uf.name)
            else:
                if doc.doc_id not in st.session_state.uploaded_doc_ids:
                    st.session_state.uploaded_doc_ids.append(doc.doc_id)
                results["ok"].append(
                    f'**"{doc.name}"** — {n} chunks '
                    f'({doc.metadata.get("pages","?")} pages, {doc.word_count:,} words)'
                )
        except PermissionError as e:
            results["err"].append(f"{uf.name}: {e}")
        except ValueError as e:
            results["err"].append(f"{uf.name}: {e}")
        except Exception as e:
            results["err"].append(f"{uf.name}: Unexpected error — {e}")

    progress.progress(1.0, text="Done.")
    st.divider()

    if results["ok"]:
        st.success(f'✔ {len(results["ok"])} file(s) indexed successfully')
        for msg in results["ok"]:
            st.markdown(f"  — {msg}")
    if results["skip"]:
        st.info(f'ℹ {len(results["skip"])} file(s) already indexed (skipped)')
        for name in results["skip"]:
            st.markdown(f"  — {name}")
    if results["err"]:
        st.error(f'✘ {len(results["err"])} file(s) failed')
        for msg in results["err"]:
            st.markdown(f"  — {msg}")

    st.rerun()
