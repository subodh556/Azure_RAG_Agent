"""Tab 3: Cross-Document Comparison."""
import streamlit as st

from agent import RAGAgent


def render(agent: RAGAgent) -> None:
    st.markdown("### ⚖️ Cross-Document Comparison")
    st.caption(
        "Full text of both documents is retrieved from Azure AI Search and "
        "analysed by GPT-4o mini for thematic similarities and differences."
    )

    docs = agent.list_documents()
    if len(docs) < 2:
        st.warning("At least 2 indexed documents are required for comparison.")
        return

    doc_names = {d.name: d.doc_id for d in docs}
    names     = list(doc_names.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        doc_a_name = st.selectbox("Document A", names, index=0, key="cmp_a")
    with col_b:
        doc_b_name = st.selectbox(
            "Document B", names, index=min(1, len(names) - 1), key="cmp_b"
        )

    if st.button("Compare Documents ▶", key="cmp_btn"):
        if doc_a_name == doc_b_name:
            st.warning("Please select two different documents.")
            return
        with st.spinner("Fetching both documents → GPT-4o mini comparing…"):
            try:
                result = agent.compare(doc_names[doc_a_name], doc_names[doc_b_name])
                st.divider()
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {e}")
