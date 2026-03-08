"""Tab 2: Document Summarisation."""
import streamlit as st

from agent import RAGAgent


def render(agent: RAGAgent) -> None:
    st.markdown("### 📋 Document Summarisation")
    st.caption(
        "All chunks for the selected document are fetched from Azure AI Search "
        "and summarised by GPT-4o mini into a structured report."
    )

    docs = agent.list_documents()
    if not docs:
        st.warning("No documents indexed yet.")
        return

    doc_names = {d.name: d.doc_id for d in docs}
    selected  = st.selectbox("Select document to summarise", list(doc_names.keys()))

    if st.button("Generate Summary ▶", key="sum_btn"):
        with st.spinner(f'Fetching chunks for "{selected}" → summarising…'):
            try:
                summary = agent.summarise(doc_names[selected])
                st.divider()
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error: {e}")
