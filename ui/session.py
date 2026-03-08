"""
Streamlit session state management and agent bootstrap.

_init_session()    — initialises all session_state keys with defaults
_bootstrap_agent() — creates and caches the RAGAgent (runs once per session)
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from agent  import RAGAgent
from config import AzureConfig

try:
    import fitz  # noqa: F401
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False


def init_session() -> None:
    """Initialise all session_state keys with safe defaults."""
    defaults = {
        "agent":            None,
        "chat_history":     [],    # list of (role, content, chunks)
        "init_log":         [],
        "init_done":        False,
        "init_error":       None,
        "uploaded_doc_ids": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(show_spinner=False)
def bootstrap_agent() -> tuple[Optional[RAGAgent], list[str], Optional[str]]:
    """
    Create and cache the RAGAgent — runs exactly once per Streamlit session.

    Returns (agent, log_messages, error_string).
    A non-None error_string means startup failed and agent is None.
    """
    log: list[str] = []

    try:
        cfg = AzureConfig.from_env()
    except EnvironmentError as e:
        return None, [], str(e)

    log.append(f"✔ Azure AI Search : `{cfg.search_endpoint}`")
    log.append(f"✔ Azure OpenAI    : `{cfg.openai_endpoint}`")
    log.append(f"✔ Index name      : `{cfg.index_name}`")
    log.append(f"✔ Embed model     : `{cfg.embed_deployment}` ({cfg.embed_dimensions}-dim)")
    log.append(f"✔ Chat model      : `{cfg.chat_deployment}` (GPT-4o mini)")
    log.append(
        f"✔ Semantic rank   : `{'enabled' if cfg.use_semantic_rank else 'disabled'}`"
    )

    if not _PYMUPDF_AVAILABLE:
        return None, log, "PyMuPDF not installed. Run: pip install pymupdf==1.24.0"

    agent = RAGAgent(cfg)
    try:
        agent.bootstrap_index()
        log.append(f'✔ Index `"{cfg.index_name}"` ready (HNSW + BM25 + Semantic)')
    except Exception as e:
        return None, log, f"Index creation failed: {e}"

    return agent, log, None
