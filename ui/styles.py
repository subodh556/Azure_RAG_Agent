"""
Azure dark-theme color palette and global CSS.

Call inject_css() once at the top of app.py (after st.set_page_config).
"""
import streamlit as st

# ── Azure blue palette ────────────────────────────────────────────────────────
AZ_BLUE   = "#0078d4"
AZ_LBLUE  = "#60a8f8"
AZ_GREEN  = "#22c55e"
AZ_RED    = "#ef4444"
AZ_YELLOW = "#fbbf24"
AZ_GRAY   = "#94a3b8"
AZ_DARK   = "#0d1829"


def inject_css() -> None:
    """Inject the global dark-theme CSS into the Streamlit page."""
    st.markdown("""
<style>
    .stApp { background-color: #080c14; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0d1829; }
    h1, h2, h3 { color: #60a8f8 !important; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1829;
        border-bottom: 2px solid #0078d4;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        background-color: #0d1829;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background-color: #0078d4 !important;
    }

    .user-bubble {
        background: #1e3a5f;
        border-left: 3px solid #0078d4;
        border-radius: 0 8px 8px 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .assistant-bubble {
        background: #0d1829;
        border-left: 3px solid #22c55e;
        border-radius: 0 8px 8px 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
    }

    .cite-wrap {
        position: relative;
        display: inline-block;
        vertical-align: super;
        line-height: 0;
        margin: 0 1px;
    }
    .cite-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 16px;
        height: 16px;
        padding: 0 4px;
        background: #0078d4;
        color: #fff;
        font-size: 10px;
        font-weight: 700;
        border-radius: 999px;
        cursor: default;
        white-space: nowrap;
        user-select: none;
    }
    .cite-tooltip {
        display: none;
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        width: 320px;
        background: #111c2e;
        color: #e2e8f0;
        font-size: 12px;
        line-height: 1.5;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #0078d4;
        box-shadow: 0 6px 24px rgba(0,0,0,0.6);
        z-index: 9999;
        white-space: normal;
        pointer-events: none;
        vertical-align: initial;
        line-height: initial;
    }
    .cite-tooltip .cite-tip-title {
        color: #60a8f8;
        font-weight: 700;
        font-size: 11px;
        margin-bottom: 6px;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 4px;
    }
    .cite-tooltip .cite-tip-body {
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.5;
    }
    .cite-wrap:hover .cite-tooltip { display: block; }

    .assistant-bubble p  { margin: 0 0 8px 0; }
    .assistant-bubble ul { margin: 4px 0 8px 1em; padding-left: 1em; }
    .assistant-bubble ol { margin: 4px 0 8px 1em; padding-left: 1.2em; }
    .assistant-bubble li { margin-bottom: 4px; line-height: 1.6; }
    .assistant-bubble strong { color: #e2e8f0; }
    .assistant-bubble h1,
    .assistant-bubble h2,
    .assistant-bubble h3 { color: #60a8f8 !important; margin: 10px 0 4px 0; }

    .chunk-card {
        background: #0d1829;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
    }
    .chunk-score { color: #0078d4; font-weight: 700; font-size: 12px; }

    .pill-ok   { background:#14532d; color:#22c55e; padding:2px 10px; border-radius:12px; font-size:12px; }
    .pill-warn { background:#451a03; color:#fbbf24; padding:2px 10px; border-radius:12px; font-size:12px; }
    .pill-err  { background:#450a0a; color:#ef4444; padding:2px 10px; border-radius:12px; font-size:12px; }

    .cost-card {
        background: #0d1829;
        border: 1px solid #0078d4;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .cost-total { font-size: 28px; font-weight: 800; color: #fbbf24; }
    .cost-label { font-size: 12px; color: #94a3b8; margin-top: 4px; }

    [data-testid="stMetric"] { background: #0d1829; border-radius: 8px; padding: 12px; }
    [data-testid="stMetricValue"] { color: #60a8f8 !important; }

    .stTextInput input, .stTextArea textarea {
        background-color: #0d1829 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e3a5f !important;
    }
    .stButton > button {
        background-color: #0078d4;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #106ebe; }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0d1829 !important;
        color: #e2e8f0 !important;
        border-color: #1e3a5f !important;
    }
    .streamlit-expanderHeader  { color: #60a8f8 !important; }
    .streamlit-expanderContent { background: #0d1829; }
    .stDataFrame { background: #0d1829; }
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)
