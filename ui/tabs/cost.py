"""Tab 6: Cost Model — monthly breakdown + per-query estimator."""
import streamlit as st
import pandas as pd

from agent         import RAGAgent
from cost_model    import CostModel
from ui.styles     import AZ_GREEN, AZ_YELLOW, AZ_GRAY


def render(agent: RAGAgent) -> None:
    st.markdown("### 💰 Cost Model & Estimation")
    st.caption(
        "Azure AI Search S1 · text-embedding-3-small · GPT-4o mini · "
        "Private Endpoints · Key Vault"
    )

    # Monthly breakdown table
    st.markdown("**Monthly Breakdown — baseline 1,000 queries/day**")
    rows = [
        {"Component": c, "Monthly Cost": m, "Daily (1K q)": d}
        for c, m, d in CostModel.MONTHLY_TABLE
    ]
    rows.append({
        "Component":    "**TOTAL**",
        "Monthly Cost": "**$1,136.30**",
        "Daily (1K q)": "**$37.88**",
    })
    st.dataframe(pd.DataFrame(rows).set_index("Component"), use_container_width=True)

    
    
