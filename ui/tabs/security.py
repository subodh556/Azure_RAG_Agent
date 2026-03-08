"""Tab 5: Security Matrix — RBAC, security controls, Bicep IaC."""
import streamlit as st
import pandas as pd

from security import RBAC_MATRIX, SECURITY_CONTROLS, BICEP_IaC
from ui.styles import AZ_GREEN, AZ_GRAY


def render() -> None:
    st.markdown("### 🔒 Data Access Control Matrix")
    st.caption("Azure RBAC · Entra ID · OData security trimming · IaC (Bicep)")

    # RBAC table
    st.markdown("**Role-Based Access Control (RBAC)**")
    rows = []
    for role, read, write, delete, manage, logs, net in RBAC_MATRIX:
        tick = lambda v: "✅" if v else "❌"  # noqa: E731
        rows.append({
            "Role":         role,
            "Read":         tick(read),
            "Write":        tick(write),
            "Delete":       tick(delete),
            "Manage Index": tick(manage),
            "View Logs":    tick(logs),
            "Network":      tick(net),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Role"), use_container_width=True)

    # Security controls
    st.divider()
    st.markdown("**Security Controls**")
    for name, _status, detail in SECURITY_CONTROLS:
        col_icon, col_name, col_detail = st.columns([0.5, 2.5, 6])
        col_icon.markdown(
            f'<span style="color:{AZ_GREEN};font-size:18px">●</span>',
            unsafe_allow_html=True,
        )
        col_name.markdown(f"**{name}**")
        col_detail.markdown(
            f'<span style="color:{AZ_GRAY}">{detail}</span>',
            unsafe_allow_html=True,
        )

   
