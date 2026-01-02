# pages/4_QC_Reports.py
from __future__ import annotations
import streamlit as st
from common.ui import sidebar_connect, require_connection

st.set_page_config(page_title="QC Reports", layout="wide")
sidebar_connect(page_prefix="qc")
conn = require_connection()

st.title("QC Reports (Coming online)")

st.write("Ideas:")
st.markdown(
    """
- Orphan FK checks (child values not in parent)
- Missing reference coverage summary
- Promote QC summary (from your PROMOTE_QC_TABLE)
- Row counts by domain and table
"""
)
