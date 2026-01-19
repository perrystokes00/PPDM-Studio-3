# pages/99_Help_Chat.py
import streamlit as st
from ppdm_loader.help_chat import render_help_chat_panel, HelpChatConfig

st.set_page_config(page_title="PPDM Studio — Help Chat", layout="wide")

ss = st.session_state

ctx = {
    "pdf": ss.get("erd_pdf_name", ""),
    "section": ss.get("erd_section", ""),
    "page": ss.get("erd_page", ""),
    "table": ss.get("explorer_target_fqn", "") or ss.get("primary_table_fqn", "") or ss.get("primary_fqn", ""),
    "schema": ss.get("expl_schema", ""),
}

render_help_chat_panel(
    HelpChatConfig(
        model="gpt-4.1-mini",
        title="PPDM Studio — Help Chat",
        context_hint=f"Current target: {ctx['table']}" if ctx.get("table") else "",
    ),
    context=ctx,
)
