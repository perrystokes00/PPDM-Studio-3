# ppdm_loader/help_chat.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# OpenAI python SDK (install in your .venv): pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = "gpt-4.1-mini"

DEFAULT_PPDM39_SCHEMA_JSON = Path(r"schema_registry/ppdm_39_schema_domain.json")
DEFAULT_LITE_SCHEMA_JSON = Path(r"schema_registry/ppdm_lite_schema_domain.json")


@dataclass
class HelpChatConfig:
    model: str = DEFAULT_MODEL
    title: str = "PPDM Studio — Help Chat"
    context_hint: str = ""
    ppdm39_schema_path: Path = DEFAULT_PPDM39_SCHEMA_JSON
    lite_schema_path: Path = DEFAULT_LITE_SCHEMA_JSON


# -----------------------------
# Schema registry helpers
# -----------------------------
def _load_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _detect_root_key(d: dict) -> Optional[str]:
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k
    for k in ("ppdm_39_schema_domain", "ppdm_lite_schema_domain"):
        if isinstance(d.get(k), list):
            return k
    return None


def _rows_for_table(rows: List[dict], table_fqn: str) -> List[dict]:
    tfqn = (table_fqn or "").strip()
    if not tfqn:
        return []
    tfqn_u = tfqn.upper()

    out: List[dict] = []
    for r in rows:
        sch = str(r.get("table_schema") or "").strip()
        tbl = str(r.get("table_name") or "").strip()
        if not sch or not tbl:
            continue
        fqn = f"{sch}.{tbl}"
        if fqn.upper() == tfqn_u:
            out.append(r)
    return out


def _summarize_table(rows: List[dict]) -> Dict[str, Any]:
    cols: List[str] = []
    pk: List[str] = []
    fks: List[Dict[str, str]] = []
    guidish: List[str] = []

    for r in rows:
        col = str(r.get("column_name") or "").strip()
        if not col:
            continue
        cols.append(col)

        if str(r.get("is_primary_key") or "").upper() == "YES":
            pk.append(col)

        if str(r.get("is_foreign_key") or "").upper() == "YES":
            fks.append(
                {
                    "column": col,
                    "ref_table": f"{r.get('fk_table_schema')}.{r.get('fk_table_name')}",
                    "ref_column": str(r.get("fk_column_name") or ""),
                }
            )

        if re.search(r"(GUID|UUID)", col, flags=re.IGNORECASE):
            guidish.append(col)

    def _uniq(xs: List[str]) -> List[str]:
        seen = set()
        out2 = []
        for x in xs:
            xu = x.upper()
            if xu in seen:
                continue
            seen.add(xu)
            out2.append(x)
        return out2

    return {
        "columns": _uniq(cols),
        "pk_columns": _uniq(pk),
        "fks": fks,
        "guidish_columns": _uniq(guidish),
    }


def _build_facts_block(*, cfg: HelpChatConfig, target_table_fqn: str) -> str:
    ppdm39 = _load_json(cfg.ppdm39_schema_path)
    lite = _load_json(cfg.lite_schema_path)

    facts_lines: List[str] = []
    facts_lines.append("FACTS (ground truth for PPDM Studio):")
    facts_lines.append("- Only use facts explicitly listed below. If missing, say you cannot confirm.")
    facts_lines.append("")

    def add_model_block(name: str, data: Optional[dict]) -> None:
        if not data:
            facts_lines.append(f"{name}: schema registry NOT LOADED.")
            facts_lines.append("")
            return

        root = _detect_root_key(data)
        if not root:
            facts_lines.append(f"{name}: could not find row-list root key in schema registry.")
            facts_lines.append("")
            return

        rows = data.get(root) or []
        if not isinstance(rows, list):
            facts_lines.append(f"{name}: root '{root}' is not a list.")
            facts_lines.append("")
            return

        facts_lines.append(f"{name}: schema registry loaded (root='{root}', rows={len(rows):,}).")

        trows = _rows_for_table(rows, target_table_fqn) if target_table_fqn else []
        if target_table_fqn:
            if not trows:
                facts_lines.append(f"- Target table '{target_table_fqn}' NOT FOUND in {name} registry.")
                facts_lines.append("")
                return

            s = _summarize_table(trows)
            facts_lines.append(f"- Target table: {target_table_fqn}")
            facts_lines.append(f"- PK columns: {', '.join(s['pk_columns']) if s['pk_columns'] else '(none detected in registry)'}")

            if s["fks"]:
                facts_lines.append("- FK columns:")
                for fk in s["fks"][:50]:
                    facts_lines.append(f"  - {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}")
            else:
                facts_lines.append("- FK columns: (none detected in registry)")

            if s["guidish_columns"]:
                facts_lines.append(f"- GUID/UUID-like columns present: {', '.join(s['guidish_columns'])}")
            else:
                facts_lines.append("- GUID/UUID-like columns present: NONE detected for this table.")
            facts_lines.append("")
        else:
            facts_lines.append("- No target table provided; only general answers allowed.")
            facts_lines.append("")

    add_model_block("PPDM 3.9", ppdm39)
    add_model_block("PPDM Lite", lite)

    facts_lines.append("RULES:")
    facts_lines.append("- Never invent column names, PK/FK strategies, or table names.")
    facts_lines.append("- If uncertain: propose how to verify in SQL Server (read-only query).")

    return "\n".join(facts_lines)


def _system_prompt(ctx: dict, facts_block: str) -> str:
    return f"""You are "PPDM Studio Help Chat" — a careful, grounded PPDM assistant.

You MUST follow these rules:
1) Use ONLY the FACTS block below for PPDM schema claims (columns, PKs, FKs, keys, IDs).
2) If a fact is not present, say you cannot confirm from the loaded schema registry.
3) Do NOT guess. Do NOT hallucinate.
4) When you give guidance, keep it actionable for PPDM Studio workflows (load → normalize → validate → match & map → promote).
5) If the user asks for database verification, provide a READ-ONLY SQL query (SELECT only).

Current context:
- ERD PDF: {ctx.get("pdf","")}
- Section: {ctx.get("section","")}
- Page: {ctx.get("page","")}
- Target table: {ctx.get("table","")}
- Schema: {ctx.get("schema","")}

{facts_block}
"""


def _supports_chat_input() -> bool:
    return hasattr(st, "chat_input") and callable(getattr(st, "chat_input"))


# -----------------------------
# Chat UI
# -----------------------------
def render_help_chat_panel(cfg: HelpChatConfig, *, context: Optional[dict] = None) -> None:
    st.subheader(cfg.title)
    if cfg.context_hint:
        st.caption(cfg.context_hint)

    ctx = context or {}
    target_table = (ctx.get("table") or "").strip()

    facts_block = _build_facts_block(cfg=cfg, target_table_fqn=target_table)
    system = _system_prompt(ctx, facts_block)

    ss = st.session_state
    ss.setdefault("help_chat_messages", [])  # [{role, content}]
    ss.setdefault("help_chat_pending_input", "")

    # Readiness checks (DO NOT return early; keep UI visible)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openai_ready = (OpenAI is not None) and bool(api_key)

    if OpenAI is None:
        st.warning("OpenAI package not installed. In your .venv run:  pip install openai")
    if not api_key:
        st.warning("OPENAI_API_KEY not set. Set it in your environment or Streamlit secrets.")
    if openai_ready:
        st.success("Help Chat is ready.")

    with st.expander("Grounding (facts the assistant is allowed to use)", expanded=False):
        st.code(facts_block)

    # Show conversation so far
    for m in ss["help_chat_messages"]:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        with st.chat_message(role):
            st.write(content)

    # Input UI (chat_input if available, else fallback)
    user_text = None
    if _supports_chat_input():
        user_text = st.chat_input("Ask PPDM Studio a question…")
    else:
        st.info("Your Streamlit version doesn’t support st.chat_input(). Using fallback input.")
        user_text = st.text_area("Ask PPDM Studio a question…", key="help_chat_fallback_text", height=90)
        if st.button("Send", type="primary", key="help_chat_fallback_send"):
            user_text = (user_text or "").strip()
        else:
            user_text = None

    if not user_text:
        return

    # Add user message
    ss["help_chat_messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # If not ready, stop AFTER recording the question
    if not openai_ready:
        with st.chat_message("assistant"):
            st.error("Help Chat isn’t ready to answer yet (missing OpenAI install or API key). Fix the warnings above and ask again.")
        return

    # Call OpenAI
    client = OpenAI(api_key=api_key)

    history = ss["help_chat_messages"][-20:]
    input_items = [{"role": "system", "content": system}]
    input_items.extend(history)

    try:
        resp = client.responses.create(
            model=cfg.model,
            input=input_items,
            temperature=0.2,
        )
        answer = (resp.output_text or "").strip()
        if not answer:
            answer = "(No response text returned.)"
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"OpenAI call failed: {e}")
        return

    ss["help_chat_messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
