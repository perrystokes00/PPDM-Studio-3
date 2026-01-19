# pages/97_Schema_PDF_Viewer.py
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PPDM Studio — Schema ERD Viewer", layout="wide")
st.title("Schema ERD Viewer")

# ============================================================
# Paths
# ============================================================
PDF_DIR = Path("docs/schema_pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

BOOKMARKS_PRIMARY = PDF_DIR / "bookmarks.json"
BOOKMARKS_SEED_39 = PDF_DIR / "bookmarks_seed_ppdm39_toc.json"

# ============================================================
# Helpers
# ============================================================
def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _merge_bookmarks(*objs: object) -> dict:
    """
    Returns merged bookmarks dict in shape:
      { "<pdf_name>": { "<label>": {"page": int, "table": str, ...}, ... }, ... }
    Later inputs win on collisions.
    """
    out: dict = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        for pdf_name, sections in o.items():
            if not isinstance(pdf_name, str):
                continue
            if pdf_name not in out:
                out[pdf_name] = {}
            if isinstance(sections, dict):
                # merge section dicts
                for label, item in sections.items():
                    if not isinstance(label, str):
                        continue
                    if isinstance(item, dict):
                        out[pdf_name][label] = item
                    else:
                        # tolerate minimal item shapes like "label": 12
                        out[pdf_name][label] = {"page": int(item) if str(item).isdigit() else 1, "table": ""}
    return out


def _get_pdf_names(pdf_files: list[Path]) -> list[str]:
    return [p.name for p in pdf_files]


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ============================================================
# Load PDFs
# ============================================================
pdf_files = sorted(PDF_DIR.glob("*.pdf"))
if not pdf_files:
    st.info(f"No PDFs found in {PDF_DIR.resolve()}. Drop your PPDM ERD PDFs there.")
    st.stop()

pdf_names = _get_pdf_names(pdf_files)

# ============================================================
# Load bookmarks (optional)
# ============================================================
bm_primary = _load_json(BOOKMARKS_PRIMARY) if BOOKMARKS_PRIMARY.exists() else None
bm_seed39 = _load_json(BOOKMARKS_SEED_39) if BOOKMARKS_SEED_39.exists() else None
bookmarks = _merge_bookmarks(bm_seed39, bm_primary)

# ============================================================
# Session defaults
# ============================================================
ss = st.session_state
ss.setdefault("erd_pdf_name", pdf_names[0])
ss.setdefault("erd_page", 1)
ss.setdefault("erd_zoom", 125)
ss.setdefault("erd_section", "")
ss.setdefault("explorer_target_fqn", "")

# ============================================================
# Query params (linking)
# ============================================================
qp = st.query_params
qp_pdf = qp.get("pdf")
qp_page = qp.get("page")
qp_zoom = qp.get("zoom")
qp_section = qp.get("section")
qp_table = qp.get("table")

# Apply query params gently (do not fight widgets)
if isinstance(qp_pdf, str) and qp_pdf in pdf_names:
    ss["erd_pdf_name"] = qp_pdf
if qp_page is not None:
    ss["erd_page"] = max(1, _safe_int(qp_page, ss.get("erd_page", 1)))
if qp_zoom is not None:
    ss["erd_zoom"] = _safe_int(qp_zoom, ss.get("erd_zoom", 125))
if isinstance(qp_section, str) and qp_section.strip():
    ss["erd_section"] = qp_section.strip()
if isinstance(qp_table, str) and qp_table.strip():
    ss["explorer_target_fqn"] = qp_table.strip()

# ============================================================
# Sidebar — Bookmarks + quick linking
# ============================================================
with st.sidebar:
    st.subheader("Bookmarks")

    pdf_name = st.selectbox(
        "Select ERD PDF",
        pdf_names,
        index=pdf_names.index(ss["erd_pdf_name"]) if ss["erd_pdf_name"] in pdf_names else 0,
        key="erd_pdf_pick_sidebar",
    )
    ss["erd_pdf_name"] = pdf_name

    # sections for selected pdf
    b_for_pdf = bookmarks.get(pdf_name, {})
    if not isinstance(b_for_pdf, dict):
        b_for_pdf = {}

    if b_for_pdf:
        labels = list(b_for_pdf.keys())
        # try to keep last section selected
        default_idx = 0
        if ss.get("erd_section") in labels:
            default_idx = labels.index(ss["erd_section"])

        pick = st.selectbox(
            "Section",
            labels,
            index=default_idx,
            key="erd_section_pick_sidebar",
        )
        ss["erd_section"] = pick

        item = b_for_pdf.get(pick) or {}
        if not isinstance(item, dict):
            item = {}
        page_hint = int(item.get("page", 1) or 1)
        table_hint = (item.get("table") or "").strip()
        tables_hint = item.get("tables") if isinstance(item.get("tables"), list) else []

        st.caption(f"Bookmark page: {page_hint}")
        if table_hint:
            st.caption(f"Primary table: {table_hint}")
        elif tables_hint:
            st.caption(f"Tables: {', '.join(tables_hint[:5])}{'…' if len(tables_hint) > 5 else ''}")

        if st.button("Go to bookmark", type="primary", key="erd_go_bookmark"):
            ss["erd_page"] = max(1, int(page_hint))
            if table_hint:
                ss["explorer_target_fqn"] = table_hint

            # write shareable params
            st.query_params["pdf"] = ss["erd_pdf_name"]
            st.query_params["page"] = str(int(ss["erd_page"]))
            st.query_params["zoom"] = str(int(ss["erd_zoom"]))
            st.query_params["section"] = ss.get("erd_section", "")
            if ss.get("explorer_target_fqn"):
                st.query_params["table"] = ss.get("explorer_target_fqn", "")

            st.rerun()
    else:
        st.caption("No bookmarks found for this PDF.")
        st.caption("Expected one of these files:")
        st.code(str(BOOKMARKS_PRIMARY))
        st.code(str(BOOKMARKS_SEED_39))

    st.divider()

    st.subheader("Linking")
    target = (ss.get("explorer_target_fqn") or "").strip()
    if target:
        st.success(f"Linked table: {target}")
        if st.button("Open linked table in Data Explorer", key="erd_open_explorer_sidebar"):
            st.switch_page("pages/98_Data_Explorer.py")
    else:
        st.caption("No linked table yet (bookmark may set one).")

# ============================================================
# Main controls
# ============================================================
c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    pdf_name_main = st.selectbox(
        "Pick a PDF",
        pdf_names,
        index=pdf_names.index(ss["erd_pdf_name"]) if ss["erd_pdf_name"] in pdf_names else 0,
        key="erd_pdf_pick_main",
    )
    ss["erd_pdf_name"] = pdf_name_main

with c2:
    page_val = st.number_input(
        "Page",
        min_value=1,
        value=int(ss["erd_page"]),
        step=1,
        key="erd_page_input",
    )
    ss["erd_page"] = int(page_val)

with c3:
    zoom_val = st.select_slider(
        "Zoom",
        options=[50, 75, 100, 125, 150, 200, 250, 300],
        value=int(ss["erd_zoom"]),
        key="erd_zoom_input",
    )
    ss["erd_zoom"] = int(zoom_val)

st.caption(f"Viewing: **{ss['erd_pdf_name']}** | Page: **{ss['erd_page']}** | Zoom: **{ss['erd_zoom']}%**")

# Keep query params updated (shareable URL)
st.query_params["pdf"] = ss["erd_pdf_name"]
st.query_params["page"] = str(int(ss["erd_page"]))
st.query_params["zoom"] = str(int(ss["erd_zoom"]))
if ss.get("erd_section"):
    st.query_params["section"] = ss.get("erd_section", "")
if ss.get("explorer_target_fqn"):
    st.query_params["table"] = ss.get("explorer_target_fqn", "")

# ============================================================
# Link to Data Explorer (main area)
# ============================================================
target = (ss.get("explorer_target_fqn") or "").strip()
if target:
    st.info(f"Linked target table: **{target}**")
    if st.button("Open linked table in Data Explorer", type="primary", key="btn_open_explorer_main"):
        st.switch_page("pages/98_Data_Explorer.py")

# ============================================================
# Render / Open PDF (robust)
# ============================================================
pdf_path = PDF_DIR / ss["erd_pdf_name"]
if not pdf_path.exists():
    st.error(f"PDF not found: {pdf_path.resolve()}")
    st.stop()

st.subheader("ERD Document")

left, right = st.columns([1, 2])

with left:
    st.download_button(
        "📄 Download / Open PDF",
        data=pdf_path.read_bytes(),
        file_name=ss["erd_pdf_name"],
        mime="application/pdf",
        key="erd_pdf_download",
    )

    st.caption("Use your browser's PDF viewer (fastest + most reliable).")
    st.caption("Tip: After you open it, jump to the same page shown above.")

with right:
    st.info(
        f"""
If the inline viewer is blank on this machine/browser, use **Download / Open PDF**.

Current navigation:
- PDF: `{ss['erd_pdf_name']}`
- Page: {ss['erd_page']}
- Zoom: {ss['erd_zoom']}%
- Section: `{ss.get('erd_section','')}`
"""
    )

with st.expander("Troubleshooting (if you still want inline PDFs)", expanded=False):
    st.markdown(
        """
Some environments block inline PDF rendering in iframes (CSP / browser PDF plugin / security settings).
If you want to try inline again later:
- Use Chrome/Edge latest
- Disable strict privacy extensions
- Ensure the browser can open PDFs normally
"""
    )
