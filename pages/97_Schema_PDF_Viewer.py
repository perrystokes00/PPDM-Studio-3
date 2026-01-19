# pages/97_Schema_PDF_Viewer.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

# Optional dependency:
#   pip install streamlit-image-zoom
try:
    from streamlit_image_zoom import image_zoom
except Exception:
    image_zoom = None

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="PPDM Studio — Schema ERD Viewer", layout="wide")
st.title("Schema ERD Viewer (Image Render)")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PDF_DIR = Path("docs/schema_pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

BOOKMARKS_OUT = PDF_DIR / "bookmarks.json"

# ------------------------------------------------------------
# Heuristics for headings & tables
# ------------------------------------------------------------
HEADING_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 /&\-\+()]*[A-Z0-9)]$")

HEADING_BLACKLIST_SUBSTR = [
    "PPDM 3.9",
    "PPDM 3",
    "DIAGRAMS LEGEND",
    "PROFESSIONAL PETROLEUM",
    "PUBLIC PETROLEUM",
    "DATA MODEL",
    "TABLE OF CONTENTS",
    "TERMS AND CONDITIONS",
]

# PPDM 3.9 diagrams often contain lines like: TABLE_NAME (PPDM39)
TABLE_PATTERN_39 = re.compile(r"^([A-Z0-9_]+)\s*\((PPDM39|PPDM 3\.9|PPDM3\.9)\)", re.IGNORECASE)

# Sometimes tables appear already schema-qualified (rare but helpful)
TABLE_PATTERN_DBO = re.compile(r"^(dbo\.[A-Za-z0-9_]+)\b")

# Lite heuristic: many Lite table names start with L_
TABLE_PATTERN_LITE = re.compile(r"^(L_[A-Z0-9_]+)\b", re.IGNORECASE)


def _is_heading_candidate(line: str) -> bool:
    if not line:
        return False
    if not HEADING_PATTERN.match(line):
        return False
    up = line.upper()
    for bad in HEADING_BLACKLIST_SUBSTR:
        if bad in up:
            return False
    return True


# ------------------------------------------------------------
# Caching: open pdf as resource; structure as data (pickle-safe)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _open_pdf(pdf_path: str) -> fitz.Document:
    return fitz.open(pdf_path)


@st.cache_data(show_spinner=True)
def extract_structure(pdf_path_str: str) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, List[int]]]:
    """
    Returns ONLY pickle-safe objects:
      headings_to_tables: dict[str, list[str]]
      heading_to_page: dict[str, int]           (0-based)
      table_to_pages: dict[str, list[int]]      (0-based)
    """
    doc = fitz.open(pdf_path_str)

    headings_to_tables: Dict[str, Set[str]] = {}
    heading_to_page: Dict[str, int] = {}
    table_to_pages: Dict[str, Set[int]] = {}

    for page_idx, page in enumerate(doc):
        text = page.get_text("text") or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        current_heading: Optional[str] = None

        # Find heading near the top
        for line in lines[:10]:
            if _is_heading_candidate(line):
                current_heading = line
                headings_to_tables.setdefault(current_heading, set())
                heading_to_page.setdefault(current_heading, page_idx)
                break

        # Find tables on the page
        for line in lines:
            l = line.strip()
            if not l:
                continue

            # dbo.<table>
            m_dbo = TABLE_PATTERN_DBO.match(l)
            if m_dbo:
                tbl = m_dbo.group(1).strip()
                table_to_pages.setdefault(tbl, set()).add(page_idx)
                if current_heading:
                    headings_to_tables.setdefault(current_heading, set()).add(tbl)
                continue

            # TABLE_NAME (PPDM39)
            m_39 = TABLE_PATTERN_39.match(l)
            if m_39:
                tbl = m_39.group(1).strip().upper()
                table_to_pages.setdefault(tbl, set()).add(page_idx)
                if current_heading:
                    headings_to_tables.setdefault(current_heading, set()).add(tbl)
                continue

            # L_TABLE_NAME (Lite)
            m_l = TABLE_PATTERN_LITE.match(l)
            if m_l:
                tbl = m_l.group(1).strip().upper()
                table_to_pages.setdefault(tbl, set()).add(page_idx)
                if current_heading:
                    headings_to_tables.setdefault(current_heading, set()).add(tbl)
                continue

    # sets -> sorted lists (pickle-safe)
    headings_to_tables_out = {k: sorted(v) for k, v in headings_to_tables.items()}
    table_to_pages_out = {k: sorted(v) for k, v in table_to_pages.items()}

    return headings_to_tables_out, heading_to_page, table_to_pages_out


def _list_tables(headings_to_tables: Dict[str, List[str]]) -> List[str]:
    s: Set[str] = set()
    for tbls in headings_to_tables.values():
        for t in (tbls or []):
            s.add(t)
    return sorted(s)


def _to_dataframe(headings_to_tables: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for heading, tables in sorted(headings_to_tables.items()):
        if not tables:
            continue
        rows.append({"Heading": heading.strip(), "Tables": ", ".join(sorted(tables))})
    return pd.DataFrame(rows)


def _render_page(doc: fitz.Document, page_idx: int, zoom_mode: str, hq_scale: float) -> None:
    page = doc[page_idx]

    if zoom_mode.startswith("Fast"):
        base_scale = 1.5
        caption_extra = "Fast mode (image zoom only – may pixelate at high zoom)"
    else:
        base_scale = float(hq_scale)
        caption_extra = f"High-quality mode (PDF re-render at scale {base_scale:.1f}×)"

    mat = fitz.Matrix(base_scale, base_scale)
    pix = page.get_pixmap(matrix=mat)

    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

    st.caption(f"Page {page_idx + 1} — {caption_extra}")

    if image_zoom is not None:
        image_zoom(
            img,
            mode="both",
            size=(min(img.width, 1800), min(img.height, 1200)),
        )
    else:
        st.warning("Optional dependency missing: streamlit-image-zoom. Showing static image.")
        st.image(img, use_container_width=True)


def _load_bookmarks(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_bookmarks(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_bookmarks_for_pdf(
    pdf_name: str,
    heading_to_page: Dict[str, int],
    headings_to_tables: Dict[str, List[str]],
    out_path: Path,
) -> None:
    """
    Viewer sidebar format:
      {
        "Some.pdf": {
          "Section Label": { "page": 130, "category": "...", "table": "", "tables": [] },
          ...
        }
      }
    Pages stored as 1-based for humans.
    """
    payload = _load_bookmarks(out_path)

    pdf_block = payload.get(pdf_name, {})
    if not isinstance(pdf_block, dict):
        pdf_block = {}

    for heading, p0 in heading_to_page.items():
        tables = headings_to_tables.get(heading, []) or []
        default_table = ""
        # If we detect a schema-qualified table, prefer it
        for t in tables:
            if isinstance(t, str) and t.lower().startswith("dbo."):
                default_table = t
                break

        pdf_block.setdefault(
            heading,
            {
                "page": int(p0) + 1,
                "category": heading,
                "table": default_table,
                "tables": [t for t in tables if isinstance(t, str)],
            },
        )

        # keep page updated
        pdf_block[heading]["page"] = int(p0) + 1

        # keep tables updated (non-destructive-ish)
        if "tables" not in pdf_block[heading] or not isinstance(pdf_block[heading]["tables"], list):
            pdf_block[heading]["tables"] = []
        # union existing + newly detected
        merged = set([str(x) for x in (pdf_block[heading]["tables"] or [])])
        merged.update([str(x) for x in tables])
        pdf_block[heading]["tables"] = sorted(merged)

        # keep default table if blank
        if not pdf_block[heading].get("table") and default_table:
            pdf_block[heading]["table"] = default_table

    payload[pdf_name] = pdf_block
    _save_bookmarks(out_path, payload)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
pdf_files = sorted(PDF_DIR.glob("*.pdf"))
if not pdf_files:
    st.info(f"No PDFs found in {PDF_DIR.resolve()}. Put your ERD PDFs there (PPDM 3.9 + Lite).")
    st.stop()

pdf_names = [p.name for p in pdf_files]

ss = st.session_state
ss.setdefault("erd_pdf_name", pdf_names[0])
ss.setdefault("erd_page", 1)                # 1-based for display
ss.setdefault("erd_section", "")
ss.setdefault("explorer_target_fqn", "")

# Left/Right layout
col_nav, col_opts = st.columns([2, 1])

with col_opts:
    zoom_mode = st.radio("Zoom quality", ["Fast (image zoom only)", "High quality (PDF re-render)"], key="erd_zoom_mode")
    if zoom_mode.startswith("High"):
        hq_scale = st.slider(
            "PDF render scale",
            min_value=2.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
            help="Higher values = sharper but slower rendering.",
            key="erd_hq_scale",
        )
    else:
        hq_scale = 1.5

    nav_mode = st.radio("Go to:", ["By heading", "By table name", "By page number"], key="erd_nav_mode")

with col_nav:
    pdf_name = st.selectbox(
        "Pick an ERD PDF",
        pdf_names,
        index=pdf_names.index(ss["erd_pdf_name"]) if ss["erd_pdf_name"] in pdf_names else 0,
        key="erd_pdf_pick",
    )
    ss["erd_pdf_name"] = pdf_name

    pdf_path = PDF_DIR / pdf_name
    headings_to_tables, heading_to_page, table_to_pages = extract_structure(str(pdf_path))
    headings = sorted(headings_to_tables.keys())
    tables = _list_tables(headings_to_tables)

    doc = _open_pdf(str(pdf_path))

    page_idx: Optional[int] = None

    if nav_mode == "By heading":
        if headings:
            heading = st.selectbox("Choose heading", headings, key="erd_heading_pick")
            ss["erd_section"] = heading
            page_idx = heading_to_page.get(heading)

            t_on = headings_to_tables.get(heading, []) or []
            st.markdown(f"**Heading:** {heading}")
            if t_on:
                st.markdown("**Tables detected on this diagram:**")
                st.write(", ".join(sorted(t_on)))
        else:
            st.warning("No headings detected in this PDF (OCR/text may be limited).")

    elif nav_mode == "By table name":
        if tables:
            table = st.selectbox("Choose table", tables, key="erd_table_pick")
            pages = table_to_pages.get(table, []) or []
            st.markdown(f"**Table:** {table}")
            if not pages:
                st.warning("No pages found for this table.")
            else:
                page_idx = st.selectbox(
                    "Occurrences (page)",
                    pages,
                    key="erd_table_page_pick",
                    format_func=lambda p: f"Page {p + 1}",
                )
        else:
            st.warning("No tables detected in this PDF (OCR/text may be limited).")

    elif nav_mode == "By page number":
        page_idx = int(
            st.number_input(
                "Page number (1-based)",
                min_value=1,
                max_value=len(doc),
                value=min(max(int(ss.get("erd_page", 1)), 1), len(doc)),
                step=1,
                key="erd_page_num",
            )
        ) - 1

# Render selected page
if page_idx is not None:
    ss["erd_page"] = int(page_idx) + 1
    _render_page(doc, int(page_idx), zoom_mode, float(hq_scale))

st.caption(f"Viewing: {ss['erd_pdf_name']} | Page: {ss['erd_page']}")

# --------------------------------------------------
# Link to Data Explorer (context-aware)
# --------------------------------------------------
with st.expander("Link to Data Explorer", expanded=False):

    # Tables detected on the currently viewed page
    page_tables: List[str] = []
    if page_idx is not None:
        for tbl, pages in (table_to_pages or {}).items():
            try:
                if int(page_idx) in list(pages):
                    page_tables.append(str(tbl))
            except Exception:
                continue

    page_tables = sorted(set(page_tables))
    all_tables = sorted([str(k) for k in (table_to_pages or {}).keys()])

    if page_tables:
        st.caption("Tables detected on this diagram page")
        table_choice = st.selectbox(
            "Select table",
            page_tables,
            key="erd_link_table_page",
        )
    elif all_tables:
        st.caption("No tables detected on this page — showing all tables in PDF")
        table_choice = st.selectbox(
            "Select table",
            all_tables,
            key="erd_link_table_all",
        )
    else:
        st.warning("No tables detected in this PDF.")
        table_choice = ""

    if table_choice:
        st.code(table_choice, language="sql")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Set linked table", type="secondary", key="btn_set_linked_table"):
                ss["explorer_target_fqn"] = table_choice
                st.success(f"Linked: {table_choice}")

        with c2:
            if st.button("Open in Data Explorer", type="primary", key="btn_open_in_explorer"):
                ss["explorer_target_fqn"] = table_choice
                st.switch_page("pages/98_Data_Explorer.py")

# --------------------------------------------------
# Export bookmarks.json
# --------------------------------------------------
with st.expander("Export bookmarks.json from detected headings", expanded=False):
    st.write("Writes/updates:", str(BOOKMARKS_OUT.resolve()))
    if st.button("Export bookmarks for this PDF", type="primary", key="erd_export_bookmarks"):
        _export_bookmarks_for_pdf(
            pdf_name=ss["erd_pdf_name"],
            heading_to_page=heading_to_page,
            headings_to_tables=headings_to_tables,
            out_path=BOOKMARKS_OUT,
        )
        st.success("Bookmarks exported/updated.")

# --------------------------------------------------
# Data view
# --------------------------------------------------
with st.expander("Heading → tables mapping (data view)", expanded=False):
    df = _to_dataframe(headings_to_tables)
    st.dataframe(df, use_container_width=True)
