from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Try PyMuPDF first (best for outlines + fast text)
try:
    import fitz  # pip install pymupdf
except Exception:
    fitz = None

# Fallback for text extraction if needed
try:
    import pdfplumber  # pip install pdfplumber
except Exception:
    pdfplumber = None


PDF_DIR = Path("docs/schema_pdfs")
BOOKMARKS_PATH = PDF_DIR / "bookmarks.json"

# -----------------------------
# Heuristics: detect table names in text
# -----------------------------
# Common patterns you’ll see on ERD pages:
#  - dbo.well_dir_srvy_station
#  - WELL_DIR_SRVY_STATION (maybe without schema)
#  - sometimes bracketed: [dbo].[well]
TABLE_FQN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b"
)

BRACKET_FQN_RE = re.compile(
    r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*\.\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)

# Optional: if your PDFs show lots of “dbo” plus table names
# this helps you capture bracketed and unbracketed.
def find_table_fqns(text: str, schema_hint: Optional[str] = None) -> List[str]:
    out = set()

    for m in BRACKET_FQN_RE.finditer(text):
        out.add(f"{m.group(1)}.{m.group(2)}")

    for m in TABLE_FQN_RE.finditer(text):
        out.add(f"{m.group(1)}.{m.group(2)}")

    # If schema isn’t present, you can add a second-pass “schema_hint”
    # (disabled by default because it can create false positives).
    # Example: detect WELL_LOG and convert to dbo.well_log if schema_hint="dbo"
    if schema_hint:
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
        # Very conservative: only take tokens that look like table names you care about
        # Customize later (eg. only tokens that also exist in your catalog).
        # out.update({f"{schema_hint}.{t.lower()}" for t in tokens if t.isupper() and len(t) > 4})

    return sorted(out)


# -----------------------------
# Extract outline / TOC if present
# -----------------------------
def extract_outline_pymupdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Returns [(page_1based, title), ...] from PDF outline.
    """
    if fitz is None:
        return []

    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)  # [level, title, page]
    doc.close()

    out = []
    for item in toc or []:
        if len(item) >= 3:
            _level, title, page = item[0], item[1], item[2]
            if isinstance(page, int) and page > 0:
                out.append((page, str(title).strip()))
    return out


# -----------------------------
# Page text scanning (PyMuPDF or pdfplumber)
# -----------------------------
def scan_pages_for_tables(pdf_path: Path) -> Dict[str, int]:
    """
    Returns { 'dbo.table': first_page_1based, ... }
    """
    table_to_page: Dict[str, int] = {}

    # Fast path: PyMuPDF
    if fitz is not None:
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            fqns = find_table_fqns(text)
            page_no = i + 1
            for fqn in fqns:
                if fqn not in table_to_page:
                    table_to_page[fqn] = page_no
        doc.close()
        return table_to_page

    # Fallback: pdfplumber
    if pdfplumber is None:
        return table_to_page

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            fqns = find_table_fqns(text)
            page_no = i + 1
            for fqn in fqns:
                if fqn not in table_to_page:
                    table_to_page[fqn] = page_no

    return table_to_page


# -----------------------------
# Load / merge existing bookmarks.json
# -----------------------------
def load_bookmarks() -> Dict[str, Any]:
    if BOOKMARKS_PATH.exists():
        try:
            data = json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_bookmarks(data: Dict[str, Any]) -> None:
    BOOKMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOKMARKS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in: {PDF_DIR.resolve()}")
        return

    existing = load_bookmarks()

    if fitz is None and pdfplumber is None:
        print("ERROR: Need either PyMuPDF or pdfplumber installed.")
        print("Try: pip install pymupdf")
        print("  or: pip install pdfplumber")
        return

    updated = dict(existing)

    for pdf_path in pdfs:
        pdf_name = pdf_path.name
        print(f"\n== {pdf_name} ==")

        # Ensure pdf key exists
        if pdf_name not in updated or not isinstance(updated.get(pdf_name), dict):
            updated[pdf_name] = {}

        pdf_map: Dict[str, Any] = updated[pdf_name]

        # 1) Outline -> bookmarks (if exists)
        outline = extract_outline_pymupdf(pdf_path)
        if outline:
            print(f"Found outline entries: {len(outline)}")
            for page_no, title in outline[:500]:  # cap just in case
                key = slugify(title)[:80] or f"outline_p{page_no}"
                # only add if not present
                pdf_map.setdefault(key, {"page": int(page_no)})
        else:
            print("No outline detected (or outline extraction unavailable).")

        # 2) Scan pages for table fqns
        table_hits = scan_pages_for_tables(pdf_path)
        print(f"Detected table references: {len(table_hits)}")
        # Add a section per table, but keep it compact
        for fqn, page_no in sorted(table_hits.items(), key=lambda kv: kv[1]):
            key = slugify(fqn.replace(".", "_"))  # eg dbo.well -> dbo_well
            # Only add if key not already present
            if key not in pdf_map:
                pdf_map[key] = {"page": int(page_no), "table": fqn}

        updated[pdf_name] = pdf_map

    save_bookmarks(updated)
    print(f"\nWrote: {BOOKMARKS_PATH.resolve()}")
    print("Next: open the ERD Viewer and see new bookmark sections.")


if __name__ == "__main__":
    main()
