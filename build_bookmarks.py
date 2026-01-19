from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import fitz  # pip install pymupdf


PDF_DIR = Path("docs/schema_pdfs")
BOOKMARKS_PATH = PDF_DIR / "bookmarks.json"

# ---------- parsing helpers ----------

# Matches: "130  Wells" or "Wells  130" (we handle both)
RE_NUM_TITLE = re.compile(r"^\s*(\d{1,4})\s+(.+?)\s*$")
RE_TITLE_NUM = re.compile(r"^\s*(.+?)\s+(\d{1,4})\s*$")

def _clean_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _extract_toc_lines(page: fitz.Page) -> List[str]:
    """
    Extract text lines from a TOC page using PyMuPDF.
    For two-column layouts, PyMuPDF usually returns lines in reading order.
    """
    text = page.get_text("text") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines

def _parse_toc_lines_to_bookmarks(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Turn TOC lines into: { "Section": {"page": <int>, "table": ""}, ... }

    Handles both patterns:
      "130 Wells"
      "Wells 130"
    """
    out: Dict[str, Dict[str, Any]] = {}

    # Skip obvious headers
    skip_prefix = ("table of contents", "contents")
    for ln in lines:
        lnl = ln.lower().strip()
        if any(lnl.startswith(p) for p in skip_prefix):
            continue

        # Try "num title"
        m = RE_NUM_TITLE.match(ln)
        if m:
            page_no = int(m.group(1))
            label = _clean_label(m.group(2))
            if label and label.lower() not in skip_prefix:
                out[label] = {"page": page_no, "table": ""}
            continue

        # Try "title num"
        m = RE_TITLE_NUM.match(ln)
        if m:
            label = _clean_label(m.group(1))
            page_no = int(m.group(2))
            if label and label.lower() not in skip_prefix:
                out[label] = {"page": page_no, "table": ""}
            continue

    return out

def _bookmarks_from_pdf_outline(doc: fitz.Document) -> Dict[str, Dict[str, Any]]:
    """
    If the PDF has native bookmarks, use them.
    doc.get_toc() returns [level, title, page] where page is 1-based.
    """
    toc = doc.get_toc(simple=True)  # [[lvl, title, page], ...]
    out: Dict[str, Dict[str, Any]] = {}
    if not toc:
        return out

    for lvl, title, page in toc:
        title = _clean_label(title)
        if not title:
            continue

        # Keep it simple: flatten label; you can prefix with level if you want
        label = title
        # Avoid collisions
        if label in out:
            label = f"{label} (L{lvl})"

        out[label] = {"page": int(page or 1), "table": ""}
    return out

def _find_toc_page_index(doc: fitz.Document, max_scan_pages: int = 15) -> int | None:
    """
    Scan first N pages looking for 'Table of Contents' text.
    Returns 0-based page index or None.
    """
    for i in range(min(max_scan_pages, len(doc))):
        page = doc.load_page(i)
        t = (page.get_text("text") or "").lower()
        if "table of contents" in t or re.search(r"\bcontents\b", t):
            return i
    return None

def build_bookmarks_for_pdf(pdf_path: Path) -> Dict[str, Dict[str, Any]]:
    doc = fitz.open(pdf_path)

    # 1) Prefer native outline/bookmarks if present
    outline = _bookmarks_from_pdf_outline(doc)
    if outline:
        return outline

    # 2) Otherwise, parse TOC page text if present
    toc_idx = _find_toc_page_index(doc)
    if toc_idx is not None:
        lines = _extract_toc_lines(doc.load_page(toc_idx))
        toc_bookmarks = _parse_toc_lines_to_bookmarks(lines)
        if toc_bookmarks:
            return toc_bookmarks

    # 3) Nothing found
    return {}

def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {PDF_DIR.resolve()}")

    all_bookmarks: Dict[str, Any] = {}
    for pdf in pdf_files:
        bm = build_bookmarks_for_pdf(pdf)
        all_bookmarks[pdf.name] = bm
        print(f"{pdf.name}: {len(bm)} bookmarks")

    BOOKMARKS_PATH.write_text(json.dumps(all_bookmarks, indent=2), encoding="utf-8")
    print(f"Wrote: {BOOKMARKS_PATH.resolve()}")

if __name__ == "__main__":
    main()
