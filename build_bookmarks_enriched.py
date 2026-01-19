from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


APP_ROOT = Path(__file__).resolve().parent
PDF_DIR = APP_ROOT / "docs" / "schema_pdfs"
BOOKMARKS_PATH = PDF_DIR / "bookmarks.json"

PPDM39_CATALOG = APP_ROOT / "schema_registry" / "ppdm_39_schema_domain.json"
LITE_CATALOG = APP_ROOT / "schema_registry" / "ppdm_lite_schema_domain.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _fqn(schema: str, table: str) -> str:
    schema = (schema or "").strip()
    table = (table or "").strip()
    if not schema or not table:
        return ""
    return f"{schema}.{table}"


def load_row_catalog(path: Path, root_key_candidates: List[str]) -> List[dict]:
    """
    Supports your "row-based schema registry" JSON shape.
    Tries known root keys, otherwise falls back to first list found.
    """
    if not path.exists():
        return []
    data = _read_json(path)

    if isinstance(data, dict):
        for k in root_key_candidates:
            rows = data.get(k)
            if isinstance(rows, list) and rows:
                return rows

        # fallback: first list value
        for v in data.values():
            if isinstance(v, list) and v:
                return v

    if isinstance(data, list):
        return data

    return []


def build_category_to_tables(rows: List[dict]) -> Dict[str, List[str]]:
    """
    Row shape expected (your schema registry):
      category, sub_category, table_schema, table_name
    """
    cat_map: Dict[str, set] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue

        cat = r.get("category") or r.get("Category") or ""
        schema = r.get("table_schema") or r.get("TABLE_SCHEMA") or ""
        table = r.get("table_name") or r.get("TABLE_NAME") or ""

        if not cat or not schema or not table:
            continue

        k = _norm(str(cat))
        cat_map.setdefault(k, set()).add(_fqn(str(schema), str(table)))

    out: Dict[str, List[str]] = {}
    for k, s in cat_map.items():
        out[k] = sorted(s)
    return out


def detect_model_from_pdf_name(pdf_name: str) -> str:
    n = (pdf_name or "").lower()
    if "lite" in n:
        return "lite"
    return "ppdm39"


def choose_default_table(section_name: str, tables: List[str]) -> str:
    sec = _norm(section_name)
    preferred = [
        ("wells", "dbo.well"),
        ("production", "dbo.prod"),
        ("seismic", "dbo.seis_set"),
        ("areas", "dbo.area"),
        ("business associates", "dbo.business_associate"),
        ("sources", "dbo.source_document"),
        ("stratigraphy", "dbo.strat_unit"),
    ]
    for key, fqn in preferred:
        if key in sec and fqn in tables:
            return fqn
    return tables[0] if tables else ""


def enrich_bookmarks(bookmarks: dict) -> Tuple[dict, int]:
    """
    bookmarks format:
    {
      "<pdf_name>.pdf": {
         "<Section>": { "page": 130, "category": "Wells", "table": "", "tables": [] },
         ...
      },
      ...
    }
    """
    ppdm39_rows = load_row_catalog(
        PPDM39_CATALOG,
        ["ppdm_39_schema_domain", "ppdm39_schema_domain", "rows"],
    )
    lite_rows = load_row_catalog(
        LITE_CATALOG,
        ["ppdm_lite_schema_domain", "ppdm_lite11_schema_domain", "rows"],
    )

    ppdm39_cat = build_category_to_tables(ppdm39_rows)
    lite_cat = build_category_to_tables(lite_rows)

    updated = 0

    for pdf_name, sections in list(bookmarks.items()):
        if not isinstance(sections, dict):
            continue

        model = detect_model_from_pdf_name(pdf_name)
        cat_map = lite_cat if model == "lite" else ppdm39_cat

        for sec_name, item in list(sections.items()):
            if not isinstance(item, dict):
                continue

            category = item.get("category") or sec_name
            cat_key = _norm(str(category))
            tables = cat_map.get(cat_key, [])

            # soft match if exact missing (helps "HSE / Incidents" style)
            if not tables and cat_key:
                for k, v in cat_map.items():
                    if cat_key in k or k in cat_key:
                        tables = v
                        break

            old_tables = item.get("tables")
            if not isinstance(old_tables, list):
                old_tables = []

            merged = sorted(set([t for t in old_tables if isinstance(t, str) and t.strip()] + tables))

            item.setdefault("page", 1)
            item.setdefault("category", category)
            item["tables"] = merged

            if merged and not (item.get("table") or "").strip():
                item["table"] = choose_default_table(sec_name, merged)

            sections[sec_name] = item
            updated += 1

        bookmarks[pdf_name] = sections

    return bookmarks, updated


def main() -> None:
    if not BOOKMARKS_PATH.exists():
        raise SystemExit(
            f"Missing {BOOKMARKS_PATH}\n"
            "Create docs/schema_pdfs/bookmarks.json first (your seed file is fine),\n"
            "or rename/copy it to bookmarks.json."
        )

    bookmarks = _read_json(BOOKMARKS_PATH)
    if not isinstance(bookmarks, dict):
        raise SystemExit("bookmarks.json must be a dict at top-level.")

    enriched, n = enrich_bookmarks(bookmarks)
    BOOKMARKS_PATH.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Enriched {n} bookmark entries.")
    print(f"Wrote: {BOOKMARKS_PATH}")


if __name__ == "__main__":
    main()
