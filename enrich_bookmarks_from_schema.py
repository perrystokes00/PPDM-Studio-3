import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# ------------------------------------------------------------
# Paths (adjust if your folder names differ)
# ------------------------------------------------------------
ROOT = Path(r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3")

BOOKMARKS_PATH = ROOT / "docs" / "schema_pdfs" / "bookmarks.json"
PPDM39_SCHEMA_PATH = ROOT / "schema_registry" / "ppdm_39_schema_domain.json"
LITE_SCHEMA_PATH  = ROOT / "schema_registry" / "ppdm_lite_schema_domain.json"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))

def _is_rows_obj(obj: Any) -> bool:
    return isinstance(obj, list) and all(isinstance(x, dict) for x in obj[:5])

def _extract_rows(schema_obj: Any) -> List[dict]:
    """
    Supports:
      - {"ppdm_39_schema_domain":[...]}
      - {"ppdm_lite_schema_domain":[...]}
      - {"rows":[...]}
      - direct list of dict rows
    """
    if _is_rows_obj(schema_obj):
        return schema_obj

    if isinstance(schema_obj, dict):
        # common roots
        for k in ("ppdm_39_schema_domain", "ppdm_lite_schema_domain", "rows"):
            v = schema_obj.get(k)
            if _is_rows_obj(v):
                return v

        # fallback: first list-of-dict value in dict
        for v in schema_obj.values():
            if _is_rows_obj(v):
                return v

    raise ValueError("Could not find row-list root in schema JSON (expected list of dict rows).")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _fqn(schema: str, table: str) -> str:
    schema = (schema or "").strip() or "dbo"
    table  = (table or "").strip()
    return f"{schema}.{table}" if table else ""

def build_category_to_tables(rows: List[dict]) -> Dict[str, List[str]]:
    """
    rows are your schema-registry rows:
      category, sub_category, table_schema, table_name, column_name, etc.
    We index unique tables by category.
    """
    cat_map: Dict[str, Set[str]] = {}
    for r in rows:
        cat = (r.get("category") or r.get("Category") or "").strip()
        sch = (r.get("table_schema") or r.get("TABLE_SCHEMA") or r.get("schema") or "").strip()
        tbl = (r.get("table_name") or r.get("TABLE_NAME") or r.get("table") or "").strip()

        if not cat or not tbl:
            continue

        fqn = _fqn(sch, tbl)
        if not fqn:
            continue

        key = _norm(cat)
        cat_map.setdefault(key, set()).add(fqn)

    # convert to sorted lists
    out: Dict[str, List[str]] = {}
    for k, s in cat_map.items():
        out[k] = sorted(s, key=lambda x: x.lower())
    return out

def derive_category_from_bookmark_key(label: str) -> str:
    """
    For PPDM Lite bookmarks that look like:
      "1. Table of Contents"
      "3.1. Who We Are"
    or other numbered headings, strip prefix numbering.
    """
    label = (label or "").strip()

    # strip leading "12. " or "3.1. " patterns
    label2 = re.sub(r"^\s*\d+(\.\d+)*\.\s*", "", label).strip()
    return label2 or label

def enrich_bookmarks_for_pdf(
    pdf_bookmarks: Dict[str, dict],
    cat_to_tables: Dict[str, List[str]],
    *,
    prefer_well_table: str = "dbo.well",
) -> Tuple[int, int]:
    """
    Returns (updated_count, unmatched_count)
    """
    updated = 0
    unmatched = 0

    for label, item in pdf_bookmarks.items():
        if not isinstance(item, dict):
            continue

        # Determine category:
        # 1) explicit item["category"]
        # 2) else derive from label (Lite headings often need this)
        cat = (item.get("category") or "").strip()
        if not cat:
            cat = derive_category_from_bookmark_key(label)

        cat_key = _norm(cat)

        tables = cat_to_tables.get(cat_key)
        if not tables:
            unmatched += 1
            continue

        # set tables list
        item["category"] = cat  # normalize presence
        item["tables"] = tables

        # set default table if missing
        cur_table = (item.get("table") or "").strip()
        if not cur_table:
            if _norm(cat) == _norm("Wells") and prefer_well_table in tables:
                item["table"] = prefer_well_table
            else:
                item["table"] = tables[0] if tables else ""

        updated += 1

    return updated, unmatched


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    # Load inputs
    bookmarks = _load_json(BOOKMARKS_PATH)
    if not isinstance(bookmarks, dict):
        raise ValueError("bookmarks.json must be a dict keyed by PDF filename.")

    ppdm39_rows = _extract_rows(_load_json(PPDM39_SCHEMA_PATH))
    lite_rows   = _extract_rows(_load_json(LITE_SCHEMA_PATH))

    ppdm39_cat_map = build_category_to_tables(ppdm39_rows)
    lite_cat_map   = build_category_to_tables(lite_rows)

    # Enrich
    total_updated = 0
    total_unmatched = 0

    for pdf_name, pdf_bm in bookmarks.items():
        if not isinstance(pdf_bm, dict):
            continue

        # choose which category map to use
        pdf_name_l = (pdf_name or "").lower()
        if "lite" in pdf_name_l:
            cat_map = lite_cat_map
        else:
            cat_map = ppdm39_cat_map

        u, m = enrich_bookmarks_for_pdf(pdf_bm, cat_map)
        total_updated += u
        total_unmatched += m

    # Backup + write
    bak = BOOKMARKS_PATH.with_suffix(".json.bak")
    bak.write_text(json.dumps(bookmarks, indent=2), encoding="utf-8")  # backup after enrichment? nope:
    # Actually backup original first: reload and write to bak
    # We'll do it correctly: write original to bak, then enriched to bookmarks.json
    # (if you want: just rename the .bak manually)

    # To ensure bak is the ORIGINAL, rewrite now:
    original = _load_json(BOOKMARKS_PATH)
    bak.write_text(json.dumps(original, indent=2), encoding="utf-8")

    BOOKMARKS_PATH.write_text(json.dumps(bookmarks, indent=2), encoding="utf-8")

    print(f"Backup: {bak}")
    print(f"Wrote:  {BOOKMARKS_PATH}")
    print(f"Updated bookmarks (matched category): {total_updated}")
    print(f"Unmatched bookmarks (no category match): {total_unmatched}")
    print("")
    print("Tip: Unmatched is normal for intro/terms sections. The important ones (Wells, Stratigraphy, etc.) should match.")


if __name__ == "__main__":
    main()
