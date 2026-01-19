from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict


# ----------------------------
# Config (edit these paths)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
PDF_DIR = ROOT / "docs" / "schema_pdfs"
BOOKMARKS_JSON = PDF_DIR / "bookmarks.json"

PPDM39_REGISTRY = ROOT / "schema_registry" / "catalog" / "ppdm_39_schema_domain.json"
PPDMLITE_REGISTRY = ROOT / "schema_registry" / "catalog" / "ppdm_lite_schema_domain.json"  # adjust if your file name differs


# ----------------------------
# Helpers
# ----------------------------
def _load_json(p: Path) -> object:
    return json.loads(p.read_text(encoding="utf-8"))


def _detect_root_key(d: dict) -> str | None:
    """
    Finds the first key whose value is a non-empty list of dict rows with table_name/column_name.
    """
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            keys = {x.lower() for x in v[0].keys()}
            if "table_name" in keys and "column_name" in keys:
                return k
    return None


def _group_tables_by_category(rows: list[dict]) -> dict[str, list[str]]:
    """
    rows are row-based schema registry records.
    returns: category_label -> sorted unique list of "dbo.table"
    """
    by_cat = defaultdict(set)

    for r in rows:
        schema = (r.get("table_schema") or r.get("TABLE_SCHEMA") or "dbo").strip()
        table = (r.get("table_name") or r.get("TABLE_NAME") or "").strip()
        if not table:
            continue

        fqn = f"{schema}.{table}"

        # Category naming: try standard fields first
        cat = (r.get("category") or r.get("CATEGORY") or "").strip()
        sub = (r.get("sub_category") or r.get("SUB_CATEGORY") or r.get("sub-category") or "").strip()

        # Prefer high-level category if present; else use sub_category; else "Unknown"
        label = cat or sub or "Unknown"

        # normalize a bit (PPDM tends to use upper category codes)
        label = label.strip()

        by_cat[label].add(fqn)

    return {k: sorted(v) for k, v in by_cat.items()}


def _match_bookmark_label_to_category(bookmark_label: str, categories: list[str]) -> str | None:
    """
    Heuristic mapping: "Wells" bookmark should match category "WEL" or "Wells" etc.
    We'll do case-insensitive contains / startswith.
    """
    b = bookmark_label.strip().lower()

    # Common PPDM category code mappings you can expand later
    aliases = {
        "wells": ["wel", "wells"],
        "stratigraphy": ["stratigraphy", "str", "strat"],
        "seismic": ["seis", "seismic"],
        "production": ["prod", "production"],
        "areas": ["area", "areas"],
        "business associates": ["ba", "business associates", "business_associate"],
        "sources": ["source", "sources"],
    }

    if b in aliases:
        needles = aliases[b]
    else:
        needles = [b]

    cats_lower = [(c, c.lower()) for c in categories]

    # exact-ish match
    for needle in needles:
        for c, cl in cats_lower:
            if cl == needle:
                return c

    # contains match
    for needle in needles:
        for c, cl in cats_lower:
            if needle in cl or cl in needle:
                return c

    # startswith match
    for needle in needles:
        for c, cl in cats_lower:
            if cl.startswith(needle) or needle.startswith(cl):
                return c

    return None


def _pick_default_table(tables: list[str], prefer: list[str]) -> str:
    """
    Pick best default table to jump to in Data Explorer.
    """
    tset = {t.lower(): t for t in tables}
    for p in prefer:
        if p.lower() in tset:
            return tset[p.lower()]
    # fallback
    return tables[0] if tables else ""


def enrich_bookmarks(bookmarks: dict, pdf_name: str, tables_by_cat: dict[str, list[str]]) -> None:
    """
    Mutates bookmarks[pdf_name] in place:
      - sets tables[] based on matched category
      - sets table to a preferred default if blank
    """
    if pdf_name not in bookmarks or not isinstance(bookmarks[pdf_name], dict):
        return

    categories = list(tables_by_cat.keys())

    for label, item in list(bookmarks[pdf_name].items()):
        if not isinstance(item, dict):
            continue

        # Determine the "category" name for matching
        cat = (item.get("category") or label).strip()
        match = _match_bookmark_label_to_category(cat, categories)

        if match:
            tables = tables_by_cat.get(match, [])
        else:
            tables = []

        # Write tables list
        item["tables"] = tables

        # Choose a default table if missing
        if not (item.get("table") or "").strip() and tables:
            item["table"] = _pick_default_table(
                tables,
                prefer=[
                    "dbo.well",
                    "dbo.strat_unit",
                    "dbo.well_dir_srvy",
                    "dbo.well_log",
                    "dbo.prod_summary",
                    "dbo.seis_set",
                ],
            )

        bookmarks[pdf_name][label] = item


def main() -> None:
    if not BOOKMARKS_JSON.exists():
        raise SystemExit(f"Missing bookmarks.json at: {BOOKMARKS_JSON}")

    bookmarks = _load_json(BOOKMARKS_JSON)
    if not isinstance(bookmarks, dict):
        raise SystemExit("bookmarks.json must be a JSON object")

    # Load registries if present
    if PPDM39_REGISTRY.exists():
        reg39 = _load_json(PPDM39_REGISTRY)
        if isinstance(reg39, dict):
            k = _detect_root_key(reg39)
            if k:
                rows = reg39.get(k, [])
                if isinstance(rows, list):
                    tables_by_cat = _group_tables_by_category(rows)
                    enrich_bookmarks(bookmarks, "PPDM_Data_Model_3.9_Diagrams.pdf", tables_by_cat)

    if PPDMLITE_REGISTRY.exists():
        regl = _load_json(PPDMLITE_REGISTRY)
        if isinstance(regl, dict):
            k = _detect_root_key(regl)
            if k:
                rows = regl.get(k, [])
                if isinstance(rows, list):
                    tables_by_cat = _group_tables_by_category(rows)
                    enrich_bookmarks(bookmarks, "PPDM_Lite11.pdf", tables_by_cat)

    BOOKMARKS_JSON.write_text(json.dumps(bookmarks, indent=2), encoding="utf-8")
    print(f"Wrote: {BOOKMARKS_JSON}")


if __name__ == "__main__":
    main()
