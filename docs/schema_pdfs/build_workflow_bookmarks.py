import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

ROOT = Path(r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3")

BOOKMARKS_PATH = ROOT / "docs" / "schema_pdfs" / "bookmarks.json"
OUT_PATH = ROOT / "docs" / "schema_pdfs" / "bookmarks.json"         # overwrite in place
OUT_BAK  = ROOT / "docs" / "schema_pdfs" / "bookmarks.json.bak"

PPDM39_SCHEMA_PATH = ROOT / "schema_registry" / "ppdm_39_schema_domain.json"
LITE_SCHEMA_PATH  = ROOT / "schema_registry" / "ppdm_lite_schema_domain.json"


# --------------------------
# JSON loading
# --------------------------
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _extract_rows(schema_obj: Any) -> List[dict]:
    if isinstance(schema_obj, list):
        return schema_obj
    if isinstance(schema_obj, dict):
        for k in ("ppdm_39_schema_domain", "ppdm_lite_schema_domain", "rows"):
            v = schema_obj.get(k)
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
        # fallback: first list value
        for v in schema_obj.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    raise ValueError("Schema JSON does not contain a recognizable rows list.")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _fqn(schema: str, table: str) -> str:
    schema = (schema or "").strip() or "dbo"
    table  = (table or "").strip()
    return f"{schema}.{table}" if table else ""


# --------------------------
# Build table inventory
# --------------------------
def build_table_inventory(rows: List[dict]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
      all_tables: set of schema.table
      cat_to_tables: category -> set(schema.table)
    """
    all_tables: Set[str] = set()
    cat_to_tables: Dict[str, Set[str]] = {}

    for r in rows:
        cat = (r.get("category") or r.get("Category") or "").strip()
        sch = (r.get("table_schema") or r.get("TABLE_SCHEMA") or "").strip()
        tbl = (r.get("table_name") or r.get("TABLE_NAME") or "").strip()
        if not tbl:
            continue

        fqn = _fqn(sch, tbl)
        all_tables.add(fqn)

        if cat:
            k = _norm(cat)
            cat_to_tables.setdefault(k, set()).add(fqn)

    return all_tables, cat_to_tables


# --------------------------
# Workflow bucket rules
# --------------------------
def pick_tables_by_keywords(all_tables: Set[str], keywords: List[str]) -> List[str]:
    """
    keywords match against table name only (not schema), case-insensitive
    """
    keys = [k.lower() for k in keywords]
    out = []
    for fqn in all_tables:
        tname = fqn.split(".", 1)[1].lower()
        if any(k in tname for k in keys):
            out.append(fqn)
    return sorted(set(out), key=lambda x: x.lower())

def workflow_buckets_ppdm39(all_tables: Set[str]) -> Dict[str, List[str]]:
    """
    PPDM 3.9 tends to use well_* / strat_* / seis_* / prod_* patterns.
    Adjust anytime.
    """
    return {
        "Wells — Header": pick_tables_by_keywords(all_tables, [
            "well", "pden_well", "well_alias", "well_status", "well_version",
            "well_area", "well_node", "well_license", "well_xref",
        ]),
        "Wells — Directional Surveys": pick_tables_by_keywords(all_tables, [
            "well_dir_srvy", "dir_srvy", "survey", "srvy_station",
        ]),
        "Wells — Tops / Strat": pick_tables_by_keywords(all_tables, [
            "strat", "strat_unit", "well_strat", "well_node_strat",
            "well_zone", "zone", "top", "marker",
        ]),
        "Wells — Checkshots": pick_tables_by_keywords(all_tables, [
            "checkshot", "time_depth", "well_time", "vel", "sonic", "td_curve",
        ]),
        "Wells — Logs": pick_tables_by_keywords(all_tables, [
            "well_log", "log", "curve",
        ]),
        "Seismic": pick_tables_by_keywords(all_tables, [
            "seis", "seismic",
        ]),
        "Production": pick_tables_by_keywords(all_tables, [
            "prod", "production",
        ]),
        "Other": [],  # we fill later as remainder
    }

def workflow_buckets_lite(all_tables: Set[str]) -> Dict[str, List[str]]:
    """
    PPDM Lite uses L_* naming (L_WELL, L_SEIS_SET, etc.)
    """
    return {
        "Wells — Header": pick_tables_by_keywords(all_tables, [
            "l_well", "well",
        ]),
        "Wells — Directional Surveys": pick_tables_by_keywords(all_tables, [
            "dir", "survey", "srvy",
        ]),
        "Wells — Tops / Strat": pick_tables_by_keywords(all_tables, [
            "strat", "unit", "top",
        ]),
        "Wells — Checkshots": pick_tables_by_keywords(all_tables, [
            "checkshot", "time_depth", "vel",
        ]),
        "Wells — Logs": pick_tables_by_keywords(all_tables, [
            "log", "curve",
        ]),
        "Seismic": pick_tables_by_keywords(all_tables, [
            "seis", "seismic", "l_seis",
        ]),
        "Production": pick_tables_by_keywords(all_tables, [
            "prod", "production", "l_prod",
        ]),
        "Other": [],
    }


def fill_other_bucket(buckets: Dict[str, List[str]], all_tables: Set[str]) -> None:
    used = set()
    for k, arr in buckets.items():
        used |= set(arr)
    other = sorted(all_tables - used, key=lambda x: x.lower())
    buckets["Other"] = other


# --------------------------
# Inject "Workflow" section into bookmarks.json
# --------------------------
def ensure_bookmarks_shape(bookmarks: Any) -> Dict[str, dict]:
    if not isinstance(bookmarks, dict):
        return {}
    # each pdf key must map to dict
    out = {}
    for pdf, bm in bookmarks.items():
        out[pdf] = bm if isinstance(bm, dict) else {}
    return out

def upsert_workflow_section(pdf_bm: Dict[str, dict], buckets: Dict[str, List[str]]) -> None:
    """
    Adds a "Workflow" bookmark entry with nested children.
    We keep it simple: top-level key "Workflow" points to an object
    the UI can treat like any other bookmark (page blank) but can render sublists.
    """
    workflow = {
        "page": 1,
        "category": "Workflow",
        "table": "",
        "tables": [],
        "children": {},
    }

    for label, tables in buckets.items():
        # choose a default table for Data Explorer linking
        default_table = ""
        if tables:
            # prefer dbo.well / dbo.well_dir_srvy / dbo.well_log etc when present
            preferred = [
                "dbo.well", "dbo.well_dir_srvy", "dbo.well_dir_srvy_station",
                "dbo.well_strat_unit", "dbo.strat_unit", "dbo.well_log",
                "dbo.l_well",
            ]
            for p in preferred:
                if p in tables:
                    default_table = p
                    break
            if not default_table:
                default_table = tables[0]

        workflow["children"][label] = {
            "page": 1,
            "category": label,
            "table": default_table,
            "tables": tables,
        }

    pdf_bm["Workflow"] = workflow


def main() -> None:
    # load existing bookmarks (if any)
    bookmarks = {}
    if BOOKMARKS_PATH.exists():
        bookmarks = ensure_bookmarks_shape(_load_json(BOOKMARKS_PATH))

    # load schema rows
    ppdm39_rows = _extract_rows(_load_json(PPDM39_SCHEMA_PATH))
    lite_rows   = _extract_rows(_load_json(LITE_SCHEMA_PATH))

    ppdm39_tables, _ = build_table_inventory(ppdm39_rows)
    lite_tables, _   = build_table_inventory(lite_rows)

    # build workflow buckets
    ppdm39_buckets = workflow_buckets_ppdm39(ppdm39_tables)
    fill_other_bucket(ppdm39_buckets, ppdm39_tables)

    lite_buckets = workflow_buckets_lite(lite_tables)
    fill_other_bucket(lite_buckets, lite_tables)

    # upsert workflow section for each pdf in bookmarks
    for pdf_name, pdf_bm in bookmarks.items():
        name_l = (pdf_name or "").lower()
        if "lite" in name_l:
            upsert_workflow_section(pdf_bm, lite_buckets)
        else:
            upsert_workflow_section(pdf_bm, ppdm39_buckets)

    # backup + write
    if BOOKMARKS_PATH.exists():
        OUT_BAK.write_text(BOOKMARKS_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    OUT_PATH.write_text(json.dumps(bookmarks, indent=2), encoding="utf-8")

    print("Wrote:", OUT_PATH)
    if OUT_BAK.exists():
        print("Backup:", OUT_BAK)
    print("")
    print("Next: update the ERD Viewer to display Workflow children (1 small UI tweak).")


if __name__ == "__main__":
    main()
