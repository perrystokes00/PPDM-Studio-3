# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ppdm_loader.db import connect, connect_master, read_sql
import ppdm_loader.db as db  # exec_sql + exec_view_ddl

from ppdm_loader.introspect import (
    list_databases,
    fetch_columns,
    fetch_fk_details,
    autodetect_parent_for_child_col,
)

from ppdm_loader.stage import save_upload, stage_bulk_insert, DELIM_MAP
from ppdm_loader.discover import discover_top_tables
from ppdm_loader.normalize import build_primary_norm_view_sql
from ppdm_loader.rules import (
    ensure_rules_tables,
    upsert_rules_from_json,
    apply_rules,
    RULES_TABLE,
)
from ppdm_loader.qc import qc_raw_vs_norm
from ppdm_loader.schema_catalog import SchemaCatalog
from ppdm_loader.seed import seed_parent_tables_from_view
from ppdm_loader.promote import (
    build_promote_plan,
    run_promote,
    read_promote_qc,
    promote_qc_adjacent,
    PROMOTE_QC_TABLE,
)
from dataclasses import dataclass
from typing import Any

# Synonyms
from ppdm_loader.synonyms_simple import apply_synonyms, save_mappings_as_synonyms


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="PPDM Loader Studio – Scaffold", layout="wide")

BULK_ROOT = Path(r"C:\Bulk\uploads")
BULK_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_RULES = Path(
    r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\rules\rules_catalog.json"
)

PPDM39_CATALOG_PATH = Path(
    r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\schema_registry\ppdm_39_schema_domain.json"
)
PPDM39_CATALOG_KEY = "ppdm_39_schema_domain"

PPDMLITE_CATALOG_PATH = Path(
    r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\schema_registry\ppdm_lite_schema_domain.json"
)
PPDMLITE_CATALOG_KEY = "ppdm_lite_schema_domain"

PPDMLITE_FK_OVERLAY_PATH = Path(
    r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\schema_registry\ppdm_lite11_fk_from_pdfplumber.json"
)
PPDMLITE_FK_OVERLAY_DEFAULT_SCHEMA = "dbo"

# ✅ YOUR catalog path (update to what you actually have)
SEED_CATALOG_PPDM39 = Path(
    r"C:\Users\perry\OneDrive\Documents\ppdm39-seed-catalog\seeds\catalog\ppdm39_seed_catalog.json"
)

_AUDIT_COLS = {
    "RID",
    "PPDM_GUID",
    "ROW_CREATED_BY", "ROW_CREATED_DATE",
    "ROW_CHANGED_BY", "ROW_CHANGED_DATE",
    "ROW_EFFECTIVE_DATE", "ROW_EXPIRY_DATE",
    "EFFECTIVE_DATE", "EXPIRY_DATE",
    "ROW_QUALITY",
    "REMARK",
}

# -----------------------------
# Small helpers
# -----------------------------
def _clear_step5_outputs() -> None:
    for k in ("seed_report_df", "invalid_rows_df", "qc_raw_norm_df"):
        st.session_state[k] = None


def _clear_promote_outputs() -> None:
    for k in ("promote_plan", "promote_qc_summary_df", "promote_qc_adj_df"):
        st.session_state[k] = None


def _sql_lit(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v).replace("'", "''")
    return f"N'{s}'"


def _read_csv_rows(csv_path: Path) -> list[dict]:
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        quotechar='"',
        skipinitialspace=True,
    )

    # ✅ Normalize headers (prevents "ROW_QUALITY_ID " vs "ROW_QUALITY_ID" mismatches)
    df.columns = [str(c).strip().upper() for c in df.columns]

    # ✅ Clean values: trim whitespace + remove wrapping quotes
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')

    # Convert to list of dicts, drop fully-empty rows
    rows = df.to_dict(orient="records")
    out: list[dict] = []
    for r in rows:
        rr = {k: (v if v != "" else None) for k, v in r.items()}
        if any(v is not None for v in rr.values()):
            out.append(rr)
    return out


def _split_fqn(fqn: str, default_schema: str = "dbo") -> tuple[str, str]:
    fqn = (fqn or "").strip()
    if "." in fqn:
        s, t = fqn.split(".", 1)
        return s.strip() or default_schema, t.strip()
    return default_schema, fqn.strip()

def _quote_ident(name: str) -> str:
    # SQL Server safe identifier quoting
    name = (name or "").replace("]", "]]")
    return f"[{name}]"

def _fqn_quoted(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"

def _fetch_pk_columns(conn, schema: str, table: str) -> list[str]:
    """
    Tight PK resolution (supports composite keys).
    Returns PK columns ordered by key ordinal.
    """
    sql = r"""
    SELECT c.name AS pk_column
    FROM sys.key_constraints kc
    JOIN sys.indexes i
      ON i.object_id = kc.parent_object_id
     AND i.index_id  = kc.unique_index_id
    JOIN sys.index_columns ic
      ON ic.object_id = i.object_id
     AND ic.index_id  = i.index_id
    JOIN sys.columns c
      ON c.object_id = ic.object_id
     AND c.column_id = ic.column_id
    JOIN sys.tables t
      ON t.object_id = kc.parent_object_id
    JOIN sys.schemas s
      ON s.schema_id = t.schema_id
    WHERE kc.type = 'PK'
      AND s.name = ?
      AND t.name = ?
    ORDER BY ic.key_ordinal;
    """
    df = read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return [str(x).strip() for x in df["pk_column"].tolist() if str(x).strip()]

def _table_columns_set(conn, schema: str, table: str) -> set[str]:
    cols = fetch_columns(conn, schema, table)
    if cols is None or cols.empty:
        return set()
    return {str(x).strip().upper() for x in cols["column_name"].tolist()}

def _sql_lit(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v).replace("'", "''")
    return f"N'{s}'"

def _build_missing_distinct_sql(
    *,
    view_name: str,
    source_cols: list[str],          # columns in view to use
    target_schema: str,
    target_table: str,
    target_key_cols: list[str],      # PK columns in target table
    defaults: dict[str, Any] | None = None,
    map_target_long_name_from: str | None = None,  # e.g. "CONFIDENTIAL_TYPE"
    top_preview: int = 200,
) -> tuple[str, str]:
    """
    Returns (preview_sql, seed_sql)

    preview_sql: shows missing distinct key tuples
    seed_sql: inserts missing tuples (no updates) and sets defaults where possible
    """

    defaults = defaults or {}

    tgt_fqn = _fqn_quoted(target_schema, target_table)

    # Build normalized expressions for source columns
    # Treat blank strings as NULL
    src_exprs = []
    for c in source_cols:
        qc = _quote_ident(c)
        src_exprs.append(f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), src.{qc}))), N'') AS {qc}")

    src_select_list = ",\n        ".join(src_exprs)

    # Build join/except columns (key tuple)
    key_select = ", ".join([f"{_quote_ident(c)}" for c in source_cols])
    tgt_key_select = ", ".join([f"t.{_quote_ident(c)}" for c in target_key_cols])

    # Preview missing tuples
    preview_sql = f"""
;WITH src_distinct AS (
    SELECT DISTINCT
        {src_select_list}
    FROM {view_name} src
),
missing AS (
    SELECT {key_select}
    FROM src_distinct
    WHERE {" AND ".join([f"{_quote_ident(c)} IS NOT NULL" for c in source_cols])}
    EXCEPT
    SELECT {tgt_key_select}
    FROM {tgt_fqn} t
)
SELECT TOP ({int(top_preview)}) *
FROM missing
ORDER BY 1;
""".strip()

    # Build insert column list + select list (keys + defaults)
    # We only insert columns that exist in target table and that we can supply.
    target_cols_upper = _table_columns_set(conn, target_schema, target_table)
    insert_cols: list[str] = []
    select_cols: list[str] = []

    # keys
    for i, tgt_key in enumerate(target_key_cols):
        src_key = source_cols[i]
        if tgt_key.upper() in target_cols_upper:
            insert_cols.append(_quote_ident(tgt_key))
            select_cols.append(f"m.{_quote_ident(src_key)}")

    # defaults if column exists
    for col, val in defaults.items():
        if str(col).upper() in target_cols_upper and str(col).upper() not in {c.upper() for c in target_key_cols}:
            insert_cols.append(_quote_ident(col))
            select_cols.append(_sql_lit(val))

    # optional LONG_NAME mapping from code (common in r_* tables)
    if map_target_long_name_from:
        if "LONG_NAME" in target_cols_upper and "LONG_NAME" not in {c.upper() for c in target_key_cols}:
            insert_cols.append("[LONG_NAME]")
            select_cols.append(f"m.{_quote_ident(map_target_long_name_from)}")

    # PPDM_GUID fill if exists and not being provided
    if "PPDM_GUID" in target_cols_upper and "PPDM_GUID" not in {c.upper() for c in insert_cols}:
        insert_cols.append("[PPDM_GUID]")
        select_cols.append("CONVERT(nvarchar(36), NEWID())")

    if not insert_cols:
        # nothing to insert -> return preview only
        seed_sql = "/* No insertable columns were resolved for this rule. */"
        return preview_sql, seed_sql

    insert_col_list = ", ".join(insert_cols)
    select_col_list = ", ".join(select_cols)

    seed_sql = f"""
;WITH src_distinct AS (
    SELECT DISTINCT
        {src_select_list}
    FROM {view_name} src
),
missing AS (
    SELECT {key_select}
    FROM src_distinct
    WHERE {" AND ".join([f"{_quote_ident(c)} IS NOT NULL" for c in source_cols])}
    EXCEPT
    SELECT {tgt_key_select}
    FROM {tgt_fqn} t
)
INSERT INTO {tgt_fqn} ({insert_col_list})
SELECT {select_col_list}
FROM missing m;
""".strip()

    return preview_sql, seed_sql


def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", (s or "").upper()).strip("_")

def _table_tokens(table: str) -> list[str]:
    # r_confidential_type -> ["CONFIDENTIAL", "TYPE"]
    return [t for t in _norm_name(table).split("_") if t]

def _ppdm_heuristic_pk_candidates(schema: str, table: str, col_names: list[str]) -> list[str]:
    """
    Heuristic ranking for PPDM-like reference tables when no PK constraint exists.
    """
    cols_u = [c.upper() for c in col_names]
    cols_set = set(cols_u)

    t = table.lower()
    toks = _table_tokens(table)  # e.g. ["CONFIDENTIAL", "TYPE"]
    last = toks[-1] if toks else ""

    ranked = []

    # 1) Exact "TABLE_ID" pattern
    exact_id = _norm_name(table) + "_ID"
    if exact_id in cols_set:
        ranked.append(exact_id)

    # 2) For r_* tables, try "<X>_TYPE" / "<X>" patterns
    if t.startswith("r_"):
        # r_confidential_type -> CONFIDENTIAL_TYPE
        if len(toks) >= 2 and toks[-1] == "TYPE":
            cand = "_".join(toks[:-1]) + "_TYPE"
            if cand in cols_set:
                ranked.append(cand)

        # r_well_class -> WELL_CLASS
        cand2 = "_".join(toks)
        if cand2 in cols_set:
            ranked.append(cand2)

        # Common PPDM code-ish columns
        for c in ("CODE", "TYPE", "STATUS"):
            # e.g. STATUS or WELL_STATUS etc.
            if c in cols_set:
                ranked.append(c)

    # 3) If table name ends with *_type, prefer any column ending in _TYPE
    if last == "TYPE":
        ends = [c for c in cols_u if c.endswith("_TYPE") and c not in _AUDIT_COLS]
        ranked.extend(ends)

    # 4) Prefer columns ending in _ID
    ends_id = [c for c in cols_u if c.endswith("_ID") and c not in _AUDIT_COLS]
    ranked.extend(ends_id)

    # 5) Prefer small “name-like” keys for reference tables
    for c in ("NAME", "LONG_NAME", "SHORT_NAME"):
        if c in cols_set:
            ranked.append(c)

    # Dedup preserving order
    out = []
    seen = set()
    for c in ranked:
        if c and c not in seen and c in cols_set:
            out.append(c)
            seen.add(c)
    return out


def resolve_pk_columns(conn, schema: str, table: str) -> list[str]:
    """
    Returns PK columns (in key order) if PK exists, else best-effort unique key.
    If neither exists, returns [].
    """
    schema = (schema or "dbo").strip()
    table = (table or "").strip()
    if not table:
        return []

    # 1) True PK constraint columns (best)
    pk_sql = """
    SET NOCOUNT ON;
    SELECT kcu.COLUMN_NAME, kcu.ORDINAL_POSITION
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
      ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
     AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
     AND tc.TABLE_NAME = kcu.TABLE_NAME
    WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
      AND tc.TABLE_SCHEMA = ?
      AND tc.TABLE_NAME = ?
    ORDER BY kcu.ORDINAL_POSITION;
    """
    try:
        df = read_sql(conn, pk_sql, params=[schema, table])
        if df is not None and not df.empty:
            return [str(x).strip() for x in df["COLUMN_NAME"].tolist() if str(x).strip()]
    except Exception:
        pass

    # 2) Unique index columns (fallback if no PK)
    uniq_sql = """
    SET NOCOUNT ON;
    SELECT c.name AS COLUMN_NAME, ic.key_ordinal AS ORDINAL_POSITION
    FROM sys.schemas s
    JOIN sys.tables  t ON t.schema_id = s.schema_id
    JOIN sys.indexes i ON i.object_id = t.object_id
    JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id
    JOIN sys.columns c ON c.object_id = t.object_id AND c.column_id = ic.column_id
    WHERE s.name = ?
      AND t.name = ?
      AND i.is_unique = 1
      AND i.is_primary_key = 0
      AND ic.is_included_column = 0
    ORDER BY i.index_id, ic.key_ordinal;
    """
    try:
        df = read_sql(conn, uniq_sql, params=[schema, table])
        if df is not None and not df.empty:
            # Take the first unique index found (lowest index_id)
            # group by index order by it, but simplest: sys already ordered by index_id
            cols = [str(x).strip() for x in df["COLUMN_NAME"].tolist() if str(x).strip()]
            if cols:
                return cols
    except Exception:
        pass

    return []


def resolve_best_single_pk(conn, schema: str, table: str) -> str | None:
    """
    Prefer true PK (single-column). If PK is composite, return None (force user choice).
    If no PK/unique, try PPDM heuristic to guess a single-column key.
    """
    cols_df = fetch_columns(conn, schema, table)
    if cols_df is None or cols_df.empty:
        return None

    all_cols = [str(c).strip() for c in cols_df["column_name"].tolist()]

    pk_cols = resolve_pk_columns(conn, schema, table)
    if len(pk_cols) == 1:
        return pk_cols[0]
    if len(pk_cols) > 1:
        # composite PK: distinct seeding needs explicit mapping strategy; don't guess
        return None

    # No PK/unique => heuristic
    cands = _ppdm_heuristic_pk_candidates(schema, table, all_cols)
    return cands[0] if cands else None


# -----------------------------
# Schema Catalog loader
# -----------------------------
def load_catalog(version_label: str) -> None:
    if version_label == "PPDM 3.9":
        st.session_state["catalog"] = SchemaCatalog.load(
            str(PPDM39_CATALOG_PATH), root_key=PPDM39_CATALOG_KEY
        )
    else:
        fk_overlay = str(PPDMLITE_FK_OVERLAY_PATH) if PPDMLITE_FK_OVERLAY_PATH.exists() else None
        st.session_state["catalog"] = SchemaCatalog.load(
            str(PPDMLITE_CATALOG_PATH),
            root_key=PPDMLITE_CATALOG_KEY,
            fk_overlay_path=fk_overlay,
            fk_overlay_default_schema=PPDMLITE_FK_OVERLAY_DEFAULT_SCHEMA,
        )
    st.session_state["catalog_version"] = version_label
    st.success(f"{version_label} catalog loaded")


def fast_discover_tables(
    conn, source_cols: list[str], schema_filter: str | None, table_prefix: str | None
) -> pd.DataFrame:
    cat: SchemaCatalog | None = st.session_state.get("catalog")
    domain = st.session_state.get("domain_filter")

    if cat is not None:
        rows = cat.discover_tables(
            source_cols=source_cols,
            category=domain,
            schema_filter=schema_filter,
            table_prefix=table_prefix,
            top_n=10,
        )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns={"schema": "schema_name", "table": "table_name", "matches": "score"})
        return df[["schema_name", "table_name", "score"]]

    # fallback
    return discover_top_tables(conn, source_cols, schema_filter, table_prefix, top_n=10)


# -----------------------------
# FK map builder (real FK + pseudo Treat-as-FK)
# -----------------------------
def build_fk_map(conn, primary_schema: str, primary_table: str, treat_fk_cols: list[str]) -> dict:
    fk_map: dict[str, tuple[str, str, str]] = {}

    cat = st.session_state.get("catalog") or st.session_state.get("schema_catalog") or st.session_state.get("cat")
    fk_df = fetch_fk_details(conn, primary_schema, primary_table)

    def _get(r, *keys, default=None):
        for k in keys:
            if k in r and r[k] is not None and str(r[k]).strip() != "":
                return r[k]
        return default

    # 1) Real FK metadata
    if fk_df is not None and not fk_df.empty:
        for _, r in fk_df.iterrows():
            rr = r.to_dict()
            child = str(_get(rr, "child_column", "COLUMN_NAME", "column_name", default="") or "").strip()
            ps = str(_get(rr, "parent_schema", "FK_TABLE_SCHEMA", "fk_table_schema", default="dbo") or "dbo").strip()
            pt = str(_get(rr, "parent_table", "FK_TABLE_NAME", "fk_table_name", default="") or "").strip()
            pc = str(_get(rr, "parent_column", "FK_COLUMN_NAME", "fk_column_name", default="") or "").strip()
            if child and pt and pc:
                fk_map.setdefault(child, (ps, pt, pc))

    def _normalize_hit(hit):
        if hit is None:
            return None
        if isinstance(hit, dict):
            ps = hit.get("parent_schema")
            pt = hit.get("parent_table")
            pc = hit.get("parent_column")
            if ps and pt and pc:
                return (ps, pt, pc)
        if isinstance(hit, (tuple, list)) and len(hit) >= 3:
            ps, pt, pc = hit[-3], hit[-2], hit[-1]
            if ps and pt and pc:
                return (ps, pt, pc)
        return None

    # 2) Pseudo-FK from "Treat as FK"
    for child_col in (treat_fk_cols or []):
        child_col = str(child_col or "").strip()
        if not child_col:
            continue
        if child_col in fk_map:
            continue

        hit = autodetect_parent_for_child_col(fk_df, child_col) if fk_df is not None else None
        norm = _normalize_hit(hit)
        if norm:
            fk_map[child_col] = norm
            continue

        if cat:
            try:
                meta = cat.resolve_fk_parent(primary_schema, primary_table, child_col)
                if meta:
                    fk_map[child_col] = meta
            except Exception:
                pass

    return fk_map


# -----------------------------
# Seed catalog helpers (FQN-keyed JSON)
# -----------------------------
def _load_seed_catalog_safe(path: Path) -> dict:
    """
    Loads FQN-keyed seed catalog JSON.

    Expected format:
      {
        "root": "C:\\...\\ppdm39-seed-catalog",
        "_meta": {... optional ...},
        "dbo.r_confidential_type": { "mode": "static", "pk": {"column": "..."}, "file": "seeds/ppdm39/CSV/r/....csv" },
        ...
      }

    Returns {} on failure.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _catalog_get_mode(spec: Any) -> str:
    if not isinstance(spec, dict):
        return "static"
    return str(spec.get("mode") or spec.get("seed_mode") or "static").strip().lower()


def _catalog_get_seed_default(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    for k in ("seed_default", "default_seed", "seedByDefault"):
        if k in spec:
            try:
                return bool(spec.get(k))
            except Exception:
                return False
    return False


def _catalog_get_rows(spec: dict) -> list[dict]:
    rows = spec.get("rows") or spec.get("data") or spec.get("static_rows") or []
    return rows if isinstance(rows, list) else []


def _catalog_get_file(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    for k in ("file", "csv", "path", "csv_file"):
        v = spec.get(k)
        if v:
            return str(v).strip()
    return None


def _catalog_get_pk_col(spec: dict) -> str | None:
    pk = spec.get("pk")
    if isinstance(pk, dict):
        c = pk.get("column") or pk.get("col") or pk.get("name")
        return str(c).strip() if c else None

    keys = spec.get("keys")
    if isinstance(keys, list) and keys:
        return str(keys[0]).strip()

    c = spec.get("pk_col") or spec.get("pk_column")
    return str(c).strip() if c else None


def _catalog_root_dir(catalog: dict, catalog_path: Path) -> Path:
    # Prefer top-level root, fallback to _meta.root, fallback to catalog file parent
    root = catalog.get("root")
    if not root:
        root = (catalog.get("_meta") or {}).get("root")
    if root:
        return Path(str(root))
    return catalog_path.parent


def generate_seed_sql_from_catalog(catalog: dict, selected_tables: list[str], catalog_path: Path) -> str:
    """
    Generate a SQL script (MERGE) for STATIC catalog rows.
    Supports:
      - spec["rows"] inline
      - spec["file"] CSV path relative to catalog root
    """
    parts: list[str] = ["SET NOCOUNT ON;", ""]

    base_dir = _catalog_root_dir(catalog, catalog_path)

    for fqn in selected_tables:
        spec = catalog.get(fqn) or catalog.get(fqn.lower())
        if not isinstance(spec, dict):
            continue

        mode = _catalog_get_mode(spec)
        if mode != "static":
            continue

        rows = _catalog_get_rows(spec)

        # If no inline rows, load from CSV file if provided
        if not rows:
            rel = _catalog_get_file(spec)
            if rel:
                csv_path = (base_dir / rel).resolve()
                if csv_path.exists():
                    rows = _read_csv_rows(csv_path)
                else:
                    parts.append(f"-- WARNING: CSV not found for {fqn}: {csv_path}")
                    parts.append("")
                    continue

        if not rows:
            continue

        pk_col = _catalog_get_pk_col(spec)

        parts.append("-- =====================================================")
        parts.append(f"-- Seed: {fqn}")
        parts.append(f"-- Mode: {mode} | Rows: {len(rows)} | PK: {pk_col or '(none)'}")
        parts.append("-- =====================================================")

        for row in rows:
            if not isinstance(row, dict):
                continue
            cols = list(row.keys())
            if not cols:
                continue

            src_select = ", ".join([f"{_sql_lit(row.get(c))} AS [{c}]" for c in cols])
            col_list = ", ".join([f"[{c}]" for c in cols])
            val_list = ", ".join([f"src.[{c}]" for c in cols])

            if pk_col and pk_col in row:
                upd = ", ".join([f"tgt.[{c}] = src.[{c}]" for c in cols if c != pk_col])
                if not upd:
                    upd = f"tgt.[{pk_col}] = tgt.[{pk_col}]"

                parts.append(
                    f"""
MERGE {fqn} AS tgt
USING (SELECT {src_select}) AS src
  ON tgt.[{pk_col}] = src.[{pk_col}]
WHEN MATCHED THEN
  UPDATE SET {upd}
WHEN NOT MATCHED THEN
  INSERT ({col_list}) VALUES ({val_list});
""".strip()
                )
            else:
                parts.append(
                    f"""
INSERT INTO {fqn} ({col_list})
SELECT {", ".join([_sql_lit(row.get(c)) for c in cols])};
""".strip()
                )

        parts.append("")

    return "\n".join(parts).strip() + "\n"
    
def seed_ref_table_from_view_distinct(
    conn,
    view_name: str,
    source_attr: str,
    ref_schema: str,
    ref_table: str,
    pk_col: str,
    defaults: dict | None = None,
) -> int:
    """
    Seeds a reference table by taking DISTINCT values from a NORM view column.

    - Inserts missing PK values only
    - Applies optional defaults if those columns exist in the ref table
    """
    defaults = defaults or {}
    target_fqn = f"[{ref_schema}].[{ref_table}]"

    # Introspect target columns to only use columns that really exist
    cols_df = fetch_columns(conn, ref_schema, ref_table)
    tgt_cols = {str(c).strip().upper() for c in cols_df["column_name"].tolist()}

    insert_cols = [pk_col]
    select_exprs = [f"s.[{pk_col}]"]

    # Optional defaults (only if those columns exist)
    for k, v in defaults.items():
        if k.upper() in tgt_cols:
            insert_cols.append(k)
            select_exprs.append(_sql_lit(v))

    col_list = ", ".join([f"[{c}]" for c in insert_cols])
    sel_list = ", ".join(select_exprs)

    sql = f"""
WITH src AS (
    SELECT DISTINCT
        NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.[{source_attr}]))), N'') AS [{pk_col}]
    FROM {view_name} v
    WHERE v.[{source_attr}] IS NOT NULL
),
src2 AS (
    SELECT [{pk_col}] FROM src WHERE [{pk_col}] IS NOT NULL
)
MERGE {target_fqn} AS t
USING src2 AS s
ON t.[{pk_col}] = s.[{pk_col}]
WHEN NOT MATCHED BY TARGET THEN
    INSERT ({col_list}) VALUES ({sel_list})
;
""".strip()

    db.exec_sql(conn, sql)

    # return count-ish (optional): just report number of distinct candidates
    df = read_sql(conn, f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT [{source_attr}] x FROM {view_name}) d;")
    return int(df.iloc[0]["n"]) if df is not None and not df.empty else 0


def exec_sql_batch(conn, sql_text: str) -> None:
    """
    Execute generated SQL script. Generated script is simple MERGE/INSERT statements,
    so splitting on ';' is OK here.
    """
    sql_text = (sql_text or "").strip()
    if not sql_text:
        return
    stmts = [s.strip() for s in sql_text.split(";") if s.strip()]
    for s in stmts:
        db.exec_sql(conn, s + ";")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Connect")
    server = st.text_input("Server", "PERRY\\SQLEXPRESS", key="server")

    if st.button("Refresh DB list", key="btn_refresh_dbs"):
        cm = connect_master(server)
        st.session_state["db_list"] = list_databases(cm)
        cm.close()

    dbs = st.session_state.get("db_list", [])
    if dbs:
        database = st.selectbox("Database", dbs, index=0, key="database")
        if st.button("Connect", key="btn_connect"):
            st.session_state["conn"] = connect(server, database)
            st.success(f"Connected: {database}")

    st.divider()
    page = st.radio("Page", ["ETL", "Rules Admin"], index=0, key="page")
    st.divider()

    st.checkbox("Run validation", value=True, key="run_validation")
    st.checkbox("Show SQL", value=True, key="show_sql")

    st.divider()
    st.subheader("Schema catalog")
    catalog_choice = st.radio("PPDM Version", ["PPDM 3.9", "PPDM Lite"], index=0, key="catalog_choice")
    st.session_state["ppdm_model"] = catalog_choice

    if st.button("Load schema catalog", key="btn_load_catalog"):
        load_catalog(catalog_choice)

    domain_val = st.session_state.get("domain_filter") or ""
    st.session_state["domain_filter"] = (
        st.text_input(
            "Domain filter (optional)",
            value=domain_val,
            help="Example: WELL, ANL, ZONE. Blank = no domain filtering.",
            key="domain_filter_input",
        ).strip()
        or None
    )

    # ---------------- Seed Catalog ----------------
    st.divider()
    st.subheader("Seed Catalog")
    st.caption("Generate MERGE SQL from seed catalog and populate STATIC reference tables.")

    conn_sidebar = st.session_state.get("conn")
    catalog_exists = SEED_CATALOG_PPDM39.exists()
    seed_disabled = (conn_sidebar is None) or (not catalog_exists)

    seed_clicked = st.button(
        "Seed reference tables now (catalog → SQL → DB)",
        key="btn_seed_catalog_sidebar",
        disabled=seed_disabled,
    )

    if conn_sidebar is None:
        st.info("Connect to a database to enable seeding.")
    if not catalog_exists:
        st.error(f"Seed catalog not found:\n{SEED_CATALOG_PPDM39}")

    if seed_clicked:
        cat = _load_seed_catalog_safe(SEED_CATALOG_PPDM39)

        if not isinstance(cat, dict) or not cat:
            st.error("Seed catalog is empty or invalid JSON.")
            st.stop()

        seed_default_tables: list[str] = []
        static_tables_all: list[str] = []

        for fqn, spec in cat.items():
            if fqn in ("_meta", "defaults", "root"):
                continue
            if not isinstance(fqn, str) or "." not in fqn:
                continue
            if not isinstance(spec, dict):
                continue

            if _catalog_get_mode(spec) != "static":
                continue

            static_tables_all.append(fqn)
            if _catalog_get_seed_default(spec):
                seed_default_tables.append(fqn)

        selected = seed_default_tables if seed_default_tables else static_tables_all
        selected = sorted(list(dict.fromkeys(selected)))

        sql_text = generate_seed_sql_from_catalog(cat, selected, catalog_path=SEED_CATALOG_PPDM39)

        st.session_state["seed_catalog_sql"] = sql_text
        st.session_state["seeded_catalog_tables"] = selected
        st.session_state["seed_catalog_ran"] = False

        if not selected:
            st.error("No STATIC table entries found in catalog.")
        elif not sql_text.strip():
            st.error("Generated seed SQL is empty. (STATIC tables found, but no rows/files to seed.)")
        else:
            try:
                exec_sql_batch(conn_sidebar, sql_text)
                st.session_state["seed_catalog_ran"] = True
                st.success(f"Seeded {len(selected)} table(s).")
            except Exception as e:
                st.session_state["seed_catalog_ran"] = False
                st.error(f"Catalog seeding failed: {e}")

    if st.session_state.get("seed_catalog_sql"):
        with st.expander("Seed SQL (last generated)", expanded=False):
            st.code(st.session_state["seed_catalog_sql"], language="sql")

    if st.session_state.get("seeded_catalog_tables"):
        st.markdown("**Seeded tables (last selection):**")
        st.dataframe(
            pd.DataFrame({"table": st.session_state["seeded_catalog_tables"]}),
            hide_index=True,
            width="stretch",
        )


# ---------------- Main ----------------
st.title("PPDM Loader Studio — Modular Scaffold")

conn = pyodbc.connect(conn_str, autocommit=True)

if not conn:
    st.info("Connect to begin.")
    st.stop()


# ---------------- Rules Admin ----------------
if page == "Rules Admin":
    st.header(f"Rules Admin ({RULES_TABLE})")
    ensure_rules_tables(conn)

    json_path = st.text_input("Rules JSON path", value=str(DEFAULT_RULES), key="rules_json_path")
    if st.button("Upsert from JSON", key="btn_upsert_rules") and json_path.strip():
        n = upsert_rules_from_json(conn, json_path.strip())
        st.success(f"Upserted {n} rules.")

    df = read_sql(conn, f"SELECT * FROM {RULES_TABLE} ORDER BY domain, phase, rule_id;")
    st.dataframe(df, width="stretch", hide_index=True)
    st.stop()


# ---------------- ETL Flow ----------------
st.header("1) Upload & Stage (Preview first)")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    uploaded = st.file_uploader("ASCII", type=["csv", "txt"], key="uploader")
with c2:
    delim_ui = st.selectbox("Delimiter", list(DELIM_MAP.keys()), index=0, key="delim_ui")
with c3:
    has_header = st.checkbox("Has header", True, key="has_header")
with c4:
    rowterm_ui = st.selectbox("RowTerm", ["LF", "CRLF"], index=0, key="rowterm_ui")

if not uploaded:
    st.stop()

delimiter = DELIM_MAP[delim_ui]
rowterm_sql = r"\n" if rowterm_ui == "LF" else r"\r\n"

st.subheader("Preview (first 10 rows)")
try:
    df_preview = pd.read_csv(
        uploaded,
        sep=delimiter,
        nrows=10,
        dtype=str,
        keep_default_na=False,
        engine="python",
        header=0 if has_header else None,
    )
    if not has_header:
        df_preview.columns = [f"COL_{i+1}" for i in range(len(df_preview.columns))]
    st.dataframe(df_preview, width="stretch", hide_index=True)
    st.caption(f"Columns: {len(df_preview.columns)} | Delimiter: {repr(delimiter)} | RowTerm: {rowterm_sql}")
except Exception as e:
    st.error(f"Preview failed with these settings: {e}")
    st.stop()

confirm = st.checkbox("Preview looks correct — proceed to Stage", value=False, key="confirm_stage")

if confirm and st.button("Stage", key="btn_stage", type="primary"):
    _clear_step5_outputs()
    _clear_promote_outputs()
    fp = save_upload(uploaded, BULK_ROOT)
    cols = stage_bulk_insert(conn, fp, delimiter, has_header, rowterm_sql)
    st.session_state["source_cols"] = cols
    st.success("Staged to stg.raw_data")

source_cols = st.session_state.get("source_cols")
if not source_cols:
    st.stop()


st.header("2) Discover (FAST)")
schema_filter = st.text_input("Schema filter", "dbo", key="schema_filter").strip() or None
table_prefix = st.text_input("Table prefix", "", key="table_prefix").strip() or None

if st.button("Run discovery", key="btn_discover"):
    st.session_state["cand"] = fast_discover_tables(conn, source_cols, schema_filter, table_prefix)

cand = st.session_state.get("cand")
if cand is None or cand.empty:
    st.info("Run discovery to continue (or no matches found).")
    st.stop()

st.dataframe(cand, width="stretch", hide_index=True)

choice = st.selectbox(
    "Primary table",
    cand.apply(lambda r: f"{r['schema_name']}.{r['table_name']}", axis=1).tolist(),
    index=0,
    key="primary_choice",
)
primary_schema, primary_table = choice.split(".", 1)


# -----------------------------
# 3) Map primary (WITH SYNONYMS)
# -----------------------------
st.header("3) Map primary (Synonyms enabled)")

cols_df = fetch_columns(conn, primary_schema, primary_table)
pm_key = f"pm::{primary_schema}.{primary_table}"

fk_child_upper = set()
try:
    fk_df = fetch_fk_details(conn, primary_schema, primary_table)
    if fk_df is not None and not fk_df.empty:
        for _, r in fk_df.iterrows():
            c = str(r.get("child_column") or r.get("COLUMN_NAME") or "").strip()
            if c:
                fk_child_upper.add(c.upper())
except Exception:
    fk_child_upper = set()

if pm_key not in st.session_state:
    pm = cols_df.copy()
    pm.insert(
        0,
        "treat_as_fk",
        pm["column_name"].astype(str).str.strip().str.upper().isin(fk_child_upper),
    )
    pm["source_column"] = ""
    pm["constant_value"] = ""

    src_lower = {c.lower(): c for c in source_cols}
    for i, r in pm.iterrows():
        tgt = str(r["column_name"]).strip().lower()
        if tgt in src_lower:
            pm.at[i, "source_column"] = src_lower[tgt]

    pm = apply_synonyms(pm, source_cols, primary_schema, primary_table)

    front = ["treat_as_fk", "column_name", "source_column", "constant_value"]
    rest = [c for c in pm.columns if c not in front]
    st.session_state[pm_key] = pm[front + rest]

with st.form(key=f"form::{pm_key}", clear_on_submit=False):
    edited_pm = st.data_editor(
        st.session_state[pm_key],
        width="stretch",
        hide_index=True,
        column_config={
            "treat_as_fk": st.column_config.CheckboxColumn("Treat as FK", width="small"),
            "column_name": st.column_config.TextColumn("Target", disabled=True, width="medium"),
            "source_column": st.column_config.SelectboxColumn("Source", options=[""] + source_cols, width="medium"),
            "constant_value": st.column_config.TextColumn(
                "Constant",
                help="Literal value applied to all rows for this target column (e.g. ADMIN, STATE, Y).",
                width="medium",
            ),
            "data_type": st.column_config.TextColumn("Type", disabled=True, width="small"),
        },
        key=f"editor::{pm_key}",
    )
    apply_map = st.form_submit_button("Apply mappings", type="primary")

cA, cB = st.columns([1, 1])
with cA:
    if apply_map:
        st.session_state[pm_key] = edited_pm
        _clear_step5_outputs()
        _clear_promote_outputs()
        st.success("Mappings applied (grid will persist).")
with cB:
    if st.button("Save mappings as synonyms", key=f"btn_save_syn::{pm_key}"):
        save_mappings_as_synonyms(st.session_state[pm_key], primary_schema, primary_table)
        st.success("Saved synonyms to synonyms.json")

pm_df = st.session_state[pm_key].copy()
pm_df["source_column"] = pm_df.get("source_column", "").fillna("").astype(str).str.strip()
pm_df["constant_value"] = pm_df.get("constant_value", "").fillna("").astype(str).str.strip()
st.session_state[pm_key] = pm_df

mapped_df = pm_df.loc[(pm_df["source_column"] != "") | (pm_df["constant_value"] != "")].copy()
if mapped_df.empty:
    st.warning("No mapped columns yet. Map at least the required fields before continuing.")
    st.stop()

mapped_targets = mapped_df["column_name"].astype(str).tolist()

treat_fk_cols = (
    pm_df.loc[
        (pm_df["treat_as_fk"] == True)
        & ((pm_df["source_column"] != "") | (pm_df["constant_value"] != "")),
        "column_name",
    ]
    .astype(str)
    .tolist()
)


# -----------------------------
# 4) Remaining FK/Reference tables (FILTERED to mapped attributes)
#     + Match/Map style grid when extra info needed
# -----------------------------
st.header("4) Remaining FK / Reference tables (filtered to mapped attributes)")

# 1) FK parents restricted to mapped FK columns only
fk_map_all = build_fk_map(conn, primary_schema, primary_table, treat_fk_cols)
treat_fk_set = set([c.strip() for c in (treat_fk_cols or [])])
fk_map = {c: meta for (c, meta) in (fk_map_all or {}).items() if c in treat_fk_set}

st.session_state["fk_map"] = fk_map
st.session_state["treat_fk_cols"] = treat_fk_cols
st.session_state["mapped_targets"] = mapped_targets

# 2) Seed-catalog refs filtered by mapped attributes
seed_catalog = _load_seed_catalog_safe(SEED_CATALOG_PPDM39) if SEED_CATALOG_PPDM39.exists() else {}
seeded_catalog_tables = set(st.session_state.get("seeded_catalog_tables") or [])

def _catalog_applies_to(spec: dict) -> list[str]:
    a = spec.get("applies_to") or spec.get("applies_to_columns") or []
    return [str(x).strip() for x in a if str(x).strip()]

def _is_seeded_by_catalog(fqn: str) -> bool:
    return (fqn in seeded_catalog_tables) or (fqn.lower() in {t.lower() for t in seeded_catalog_tables})

remaining_rows: list[dict] = []

# (A) FK parents
for child_col, (ps, pt, pk_col) in (fk_map or {}).items():
    parent_fqn = f"{ps}.{pt}"
    remaining_rows.append({
        "kind": "FK_PARENT",
        "ref_table": parent_fqn,
        "pk_column": pk_col,
        "driven_by": f"FK value from {child_col}",
        "needs_mapping": True,
    })

# (B) Reference tables from catalog:
mapped_upper = {m.upper() for m in mapped_targets}
for ref_fqn, spec in (seed_catalog or {}).items():
    if ref_fqn in ("_meta", "defaults"):
        continue
    if not isinstance(spec, dict):
        continue

    applies = _catalog_applies_to(spec)
    if not applies:
        continue

    if not {a.upper() for a in applies}.intersection(mapped_upper):
        continue

    if _is_seeded_by_catalog(ref_fqn):
        continue

    mode = _catalog_get_mode(spec)
    remaining_rows.append({
        "kind": "REFERENCE",
        "ref_table": ref_fqn,
        "pk_column": _catalog_get_pk_col(spec),
        "driven_by": f"Catalog ({mode}) / mapped attributes: {', '.join(applies[:5])}" + ("..." if len(applies) > 5 else ""),
        "needs_mapping": (mode != "static"),
    })

remaining_df = pd.DataFrame(remaining_rows).drop_duplicates(subset=["kind", "ref_table"])

if remaining_df.empty:
    st.success("No remaining FK or reference tables detected for the mapped attributes.")
    st.stop()

st.dataframe(remaining_df, width="stretch", hide_index=True)

# --------------------------------------------------------------------
# Match & Map-style grid to supply additional info for a remaining table
# --------------------------------------------------------------------
st.subheader("Configure seeding for a remaining table (Match & Map)")

options = remaining_df.apply(lambda r: f"{r['kind']} :: {r['ref_table']}", axis=1).tolist()
sel = st.selectbox("Choose a remaining table", options, key="remaining_table_pick")

sel_kind, sel_fqn = [s.strip() for s in sel.split("::", 1)]
sel_schema, sel_table = sel_fqn.split(".", 1)

dropdown_options = [""] + sorted(list(set(mapped_targets)))

if "ref_table_mappings" not in st.session_state:
    st.session_state["ref_table_mappings"] = {}

ref_cols_df = fetch_columns(conn, sel_schema, sel_table).copy()
ref_cols_df["seed_default"] = False
ref_cols_df["source_column"] = ""
ref_cols_df["constant_value"] = ""

saved = st.session_state["ref_table_mappings"].get(sel_fqn)
if saved is not None and isinstance(saved, pd.DataFrame) and not saved.empty:
    ref_cols_df = ref_cols_df.merge(
        saved[["column_name", "seed_default", "source_column", "constant_value"]],
        on="column_name",
        how="left",
        suffixes=("", "_saved"),
    )
    for col in ["seed_default", "source_column", "constant_value"]:
        ref_cols_df[col] = ref_cols_df[f"{col}_saved"].where(ref_cols_df[f"{col}_saved"].notna(), ref_cols_df[col])
        ref_cols_df.drop(columns=[f"{col}_saved"], inplace=True)

st.caption(
    "Map additional columns if needed. "
    "For FK parents, PK-only seeding is enough unless you want to populate more columns."
)

edited_ref_map = st.data_editor(
    ref_cols_df[["seed_default", "column_name", "data_type", "source_column", "constant_value"]],
    hide_index=True,
    width="stretch",
    column_config={
        "seed_default": st.column_config.CheckboxColumn("Seed", width="small"),
        "column_name": st.column_config.TextColumn("Target column", disabled=True, width="medium"),
        "data_type": st.column_config.TextColumn("Type", disabled=True, width="small"),
        "source_column": st.column_config.SelectboxColumn(
            "Source attribute",
            options=dropdown_options,
            help="Pick from mapped attributes (NORM columns).",
            width="medium",
        ),
        "constant_value": st.column_config.TextColumn(
            "Constant",
            help="Optional constant applied when seeding.",
            width="medium",
        ),
    },
    key=f"editor::refmap::{sel_fqn}",
)

if st.button("Save seeding mapping for this table", key=f"btn_save_refmap::{sel_fqn}", type="primary"):
    st.session_state["ref_table_mappings"][sel_fqn] = edited_ref_map
    st.success(f"Saved seeding mapping for {sel_fqn}.")

# -----------------------------
# 5) Build / Validate / QC (+ parent seeding)
# -----------------------------
st.header("5) Build / Validate / QC (+ parent seeding)")

domain = (st.text_input("Rules domain", "well", key="rules_domain") or "").strip() or None
seed_parents = st.checkbox("Seed FK parent tables before promote", True, key="seed_parents")
key_strategy = st.selectbox(
    "Parent key strategy",
    ["hash_if_too_long", "hash_always", "raw"],
    index=0,
    key="key_strategy_seed",
)

# recompute mapped + treat_fk from session state
pm_df = st.session_state[pm_key].copy()
pm_df["source_column"] = pm_df.get("source_column", "").fillna("").astype(str).str.strip()
pm_df["constant_value"] = pm_df.get("constant_value", "").fillna("").astype(str).str.strip()

treat_fk_cols = (
    pm_df.loc[
        (pm_df["treat_as_fk"] == True)
        & ((pm_df["source_column"] != "") | (pm_df["constant_value"] != "")),
        "column_name",
    ]
    .astype(str)
    .tolist()
)

pm_mapped = pm_df.loc[(pm_df["source_column"] != "") | (pm_df["constant_value"] != "")].copy()

# ------------------------------------------------------------
# NEW: Distinct seeding rules (seed R tables from NORM distinct)
# ------------------------------------------------------------
st.subheader("Seed reference (R) tables from distinct values in the load")

if "distinct_ref_seeds" not in st.session_state:
    st.session_state["distinct_ref_seeds"] = []  # list[dict]

src_attr = st.selectbox(
    "Source attribute (from NORM)",
    [""] + sorted(mapped_targets),
    key="refseed_src_attr",
)

ref_fqn = st.text_input(
    "Target reference table (schema.table)",
    value="dbo.r_confidential_type",
    key="refseed_ref_fqn",
)

default_active = st.checkbox("Set ACTIVE_IND='Y' if column exists", value=True, key="refseed_default_active")

add_clicked = st.button("Add distinct seeding rule", key="btn_add_refseed")
if add_clicked:
    if not src_attr:
        st.error("Pick a source attribute.")
    elif not ref_fqn or "." not in ref_fqn:
        st.error("Enter a target ref table like dbo.r_confidential_type")
    else:
        st.session_state["distinct_ref_seeds"].append({
            "source_attr": src_attr.strip(),
            "ref_table": ref_fqn.strip(),
            "pk_column": None,  # resolve later
            "defaults": {"ACTIVE_IND": "Y"} if default_active else {},
        })
        st.success(f"Added: seed {ref_fqn} from distinct {src_attr}")

rules = st.session_state.get("distinct_ref_seeds") or []
if rules:
    st.dataframe(pd.DataFrame(rules), hide_index=True, width="stretch")

    c_rm1, c_rm2 = st.columns([1, 2])
    with c_rm1:
        if st.button("Remove last rule", key="btn_remove_last_refseed"):
            st.session_state["distinct_ref_seeds"] = st.session_state["distinct_ref_seeds"][:-1]
            st.rerun()
    with c_rm2:
        if st.button("Clear all rules", key="btn_clear_all_refseed"):
            st.session_state["distinct_ref_seeds"] = []
            st.rerun()

# --------------------------
# Build/Validate/QC button
# --------------------------
if st.button("Build/Validate/QC", key="btn_build_validate_qc"):
    _clear_step5_outputs()
    _clear_promote_outputs()

    view_sql, view_name, maxlen_checks = build_primary_norm_view_sql(
        primary_schema,
        primary_table,
        cols_df,
        st.session_state[pm_key],
        treat_fk_cols,
        pk_hash_enabled=False,
        pk_hash_src=None,
    )

    if st.session_state.get("show_sql", True):
        st.code(view_sql[:800], language="sql")

    db.exec_view_ddl(conn, view_sql)

    st.session_state["view_name"] = view_name
    st.session_state["maxlen_checks"] = maxlen_checks

    # FK map restricted to mapped FK columns only
    fk_map_all = build_fk_map(conn, primary_schema, primary_table, treat_fk_cols)
    treat_fk_set = set([c.strip() for c in (treat_fk_cols or [])])
    fk_map = {c: meta for (c, meta) in (fk_map_all or {}).items() if c in treat_fk_set}
    st.session_state["fk_map"] = fk_map
    st.session_state["treat_fk_cols"] = treat_fk_cols

    # ------------------------------------------------------------
    # NEW: Run distinct reference table seeding rules FIRST
    # ------------------------------------------------------------
    distinct_seed_report = []
for rule in (st.session_state.get("distinct_ref_seeds") or []):
    try:
        src_attr = str(rule.get("source_attr") or "").strip()
        ref_table = str(rule.get("ref_table") or "").strip()
        defaults = rule.get("defaults") or {}

        if not src_attr or not ref_table or "." not in ref_table:
            continue

        ref_schema, ref_name = ref_table.split(".", 1)

        pk_col = rule.get("pk_column")
        if not pk_col:
            pk_col = resolve_best_single_pk(conn, ref_schema, ref_name)
        if not pk_col:
            raise ValueError(
                f"Could not resolve a single-column PK for {ref_table}. "
                f"Set pk_column explicitly (table may have composite PK)."
            )

        n = seed_ref_table_from_view_distinct(
            conn=conn,
            view_name=view_name,
            source_attr=src_attr,
            ref_schema=ref_schema,
            ref_table=ref_name,
            pk_col=pk_col,
            defaults=defaults,
        )
        distinct_seed_report.append({"ref_table": ref_table, "source_attr": src_attr, "pk": pk_col, "inserted_or_merged": n})

    except Exception as e:
        distinct_seed_report.append({"ref_table": rule.get("ref_table"), "source_attr": rule.get("source_attr"), "error": str(e)})


    if distinct_seed_report:
        st.subheader("Distinct reference seeding report (last run)")
        st.dataframe(pd.DataFrame(distinct_seed_report), hide_index=True, width="stretch")

    # Parent mappings (your existing hook)
    parent_mappings = _grid_to_parent_mappings(st.session_state.get("ref_seed_grid"))

    # Seed parents (PK-only if parent_mappings empty)
    if seed_parents and fk_map:
        report = seed_parent_tables_from_view(
            conn=conn,
            view_name=view_name,
            fk_map=fk_map,
            treat_as_fk_cols=treat_fk_cols,
            parent_mappings=parent_mappings,
            key_strategy=key_strategy,
        )
        st.session_state["seed_report_df"] = report

    # Validation rules
    ensure_rules_tables(conn)

    if st.session_state.get("run_validation", True):
        apply_rules(conn, domain, view_name, maxlen_checks, treat_as_fk_cols=treat_fk_cols)
        st.session_state["invalid_rows_df"] = read_sql(
            conn,
            "SELECT TOP (50) * FROM stg.invalid_rows ORDER BY RID, rule_id;",
        )
    else:
        st.session_state["invalid_rows_df"] = None

    # QC pairs: mapped columns only
    pairs = [(r["column_name"], r["source_column"]) for _, r in pm_mapped.iterrows()]
    st.session_state["qc_raw_norm_df"] = qc_raw_vs_norm(conn, view_name, pairs)

# Show outputs
if st.session_state.get("seed_report_df") is not None:
    st.subheader("Parent seeding report (last run)")
    st.dataframe(st.session_state["seed_report_df"], width="stretch", hide_index=True)

if st.session_state.get("invalid_rows_df") is not None:
    st.subheader("Invalid rows (top 50) — last run")
    st.dataframe(st.session_state["invalid_rows_df"], width="stretch", hide_index=True)
elif st.session_state.get("run_validation", True) and st.session_state.get("view_name"):
    st.info("Validation is enabled. Click Build/Validate/QC to populate invalid rows.")

if st.session_state.get("qc_raw_norm_df") is not None:
    st.subheader("QC RAW vs NORM (adjacent) — last run")
    st.dataframe(st.session_state["qc_raw_norm_df"], width="stretch", hide_index=True)
# -----------------------------
# 6) Promote
# -----------------------------
st.header("6) Promote (MERGE into target)")

use_valid_only = st.checkbox("Promote VALID rows only (stg.valid_rid)", True, key="use_valid_only_promote")
key_strategy_promote = st.selectbox(
    "Key strategy (must match seeding)",
    ["hash_if_too_long", "hash_always", "raw"],
    index=0,
    key="key_strategy_promote",
)

pm_df = st.session_state[pm_key].copy()
pm_df["source_column"] = pm_df.get("source_column", "").fillna("").astype(str).str.strip()
pm_df["constant_value"] = pm_df.get("constant_value", "").fillna("").astype(str).str.strip()
mapped_df = pm_df.loc[(pm_df["source_column"] != "") | (pm_df["constant_value"] != "")].copy()
mapped_pairs = [(r["column_name"], r["source_column"]) for _, r in mapped_df.iterrows()]

fk_map = st.session_state.get("fk_map", {}) or {}
treat_fk_cols = st.session_state.get("treat_fk_cols", []) or []

if st.button("Generate Promote SQL", key="btn_generate_promote"):
    _clear_promote_outputs()

    if not st.session_state.get("view_name"):
        st.error("Build the NORM view first (Step 5).")
    else:
        view_name = st.session_state["view_name"]
        try:
            plan = build_promote_plan(
                conn=conn,
                view_name=view_name,
                target_schema=primary_schema,
                target_table=primary_table,
                mapped_cols=mapped_pairs,
                fk_map=fk_map,
                treat_as_fk_cols=treat_fk_cols,
                key_strategy=key_strategy_promote,
                use_valid_rid=use_valid_only,
            )
            st.session_state["promote_plan"] = plan
            st.success("Promote SQL generated.")
        except Exception as e:
            st.session_state["promote_plan"] = None
            st.error(f"Could not generate Promote SQL: {e}")

plan = st.session_state.get("promote_plan")
if plan:
    st.subheader("Promote SQL Preview")
    st.code(plan.merge_sql, language="sql")
else:
    st.info("Generate Promote SQL to preview and run the MERGE.")

if st.button("RUN Promote", key="btn_run_promote"):
    plan = st.session_state.get("promote_plan")
    if not plan:
        st.error("Generate Promote SQL first.")
    else:
        try:
            run_promote(conn, plan)
            st.success(f"Promoted into {plan.target_fqn}")
        except Exception as e:
            st.error(f"Promote failed: {e}")

        try:
            qc_summary = read_promote_qc(conn, target_fqn=plan.target_fqn, top_n=20)
            st.session_state["promote_qc_summary_df"] = qc_summary
        except Exception as e:
            st.session_state["promote_qc_summary_df"] = None
            st.warning(f"Could not read {PROMOTE_QC_TABLE}: {e}")

        try:
            qc_adj = promote_qc_adjacent(
                conn=conn,
                view_name=st.session_state["view_name"],
                target_schema=primary_schema,
                target_table=primary_table,
                mapped_target_cols=[t for (t, _src) in mapped_pairs],
                use_valid_rid=use_valid_only,
                top_n=200,
            )
            st.session_state["promote_qc_adj_df"] = qc_adj
        except Exception as e:
            st.session_state["promote_qc_adj_df"] = None
            st.warning(f"Could not generate adjacent Promote QC: {e}")

if st.session_state.get("promote_qc_summary_df") is not None:
    st.subheader("Promote QC — Summary (last run)")
    st.dataframe(st.session_state["promote_qc_summary_df"], width="stretch", hide_index=True)

if st.session_state.get("promote_qc_adj_df") is not None:
    st.subheader("Promote QC — SRC vs TGT (adjacent, last run)")
    st.dataframe(st.session_state["promote_qc_adj_df"], width="stretch", hide_index=True)

if st.session_state.get("promote_qc_summary_df") is not None and st.session_state.get("promote_qc_adj_df") is not None:
    st.markdown("---")
    st.success("🎉🎉🎉 **SUCCESS!!! Promote completed and QC verified** 🎉🎉🎉")
