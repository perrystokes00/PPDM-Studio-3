# pages/1_Seed_R_Tables.py
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

import ppdm_loader.db as db  # db.read_sql, db.exec_sql
from common.ui import sidebar_connect

sidebar_connect()


# ============================================================
# Helpers
# ============================================================
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


def _quote_ident(name: str) -> str:
    name = (name or "").replace("]", "]]")
    return f"[{name}]"


def _fqn_quoted(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": None, "nan": None, "None": None})
    return s


def table_columns_set(conn, schema: str, table: str) -> set[str]:
    sql = """
    SELECT c.name AS column_name
    FROM sys.schemas s
    JOIN sys.tables  t ON t.schema_id = s.schema_id
    JOIN sys.columns c ON c.object_id = t.object_id
    WHERE s.name = ?
      AND t.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return set()
    return {str(x).strip().upper() for x in df["column_name"].tolist()}


def _get_col_type(conn, schema: str, table: str, col: str) -> str | None:
    sql = """
    SELECT c.DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS c
    WHERE c.TABLE_SCHEMA = ?
      AND c.TABLE_NAME = ?
      AND c.COLUMN_NAME = ?;
    """
    df = db.read_sql(conn, sql, params=[schema, table, col])
    if df is None or df.empty:
        return None
    return str(df.iloc[0]["DATA_TYPE"]).strip().lower()


def resolve_best_single_pk(conn, schema: str, table: str) -> str | None:
    # Try PK constraint first
    pk_sql = """
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
        df = db.read_sql(conn, pk_sql, params=[schema, table])
        if df is not None and not df.empty:
            cols = [str(x).strip() for x in df["COLUMN_NAME"].tolist() if str(x).strip()]
            if len(cols) == 1:
                return cols[0]
            return None  # composite PK -> not supported in this quick seeder
    except Exception:
        pass

    # Otherwise heuristic: common PPDM-ish “code” keys
    cols = table_columns_set(conn, schema, table)
    if not cols:
        return None

    for cand in ("CODE", "TYPE", "STATUS", "SOURCE", "NAME"):
        if cand in cols:
            return cand

    # Any *_ID that isn't audit
    id_cols = [c for c in cols if c.endswith("_ID") and c not in _AUDIT_COLS]
    if len(id_cols) == 1:
        return id_cols[0]
    if id_cols:
        return id_cols[0]

    # last resort: first non-audit col
    non_audit = [c for c in cols if c not in _AUDIT_COLS]
    return non_audit[0] if non_audit else None


def list_ref_tables(conn, schema: str = "dbo", include_ra: bool = True) -> list[str]:
    # r_ + optionally ra_
    like_parts = ["t.name LIKE 'r[_]%' ESCAPE '\\'"]
    if include_ra:
        like_parts.append("t.name LIKE 'ra[_]%' ESCAPE '\\'")

    where_like = " OR ".join(like_parts)

    sql = f"""
    SELECT s.name AS schema_name, t.name AS table_name
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ?
      AND ({where_like})
    ORDER BY t.name;
    """
    df = db.read_sql(conn, sql, params=[schema])
    if df is None or df.empty:
        return []
    return [f"{r['schema_name']}.{r['table_name']}" for _, r in df.iterrows()]


def preview_missing_codes_df(
    conn,
    *,
    target_schema: str,
    target_table: str,
    pk_col: str,
    items: list[dict[str, Any]],
    top_n: int = 2000,
) -> tuple[pd.DataFrame, int]:
    tgt = _fqn_quoted(target_schema, target_table)
    pkq = _quote_ident(pk_col)

    payload = json.dumps(items, ensure_ascii=False)

    sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), j.code))), N'') AS code
    FROM OPENJSON(?)
    WITH (code nvarchar(4000) '$.code') j
),
missing AS (
    SELECT s.code
    FROM src s
    WHERE s.code IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{pkq}))), N'') = s.code
      )
)
SELECT TOP ({int(top_n)}) code
FROM missing
ORDER BY code;
""".strip()

    df_sample = db.read_sql(conn, sql_sample, params=[payload])
    if df_sample is None or df_sample.empty:
        df_sample = pd.DataFrame({"code": []})

    sql_count = f"""
;WITH src AS (
    SELECT DISTINCT
        NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), j.code))), N'') AS code
    FROM OPENJSON(?)
    WITH (code nvarchar(4000) '$.code') j
),
missing AS (
    SELECT s.code
    FROM src s
    WHERE s.code IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{pkq}))), N'') = s.code
      )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

    df_count = db.read_sql(conn, sql_count, params=[payload])
    missing_total = 0
    if df_count is not None and not df_count.empty:
        missing_total = int(df_count.iloc[0]["missing_total"])

    return df_sample, missing_total


def seed_missing_codes(
    conn,
    *,
    target_schema: str,
    target_table: str,
    pk_col: str,
    items: list[dict[str, Any]],
    long_name_mode: str = "none",  # none | code | column
    defaults: dict[str, Any] | None = None,
    loaded_by: str = "Perry M Stokes",
    set_audit: bool = True,
) -> int:
    """
    Insert-only seeding. Adds optional LONG_NAME/ACTIVE_IND/PPDM_GUID and
    (optionally) audit columns if they exist on the target table.
    """
    defaults = defaults or {}
    long_name_mode = (long_name_mode or "none").strip().lower()
    loaded_by = (loaded_by or "Perry M Stokes").strip() or "Perry M Stokes"

    tgt_cols = table_columns_set(conn, target_schema, target_table)
    pk_u = pk_col.strip().upper()
    if pk_u not in tgt_cols:
        raise ValueError(f"PK column '{pk_col}' not found on {target_schema}.{target_table}")

    has_long_name = "LONG_NAME" in tgt_cols
    has_active = "ACTIVE_IND" in tgt_cols
    has_guid = "PPDM_GUID" in tgt_cols

    # audit columns presence
    has_rcb = "ROW_CREATED_BY" in tgt_cols
    has_rcd = "ROW_CREATED_DATE" in tgt_cols
    has_rchb = "ROW_CHANGED_BY" in tgt_cols
    has_rchd = "ROW_CHANGED_DATE" in tgt_cols
    has_red = "ROW_EFFECTIVE_DATE" in tgt_cols

    insert_cols: list[str] = [pk_col.strip()]
    select_exprs: list[str] = ["m.code"]
    already = {c.upper() for c in insert_cols}

    # LONG_NAME
    if has_long_name and long_name_mode != "none":
        insert_cols.append("LONG_NAME")
        if long_name_mode == "code":
            select_exprs.append("m.code")
        elif long_name_mode == "column":
            select_exprs.append("COALESCE(m.long_name, m.code)")
        else:
            raise ValueError(f"Unknown long_name_mode: {long_name_mode}")
        already.add("LONG_NAME")

    # ACTIVE_IND (type-aware)
    if has_active and "ACTIVE_IND" not in already and defaults.get("ACTIVE_IND", None) is not None:
        active_val = defaults.get("ACTIVE_IND", "Y")
        active_type = _get_col_type(conn, target_schema, target_table, "ACTIVE_IND")

        insert_cols.append("ACTIVE_IND")
        if active_type in ("bit",):
            select_exprs.append("CAST(1 AS bit)" if str(active_val).strip().upper() in ("Y", "YES", "TRUE", "1") else "CAST(0 AS bit)")
        elif active_type in ("int", "smallint", "tinyint", "bigint"):
            select_exprs.append("1" if str(active_val).strip().upper() in ("Y", "YES", "TRUE", "1") else "0")
        else:
            vv = str(active_val).replace("'", "''")
            select_exprs.append(f"N'{vv}'")
        already.add("ACTIVE_IND")

    # PPDM_GUID if exists
    if has_guid and "PPDM_GUID" not in already:
        insert_cols.append("PPDM_GUID")
        select_exprs.append("CONVERT(nvarchar(36), NEWID())")
        already.add("PPDM_GUID")

    # Audit columns (insert-only => created/changed same)
    if set_audit:
        by_sql = "N'" + loaded_by.replace("'", "''") + "'"
        now_sql = "SYSUTCDATETIME()"

        if has_rcb and "ROW_CREATED_BY" not in already:
            insert_cols.append("ROW_CREATED_BY")
            select_exprs.append(by_sql)
            already.add("ROW_CREATED_BY")
        if has_rcd and "ROW_CREATED_DATE" not in already:
            insert_cols.append("ROW_CREATED_DATE")
            select_exprs.append(now_sql)
            already.add("ROW_CREATED_DATE")
        if has_rchb and "ROW_CHANGED_BY" not in already:
            insert_cols.append("ROW_CHANGED_BY")
            select_exprs.append(by_sql)
            already.add("ROW_CHANGED_BY")
        if has_rchd and "ROW_CHANGED_DATE" not in already:
            insert_cols.append("ROW_CHANGED_DATE")
            select_exprs.append(now_sql)
            already.add("ROW_CHANGED_DATE")
        if has_red and "ROW_EFFECTIVE_DATE" not in already:
            insert_cols.append("ROW_EFFECTIVE_DATE")
            select_exprs.append(now_sql)
            already.add("ROW_EFFECTIVE_DATE")

    col_list = ", ".join(_quote_ident(c) for c in insert_cols)
    sel_list = ", ".join(select_exprs)

    tgt = _fqn_quoted(target_schema, target_table)
    pkq = _quote_ident(pk_col)

    # clean distinct payload (case-insensitive)
    clean_items: list[dict[str, Any]] = []
    seen = set()
    for it in items:
        code = str((it.get("code") or "")).strip()
        if not code:
            continue
        key = code.upper()
        if key in seen:
            continue
        seen.add(key)
        ln = it.get("long_name")
        ln = str(ln).strip() if ln is not None and str(ln).strip() else None
        clean_items.append({"code": code, "long_name": ln})

    if not clean_items:
        return 0

    payload = json.dumps(clean_items, ensure_ascii=False)

    sql = f"""
;WITH src AS (
    SELECT DISTINCT
        NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), j.code))), N'') AS code,
        NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), j.long_name))), N'') AS long_name
    FROM OPENJSON(?)
    WITH (
        code      nvarchar(4000) '$.code',
        long_name nvarchar(4000) '$.long_name'
    ) j
),
missing AS (
    SELECT s.code, s.long_name
    FROM src s
    WHERE s.code IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{pkq}))), N'') = s.code
      )
)
INSERT INTO {tgt} ({col_list})
SELECT {sel_list}
FROM missing m;
""".strip()

    cur = conn.cursor()
    cur.execute(sql, (payload,))
    inserted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return int(inserted)


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Seed r_/ra_ tables", layout="wide")
st.title("Seed r_ / ra_ tables (drag & drop)")

conn = st.session_state.get("conn")
if conn is None:
    st.error("Not connected. Use the Launchpad sidebar to connect first.")
    st.stop()

# sanity: make sure this is a DBAPI connection, not a string
if isinstance(conn, str) or not hasattr(conn, "cursor"):
    st.error(f"Session 'conn' is not a live DB connection: {type(conn)}. Reconnect in the sidebar.")
    st.stop()

with st.expander("🔎 Debug: connection + target sanity", expanded=True):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(who, hide_index=True, width="stretch")
    try:
        rc = db.read_sql(conn, "SELECT COUNT(*) AS r_source_rows FROM dbo.r_source;")
        st.dataframe(rc, hide_index=True, width="stretch")
    except Exception as e:
        st.caption(f"dbo.r_source not readable here: {e}")

st.caption("Drop a CSV/TXT, pick the target reference table, compute missing codes, then seed missing (insert-only).")

up = st.file_uploader("Drop CSV/TXT here", type=["csv", "txt"], key="rseed_upload")
if not up:
    st.stop()

sep = st.selectbox("Delimiter", [",", "\t", "|", ";"], index=0, key="rseed_sep")

df = pd.read_csv(up, sep=sep, dtype=str, keep_default_na=False)
df.columns = [str(c).strip().upper() for c in df.columns]

st.subheader("Preview")
st.dataframe(df.head(50), hide_index=True, width="stretch")
st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

# table selection
st.subheader("Target table")
schema = st.text_input("Schema", value="dbo", key="rseed_schema").strip() or "dbo"
include_ra = st.checkbox("Include ra_ tables", value=True, key="rseed_include_ra")

tables = [""] + list_ref_tables(conn, schema=schema, include_ra=include_ra)
target_fqn = st.selectbox("Target r_/ra_ table", options=tables, key="rseed_target_fqn")
if not target_fqn:
    st.stop()

t_schema, t_table = target_fqn.split(".", 1)

# PK
pk_guess = resolve_best_single_pk(conn, t_schema, t_table) or ""
pk_col = st.text_input("PK column (auto-detected)", value=pk_guess, key="rseed_pk_col").strip()
if not pk_col:
    st.warning("Enter a PK column (single-column PK only for this quick seeder).")
    st.stop()

# source column
st.subheader("Source mapping (from file)")
default_code_col = "CODE" if "CODE" in df.columns else (df.columns[0] if len(df.columns) else "")
src_col = st.selectbox(
    "Code column in file",
    options=[""] + df.columns.tolist(),
    index=(df.columns.tolist().index(default_code_col) + 1 if default_code_col in df.columns else 0),
    key="rseed_src_col",
)
if not src_col:
    st.stop()

# LONG_NAME strategy (safe default)
long_mode = st.selectbox(
    "LONG_NAME strategy",
    ["none", "code", "column"],
    index=2 if "LONG_NAME" in df.columns else 0,  # safe default
    key="rseed_long_mode",
)
ln_col = st.selectbox(
    "LONG_NAME column in file (only if strategy=column)",
    options=[""] + df.columns.tolist(),
    disabled=(long_mode != "column"),
    key="rseed_ln_col",
)

# Audit behavior
st.subheader("Defaults")
loaded_by = st.text_input("Loaded by (ROW_CREATED_BY / ROW_CHANGED_BY)", value="Perry M Stokes", key="rseed_loaded_by")
set_audit = st.checkbox("Auto-populate audit columns if present", value=True, key="rseed_set_audit")

# ACTIVE default (safe OFF)
default_active = st.checkbox("Set ACTIVE_IND (if column exists)", value=False, key="rseed_default_active")

top_n = st.number_input("Show top N missing", min_value=10, max_value=50000, value=2000, step=100, key="rseed_topn")

c1, c2 = st.columns([1, 1])
with c1:
    compute_btn = st.button("Compute missing", key="rseed_compute", type="primary")
with c2:
    seed_btn = st.button("Seed missing now", key="rseed_seed", type="secondary")


def build_items_from_file() -> list[dict[str, Any]]:
    codes = _normalize_series(df[src_col]).dropna().tolist()

    ln_map: dict[str, str] = {}
    if long_mode == "column" and ln_col:
        tmp = df[[src_col, ln_col]].copy()
        tmp[src_col] = _normalize_series(tmp[src_col])
        tmp[ln_col] = _normalize_series(tmp[ln_col])
        tmp = tmp.dropna(subset=[src_col])
        tmp = tmp.drop_duplicates(subset=[src_col], keep="first")
        for _, r in tmp.iterrows():
            c = r[src_col]
            ln = r[ln_col]
            if c is not None:
                ln_map[str(c)] = (str(ln).strip() if ln is not None else None)

    out: list[dict[str, Any]] = []
    for c in codes:
        cc = str(c).strip()
        if not cc:
            continue
        out.append({"code": cc, "long_name": ln_map.get(cc)})

    seen = set()
    dedup = []
    for it in out:
        key = it["code"].upper()
        if key not in seen:
            dedup.append(it)
            seen.add(key)
    return dedup


if compute_btn:
    try:
        items = build_items_from_file()
        miss_df, miss_total = preview_missing_codes_df(
            conn,
            target_schema=t_schema,
            target_table=t_table,
            pk_col=pk_col,
            items=items,
            top_n=int(top_n),
        )
        st.session_state["rseed_items"] = items
        st.session_state["rseed_missing_df"] = miss_df
        st.session_state["rseed_missing_total"] = miss_total
        st.success("Missing computed.")
    except Exception as e:
        st.error(f"Compute missing failed: {e}")

missing_df = st.session_state.get("rseed_missing_df")
missing_total = st.session_state.get("rseed_missing_total", None)

if missing_df is not None:
    st.subheader("Missing codes")
    st.dataframe(missing_df, hide_index=True, width="stretch")
    shown = len(missing_df) if isinstance(missing_df, pd.DataFrame) else 0
    tot = int(missing_total) if missing_total is not None else 0
    st.caption(f"Missing shown: {shown:,} (sample) | Missing total: {tot:,}")

if seed_btn:
    try:
        items = st.session_state.get("rseed_items") or build_items_from_file()
        defaults = {"ACTIVE_IND": "Y"} if default_active else {}
        inserted = seed_missing_codes(
            conn,
            target_schema=t_schema,
            target_table=t_table,
            pk_col=pk_col,
            items=items,
            long_name_mode=long_mode,
            defaults=defaults,
            loaded_by=loaded_by,
            set_audit=set_audit,
        )
        st.success(f"Seed completed. Inserted {inserted:,} row(s). Recomputing missing…")

        miss_df2, miss_total2 = preview_missing_codes_df(
            conn,
            target_schema=t_schema,
            target_table=t_table,
            pk_col=pk_col,
            items=items,
            top_n=int(top_n),
        )
        st.session_state["rseed_missing_df"] = miss_df2
        st.session_state["rseed_missing_total"] = miss_total2

        st.subheader("Missing after seeding")
        st.dataframe(miss_df2, hide_index=True, width="stretch")
        st.caption(f"Missing shown: {len(miss_df2):,} (sample) | Missing total: {int(miss_total2):,}")
    except Exception as e:
        st.error(f"Seeding failed: {e}")
