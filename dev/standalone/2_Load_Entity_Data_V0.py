# pages/2_Load_Entity_Data.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import ppdm_loader.db as db
from common.ui import sidebar_connect, require_connection

from ppdm_loader.stage import save_upload, stage_bulk_insert, DELIM_MAP
from ppdm_loader.normalize import build_primary_norm_view_sql
from ppdm_loader.fk_suggest import suggest_fk_candidates_step4
from ppdm_loader.fk_introspect import introspect_fk_by_child_col


# --------------------------
# Page setup + connection
# --------------------------
st.set_page_config(page_title="Load entity data", layout="wide")

# SINGLE sidebar owner (shared across all pages)
sidebar_connect(page_prefix="entity")
conn = require_connection()

BULK_DIR = Path(r"C:\Bulk\uploads")
BULK_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# Small helpers
# --------------------------
def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if (df is not None and not df.empty) else pd.DataFrame()


def _rowterm_to_sql(rt: str) -> str:
    rt = (rt or "LF").upper().strip()
    return r"\n" if rt == "LF" else r"\r\n"


def _fetch_child_columns(conn, schema: str, table: str) -> pd.DataFrame:
    sql = """
    SELECT c.name AS column_name
    FROM sys.columns c
    JOIN sys.tables t ON t.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    return df if df is not None else pd.DataFrame(columns=["column_name"])


def _fetch_cols_meta_types(conn, schema: str, table: str) -> pd.DataFrame:
    """
    Return column_name + data_type string suitable for casting (includes lengths/precision/scale).
    """
    sql = """
    SELECT
        c.name AS column_name,
        CASE
            WHEN ty.name IN ('nvarchar','varchar','nchar','char') AND c.max_length = -1 THEN ty.name + '(max)'
            WHEN ty.name IN ('nvarchar','nchar') THEN ty.name + '(' + CAST(c.max_length/2 AS varchar(10)) + ')'
            WHEN ty.name IN ('varchar','char')  THEN ty.name + '(' + CAST(c.max_length AS varchar(10)) + ')'
            WHEN ty.name IN ('decimal','numeric') THEN ty.name + '(' + CAST(c.precision AS varchar(10)) + ',' + CAST(c.scale AS varchar(10)) + ')'
            ELSE ty.name
        END AS data_type
    FROM sys.columns c
    JOIN sys.types ty ON ty.user_type_id = c.user_type_id
    JOIN sys.tables tb ON tb.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = tb.schema_id
    WHERE s.name = ? AND tb.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    return df if df is not None else pd.DataFrame(columns=["column_name", "data_type"])


# --------------------------
# UI
# --------------------------
st.title("Load Entity Data (Primary tables)")

with st.expander("🔎 Debug: connection sanity", expanded=False):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(_safe_df(who), hide_index=True, use_container_width=True)

with st.expander("🔎 Debug: sidebar settings", expanded=False):
    st.write("PPDM version:", st.session_state.get("ppdm_version"))
    st.write("Domain:", st.session_state.get("ppdm_domain"))
    st.write("Catalog loaded:", st.session_state.get("catalog_json") is not None)


# ============================================================
# STEP 1 — Upload / Preview / Save
# ============================================================
st.header("Step 1 — Upload / Preview / Save to C:\\Bulk")

uploaded = st.file_uploader("Drop CSV/TXT here", type=["csv", "txt"], key="entity_upload")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    delim_ui = st.selectbox("Delimiter", list(DELIM_MAP.keys()), index=0, key="entity_delim_ui")
with c2:
    has_header = st.checkbox("Has header", value=True, key="entity_has_header")
with c3:
    rowterm_ui = st.selectbox("RowTerm", ["LF", "CRLF"], index=0, key="entity_rowterm_ui")
with c4:
    preview_n = st.number_input("Preview rows", 5, 200, 25, 5, key="entity_preview_n")

delimiter = DELIM_MAP[delim_ui]
rowterm_sql = _rowterm_to_sql(rowterm_ui)

if uploaded is not None:
    try:
        df_preview = pd.read_csv(
            uploaded,
            sep=delimiter,
            nrows=int(preview_n),
            dtype=str,
            keep_default_na=False,
            engine="python",
            header=0 if has_header else None,
        )
        if not has_header:
            df_preview.columns = [f"COL_{i+1}" for i in range(len(df_preview.columns))]
        st.dataframe(df_preview, hide_index=True, use_container_width=True)
        st.caption(f"Delimiter={repr(delimiter)} RowTerm={rowterm_sql} Cols={len(df_preview.columns)}")
    except Exception as e:
        st.error(f"Preview failed: {e}")
        st.stop()

    if st.button("Save upload to C:\\Bulk", type="secondary", key="entity_save_btn"):
        fp = save_upload(uploaded, BULK_DIR)
        st.session_state["entity_saved_fp"] = str(fp)
        st.success("Saved upload.")
        st.code(str(fp))


# ============================================================
# STEP 2 — Stage
# ============================================================
st.header("Step 2 — Stage (BULK INSERT → stg.raw_data)")

saved_fp = st.session_state.get("entity_saved_fp")
if not saved_fp:
    st.info("Upload + Save to enable staging.")
    st.stop()

st.code(f"File to stage:\n{saved_fp}")

if st.button("Stage", type="primary", key="entity_stage_btn"):
    try:
        cols = stage_bulk_insert(
            conn,
            file_path=saved_fp,
            delimiter=delimiter,
            has_header=has_header,
            rowterm_sql=rowterm_sql,
        )
        st.session_state["entity_source_cols"] = cols
        st.success(f"Staged. Columns detected: {len(cols)}")

        cnt = db.read_sql(conn, "SELECT COUNT(*) AS n FROM stg.raw_data;")
        st.dataframe(_safe_df(cnt), hide_index=True, use_container_width=True)

        prev = db.read_sql(conn, "SELECT TOP (25) * FROM stg.v_raw_with_rid ORDER BY RID;")
        st.subheader("stg.v_raw_with_rid (top 25)")
        st.dataframe(_safe_df(prev), hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Stage failed: {e}")
        st.stop()

source_cols = st.session_state.get("entity_source_cols") or []
if not source_cols:
    st.stop()


# ============================================================
# STEP 3 — Choose Primary Table (child)
# ============================================================
st.header("Step 3 — Choose Primary Table (child)")

primary_fqn = st.text_input(
    "Primary table (schema.table)",
    value=st.session_state.get("primary_table_fqn", "dbo.well"),
    key="primary_fqn",
)
st.session_state["primary_table_fqn"] = primary_fqn

if "." not in primary_fqn:
    st.error("Enter primary table like dbo.well_status")
    st.stop()

child_schema, child_table = primary_fqn.split(".", 1)
child_cols_df = _fetch_child_columns(conn, child_schema, child_table)

if child_cols_df.empty:
    st.error(f"Could not read columns for {child_schema}.{child_table}")
    st.stop()

child_columns = child_cols_df["column_name"].astype(str).tolist()
st.caption("Next: map staged columns → target columns and build the NORM view.")


# ============================================================
# STEP 4 — Mapping grid (primary)
# ============================================================
st.header("Step 4 — Map staged columns → primary table columns")

src_u = {c.upper(): c for c in source_cols}
default_rows = []
for tgt in child_columns:
    guess = src_u.get(str(tgt).upper(), "")
    default_rows.append(
        {
            "column_name": str(tgt),
            "source_column": guess,
            "constant_value": "",
            "transform": "trim",
            "treat_as_fk": False,
        }
    )
map_df_base = pd.DataFrame(default_rows)

if "primary_map_df" in st.session_state and isinstance(st.session_state["primary_map_df"], pd.DataFrame):
    base = st.session_state["primary_map_df"]
    if not base.empty and "column_name" in base.columns:
        map_df_base = base.copy()

c_fk1, c_fk2 = st.columns([1, 3])
with c_fk1:
    auto_fk = st.button("Auto-suggest FK ticks", key="auto_fk_btn", type="secondary")
with c_fk2:
    st.caption("Uses SQL Server FK metadata (sys.foreign_key_columns) for the selected primary table.")

if auto_fk:
    try:
        fk_cols = suggest_fk_candidates_step4(conn, child_schema=child_schema, child_table=child_table)
        fk_set = {c.upper() for c in (fk_cols or [])}
        col_u = map_df_base["column_name"].astype(str).str.upper()
        map_df_base["treat_as_fk"] = col_u.isin(fk_set)

        st.session_state["primary_map_df"] = map_df_base.copy()
        st.success(f"Auto-ticked Treat-as-FK for {len(fk_set)} column(s).")
        st.rerun()
    except Exception as e:
        st.error(f"FK auto-suggest failed: {e}")

map_df = st.data_editor(
    map_df_base,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "column_name": st.column_config.TextColumn("Target column (primary)", required=True, disabled=True),
        "source_column": st.column_config.SelectboxColumn("Staged source column", options=[""] + source_cols),
        "constant_value": st.column_config.TextColumn("Constant (optional)"),
        "transform": st.column_config.SelectboxColumn("Transform", options=["none", "trim", "upper"]),
        "treat_as_fk": st.column_config.CheckboxColumn("Treat as FK (QC / optional seed)"),
    },
    key="primary_map_grid",
)

st.session_state["primary_map_df"] = map_df.copy()


# ============================================================
# STEP 5 — Build NORM view
# ============================================================
st.header("Step 5 — Build NORM view")

if st.button("Build NORM view", type="primary", key="build_norm_btn"):
    try:
        m2 = st.session_state["primary_map_df"].copy()

        for c in ("source_column", "constant_value", "column_name", "treat_as_fk"):
            if c not in m2.columns:
                m2[c] = ""

        m2["source_column"] = m2["source_column"].fillna("").astype(str).str.strip()
        m2["constant_value"] = m2["constant_value"].fillna("").astype(str).str.strip()
        m2["column_name"] = m2["column_name"].fillna("").astype(str).str.strip()

        cols_meta = _fetch_cols_meta_types(conn, child_schema, child_table)
        treat_fk_cols = m2.loc[m2["treat_as_fk"] == True, "column_name"].astype(str).tolist()

        view_sql, view_name, _ = build_primary_norm_view_sql(
            primary_schema=child_schema,
            primary_table=child_table,
            cols_df=cols_meta,
            mapping_df=m2[["column_name", "source_column", "constant_value"]],
            treat_as_fk_cols=treat_fk_cols,
        )

        db.exec_view_ddl(conn, view_sql)
        st.session_state["norm_view_name"] = view_name

        st.success(f"Built NORM view: {view_name}")
        prev = db.read_sql(conn, f"SELECT TOP (25) * FROM {view_name} ORDER BY RID;")
        st.subheader("NORM view preview (top 25)")
        st.dataframe(_safe_df(prev), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Build NORM view failed: {e}")

norm_view = st.session_state.get("norm_view_name")
if not norm_view:
    st.info("Build the NORM view above to enable Step 6.")
    st.stop()


# ============================================================
# STEP 6 — FK QC (missing parent keys) + OPTIONAL seeding
# Default is PASS-THROUGH (QC only)
# ============================================================
st.header("Step 6 — FK QC (missing reference values) + optional seeding")

primary_map_df = st.session_state.get("primary_map_df")
treat_fk_cols = (
    primary_map_df.loc[primary_map_df.get("treat_as_fk") == True, "column_name"]
    .astype(str)
    .tolist()
)
treat_fk_cols = [c for c in treat_fk_cols if c.strip()]

cA, cB, cC = st.columns([2, 2, 2])
with cA:
    pass_through = st.checkbox(
        "Pass-through (QC only, no seeding)",
        value=True,
        key="step6_pass_through",
    )
with cB:
    allow_seeding = st.checkbox(
        "Allow seeding (inserts into parent tables)",
        value=False,
        key="step6_allow_seeding",
    )
with cC:
    top_n = st.number_input("Show top N missing", 10, 50000, 2000, 100, key="step6_topn")

if not treat_fk_cols:
    st.info("No FK columns marked. Tick 'Treat as FK' in Step 4 if you want FK QC.")
else:
    # --- Local helpers for Step 6 ---
    def _qident(name: str) -> str:
        return "[" + (name or "").replace("]", "]]") + "]"

    def _qfqn(schema: str, table: str) -> str:
        return f"{_qident(schema)}.{_qident(table)}"

    def _sql_missing_parent_keys(*, norm_view_fqn: str, fkinfo, top_n: int = 2000) -> tuple[str, str]:
        parent_fqn = _qfqn(fkinfo.parent_schema, fkinfo.parent_table)

        proj_exprs = []
        req_preds = []
        join_preds = []
        order_cols = []

        for child_col, parent_col in fkinfo.pairs:
            expr = f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_qident(child_col + '__NAT')}))), N'')"
            proj_exprs.append(f"{expr} AS {_qident(parent_col)}")
            req_preds.append(f"{_qident(parent_col)} IS NOT NULL")
            order_cols.append(_qident(parent_col))

            join_preds.append(
                f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{_qident(parent_col)}))), N'')"
                f" = s.{_qident(parent_col)}"
            )

        proj_sql = ",\n        ".join(proj_exprs)
        req_sql = " AND ".join(req_preds)
        join_sql = " AND ".join(join_preds)
        order_sql = ", ".join(order_cols)

        sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_sql}
    FROM {norm_view_fqn} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {parent_fqn} t
          WHERE {join_sql}
      )
)
SELECT TOP ({int(top_n)}) *
FROM missing
ORDER BY {order_sql};
""".strip()

        sql_count = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_sql}
    FROM {norm_view_fqn} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {parent_fqn} t
          WHERE {join_sql}
      )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

        return sql_sample, sql_count

    for child_fk_col in treat_fk_cols:
        fkinfo = introspect_fk_by_child_col(
            conn,
            child_schema=child_schema,
            child_table=child_table,
            child_col=child_fk_col,
        )

        if fkinfo is None:
            with st.expander(f"⚠️ {child_fk_col} — No FK found in metadata", expanded=False):
                st.warning("Marked Treat-as-FK, but SQL Server FK metadata did not find a relationship.")
            continue

        parent_label = f"{fkinfo.parent_schema}.{fkinfo.parent_table}"
        title = f"FK: {child_fk_col} → {parent_label}  ({fkinfo.fk_name})"

        with st.expander(title, expanded=False):
            st.dataframe(pd.DataFrame(fkinfo.pairs, columns=["child_col", "parent_col"]), hide_index=True, use_container_width=True)

            sql_sample, sql_count = _sql_missing_parent_keys(norm_view_fqn=norm_view, fkinfo=fkinfo, top_n=int(top_n))

            c1, c2 = st.columns([1, 2])
            if c1.button("Compute missing", key=f"step6_missing_{child_fk_col}", type="primary"):
                miss_df = db.read_sql(conn, sql_sample)
                cnt_df = db.read_sql(conn, sql_count)
                missing_total = int(cnt_df.iloc[0]["missing_total"]) if (cnt_df is not None and not cnt_df.empty) else 0

                st.session_state[f"step6_miss_df_{child_fk_col}"] = _safe_df(miss_df)
                st.session_state[f"step6_miss_total_{child_fk_col}"] = missing_total
                st.session_state[f"step6_sql_sample_{child_fk_col}"] = sql_sample
                st.session_state[f"step6_sql_count_{child_fk_col}"] = sql_count

            # Seeding stays intentionally disabled by default
            seed_enabled = (allow_seeding and (not pass_through))
            c2.button("Seed missing (disabled by default)", disabled=(not seed_enabled), key=f"step6_seed_{child_fk_col}")

            miss_df = st.session_state.get(f"step6_miss_df_{child_fk_col}")
            miss_total = st.session_state.get(f"step6_miss_total_{child_fk_col}")

            if miss_df is not None:
                st.subheader("Missing reference values (sample)")
                st.dataframe(_safe_df(miss_df), hide_index=True, use_container_width=True)
                st.caption(f"Missing shown: {len(miss_df)} | Missing total: {miss_total}")

            with st.expander("SQL — missing sample", expanded=False):
                st.code(sql_sample, language="sql")
            with st.expander("SQL — missing count", expanded=False):
                st.code(sql_count, language="sql")


# ============================================================
# STEP 7 — Promote (insert new only)
# ============================================================
st.header("Step 7 — Promote to target (insert new only)")

# Default: keep your original behavior promoting to dbo.well
# (If you want target to follow Step 3 table selection, tell me and I’ll switch it.)
target_fqn = "dbo.well"
t_schema, t_table = target_fqn.split(".", 1)

view_cols = db.read_sql(conn, f"SELECT TOP (0) * FROM {norm_view};").columns.tolist()
tgt_cols = db.read_sql(conn, f"SELECT TOP (0) * FROM {target_fqn};").columns.tolist()

insert_cols = [c for c in view_cols if c in tgt_cols]
if "UWI" in insert_cols:
    insert_cols = ["UWI"] + [c for c in insert_cols if c != "UWI"]

st.caption(f"Columns to insert: {len(insert_cols)}")
st.code(", ".join(insert_cols))

col_sql = ", ".join(f"[{c}]" for c in insert_cols)
sel_sql = ", ".join(f"v.[{c}]" for c in insert_cols)

sql_count = f"""
SET NOCOUNT ON;
SELECT COUNT(*) AS would_insert
FROM {norm_view} v
WHERE v.[UWI] IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM {target_fqn} w WHERE w.[UWI] = v.[UWI]);
"""

sql_insert = f"""
SET NOCOUNT ON;
INSERT INTO {target_fqn} ({col_sql})
SELECT {sel_sql}
FROM {norm_view} v
WHERE v.[UWI] IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM {target_fqn} w WHERE w.[UWI] = v.[UWI]);
"""

cP1, cP2 = st.columns([1, 1])
if cP1.button("Preview would-insert count", key="promote_preview"):
    df = db.read_sql(conn, sql_count)
    st.dataframe(_safe_df(df), hide_index=True, use_container_width=True)

if cP2.button("Promote now (insert new only)", type="primary", key="promote_insert"):
    db.exec_sql(conn, sql_insert)
    st.success("Promote completed.")

with st.expander("SQL — promote preview / insert", expanded=False):
    st.code(sql_count, language="sql")
    st.code(sql_insert, language="sql")
