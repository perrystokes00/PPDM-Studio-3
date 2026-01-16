# pages/2_Load_Entity_Data.py
from __future__ import annotations

from pathlib import Path
import uuid
import getpass
import pandas as pd
import streamlit as st
import json

from dataclasses import dataclass

import ppdm_loader.db as db
from ppdm_loader.stage import save_upload, stage_bulk_insert, DELIM_MAP
from ppdm_loader.normalize import build_primary_norm_view_sql
from ppdm_loader.fk_introspect import introspect_fk_by_child_col
from ppdm_loader.fk_suggest import suggest_fk_candidates_step4

from common.ui import sidebar_connect, require_connection
from ppdm_loader.child_insert import fetch_fk_map, fetch_pk_columns, build_insert_new_sql_generic
from ppdm_loader.fk_introspect import introspect_all_fks_for_child_table

from ppdm_loader.station_promote import (
    fetch_station_signature,
    plan_station_mode,
    build_station_seq_sql,
    fetch_station_signature, 
    plan_station_mode,
    station_seq_expr_sql,
)

# ============================================================
# Page setup (MULTIPAGE)
# ============================================================
st.set_page_config(page_title="Load entity data", layout="wide")

# SINGLE sidebar owner for multipage
sidebar_connect(page_prefix="ent")

# shared connection from sidebar
conn = require_connection()

BULK_DIR = Path(r"C:\Bulk\uploads")
BULK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
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


def _qident(name: str) -> str:
    return "[" + (name or "").replace("]", "]]") + "]"


def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"

def _strip_brackets(s: str) -> str:
    return (s or "").replace("[", "").replace("]", "").strip()

def _require_schema_qualified(fqn: str, label: str = "object") -> tuple[str, str]:
    fqn = _strip_brackets(fqn or "")
    if "." not in fqn:
        raise ValueError(f"{label} must be schema-qualified like dbo.name, got: {fqn}")
    a, b = fqn.split(".", 1)
    return a.strip(), b.strip()

# ============================================================
# UI
# ============================================================
st.title("Load Entity Data (Primary tables)")

with st.expander("🔎 Debug: connection sanity", expanded=False):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(_safe_df(who), hide_index=True, width="stretch")

with st.expander("🔎 Debug: sidebar settings", expanded=False):
    st.write("PPDM model:", st.session_state.get("ppdm_model") or st.session_state.get("ppdm_version"))
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
        st.dataframe(df_preview, hide_index=True, width="stretch")
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
        st.dataframe(_safe_df(cnt), hide_index=True, width="stretch")

        prev = db.read_sql(conn, "SELECT TOP (10) * FROM stg.v_raw_with_rid ORDER BY RID;")
        st.subheader("stg.v_raw_with_rid (top 10)")
        st.dataframe(_safe_df(prev), hide_index=True, width="stretch")
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
# --- RESET dependent state when primary table changes ---
ss = st.session_state
prev = ss.get("primary_fqn_prev")
if prev != primary_fqn:
    ss["primary_fqn_prev"] = primary_fqn

    # clear downstream state that depends on table choice
    for k in [
        "primary_map_df",
        "primary_map_grid",
        "norm_view_name",
        "entity_fk_cols",
        "entity_fk_pairs",
    ]:
        if k in ss:
            del ss[k]

    # optional: also clear staged-column auto-mapping guesses if you store them
    # ss.pop("entity_source_cols", None)

    st.info(f"Primary table changed: {prev} → {primary_fqn}. Resetting mapping / view.")
    st.rerun()

if "." not in primary_fqn:
    st.error("Enter primary table like dbo.well_dir_srvy")
    st.stop()

child_schema, child_table = primary_fqn.split(".", 1)
child_cols_df = _fetch_child_columns(conn, child_schema, child_table)

if child_cols_df.empty:
    st.error(f"Could not read columns for {child_schema}.{child_table}")
    st.stop()

child_columns = child_cols_df["column_name"].astype(str).tolist()
st.caption("Next you will map staged columns → primary table columns and build the NORM view.")
st.subheader("Suggested tables (from staged columns)")

if st.button("Recommend tables", key="btn_recommend_tables"):
    try:
        # Simple overlap-based suggestion (no extra modules required)
        src_set = {c.strip().upper() for c in source_cols}

        # Use your loaded catalog_json if present
        cat = st.session_state.get("catalog_json") or {}
        rows = []

        # Expect catalog to be either {"tables":[...]} or list-like; adapt lightly:
        tables = cat.get("tables") if isinstance(cat, dict) else None
        if tables is None and isinstance(cat, list):
            tables = cat

        if not tables:
            st.warning("No tables found in catalog_json. Check catalog structure.")
        else:
            for t in tables:
                # try common shapes:
                fqn = t.get("name") or t.get("table") or t.get("fqn")
                cols = t.get("columns") or []
                col_names = []
                for cc in cols:
                    if isinstance(cc, dict):
                        col_names.append(str(cc.get("name", "")).strip())
                    else:
                        col_names.append(str(cc).strip())

                if not fqn or not col_names:
                    continue

                tgt_set = {c.upper() for c in col_names if c}
                overlap = len(src_set & tgt_set)
                if overlap <= 0:
                    continue

                rows.append(
                    {
                        "table": fqn,
                        "overlap": overlap,
                        "target_cols": len(tgt_set),
                        "pct": round(100.0 * overlap / max(1, len(tgt_set)), 1),
                    }
                )

            df = pd.DataFrame(rows).sort_values(["overlap", "pct"], ascending=False).head(25)
            st.session_state["table_recos_df"] = df

    except Exception as e:
        st.error(f"Recommend failed: {e}")

recos = st.session_state.get("table_recos_df")
if isinstance(recos, pd.DataFrame) and not recos.empty:
    st.dataframe(recos, hide_index=True, width="stretch")
    pick = st.selectbox("Pick a suggested table", recos["table"].tolist(), key="pick_reco_table")
    if st.button("Use selected table", type="primary", key="btn_apply_reco"):
        st.session_state["primary_table_fqn"] = pick
        st.session_state["primary_fqn"] = pick  # sync text_input key
        st.success(f"Primary table set to {pick}")
        st.rerun()
else:
    st.caption("No suggestions yet. Click “Recommend tables”.")

# ============================================================
# STEP 4 — Mapping grid (primary)  ✅ stable + batch edits
# ============================================================
st.header("Step 4 — Map staged columns → primary table columns")

ss = st.session_state

# --- detect when we need to (re)initialize defaults ---
sig = {
    "child_schema": child_schema,
    "child_table": child_table,
    "source_cols": tuple(source_cols),
}
sig_key = "primary_map_signature"

def _build_default_map_df(child_columns: list[str], source_cols: list[str]) -> pd.DataFrame:
    src_u = {c.upper(): c for c in source_cols}
    rows = []
    for tgt in child_columns:
        guess = src_u.get(str(tgt).upper(), "")
        rows.append(
            {
                "treat_as_fk": False,               # ✅ FIRST column
                "column_name": str(tgt),            # target
                "source_column": guess,             # source
                "constant_value": "",
                "transform": "trim",
            }
        )
    df = pd.DataFrame(rows)
    return df[["treat_as_fk", "column_name", "source_column", "constant_value", "transform"]]

# Initialize mapping ONLY when needed
need_init = (
    "primary_map_df" not in ss
    or not isinstance(ss["primary_map_df"], pd.DataFrame)
    or ss["primary_map_df"].empty
    or ss.get(sig_key) != sig
)

if need_init:
    ss["primary_map_df"] = _build_default_map_df(child_columns, source_cols)
    ss[sig_key] = sig

# --- Buttons that intentionally change the DF (outside the form is fine) ---
c_fk1, c_fk2, c_fk3 = st.columns([1.2, 1.2, 3.6])

with c_fk1:
    auto_fk = st.button("Auto-tick FK", type="secondary", key="auto_fk_btn")
with c_fk2:
    clear_fk = st.button("Clear FK", type="secondary", key="clear_fk_btn")
with c_fk3:
    st.caption("Edits are applied only when you click **Apply mapping changes**.")

if auto_fk:
    try:
        fk_cols = suggest_fk_candidates_step4(conn, child_schema=child_schema, child_table=child_table) or []
        fk_set = {c.upper() for c in fk_cols}
        df = ss["primary_map_df"].copy()
        df["treat_as_fk"] = df["column_name"].astype(str).str.upper().isin(fk_set)
        ss["primary_map_df"] = df
    except Exception as e:
        st.error(f"FK auto-suggest failed: {e}")

if clear_fk:
    df = ss["primary_map_df"].copy()
    df["treat_as_fk"] = False
    ss["primary_map_df"] = df

# --- The editor (in a form) prevents "refresh on every keypress" problems ---
with st.form("primary_map_form", clear_on_submit=False, border=True):
    edited_df = st.data_editor(
        ss["primary_map_df"],
        width="stretch",          # ✅ Streamlit 2026+ replacement for use_container_width=True
        num_rows="fixed",
        column_config={
            "treat_as_fk": st.column_config.CheckboxColumn("FK", help="Treat as FK (QC / optional seeding)"),
            "column_name": st.column_config.TextColumn("Target column (primary)", disabled=True),
            "source_column": st.column_config.SelectboxColumn("Staged source column", options=[""] + source_cols),
            "constant_value": st.column_config.TextColumn("Constant (optional)"),
            "transform": st.column_config.SelectboxColumn("Transform", options=["none", "trim", "upper"]),
        },
        key="primary_map_grid_editor",  # note: separate key from the df storage
    )

    apply_btn = st.form_submit_button("Apply mapping changes", type="primary")

if apply_btn:
    # Ensure column order stays stable (FK first)
    edited_df = edited_df[["treat_as_fk", "column_name", "source_column", "constant_value", "transform"]].copy()
    ss["primary_map_df"] = edited_df
    st.success("Mapping saved. Proceed to Step 5.")



# ============================================================
# STEP 5 — Build NORM view (schema-qualified + FK __NAT safety)
# ============================================================
st.header("Step 5 — Build NORM view")

if st.button("Build NORM view", type="primary", key="build_norm_btn"):
    try:
        m2 = st.session_state["primary_map_df"].copy()

        # Ensure expected cols exist
        for c in ("source_column", "constant_value", "column_name", "treat_as_fk"):
            if c not in m2.columns:
                m2[c] = ""

        m2["source_column"] = m2["source_column"].fillna("").astype(str).str.strip()
        m2["constant_value"] = m2["constant_value"].fillna("").astype(str).str.strip()
        m2["column_name"] = m2["column_name"].fillna("").astype(str).str.strip()

        # Only FK columns that are actually mapped (source OR constant) should be treated as FK
        m2["treat_as_fk"] = m2["treat_as_fk"].fillna(False).astype(bool)
        treat_fk_cols = m2.loc[
            (m2["treat_as_fk"] == True)
            & (m2["column_name"] != "")
            & ((m2["source_column"] != "") | (m2["constant_value"] != "")),
            "column_name",
        ].astype(str).tolist()
        treat_fk_cols = [c for c in treat_fk_cols if c.strip()]

        # ------------------------------------------------------------
        # Ensure FK-marked columns exist as mapping rows (so builder can emit __NAT)
        # If a FK column is marked+mapped but somehow absent as a mapping row, add passthrough.
        # ------------------------------------------------------------
        mapped_targets = set(m2["column_name"].astype(str).str.strip().tolist())
        for fk_col in treat_fk_cols:
            if fk_col not in mapped_targets:
                m2 = pd.concat(
                    [
                        m2,
                        pd.DataFrame(
                            [
                                {
                                    "column_name": fk_col,
                                    "source_column": fk_col,  # passthrough
                                    "constant_value": "",
                                    "treat_as_fk": True,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        cols_meta = _fetch_cols_meta_types(conn, child_schema, child_table)

        view_sql, view_name, _ = build_primary_norm_view_sql(
            primary_schema=child_schema,
            primary_table=child_table,
            cols_df=cols_meta,
            mapping_df=m2[["column_name", "source_column", "constant_value"]],
            treat_as_fk_cols=treat_fk_cols,
        )

        db.exec_view_ddl(conn, view_sql)

        # IMPORTANT: persist schema-qualified name for Step 6/7
        # build_primary_norm_view_sql typically returns something like dbo.stg_v_norm_dbo_well
        # If it did not, we force dbo.<name>.
        if "." in view_name:
            norm_view_fqn = view_name
        else:
            norm_view_fqn = f"dbo.{view_name}"

        st.session_state["norm_view_name"] = view_name
        st.session_state["norm_view_fqn"] = norm_view_fqn

        st.success(f"Built NORM view: {st.session_state['norm_view_fqn']}")
        st.code(st.session_state["norm_view_fqn"])

        prev = db.read_sql(conn, f"SELECT TOP (25) * FROM {st.session_state['norm_view_fqn']} ORDER BY RID;")
        st.subheader("NORM view preview (top 25)")
        st.dataframe(_safe_df(prev), hide_index=True, width="stretch")

    except Exception as e:
        st.error(f"Build NORM view failed: {e}")

norm_view_fqn = st.session_state.get("norm_view_fqn")
if not norm_view_fqn:
    st.info("Build the NORM view above to enable Step 6/7.")
    st.stop()

# Read view columns once for Step 6/7 usage
try:
    _v0 = db.read_sql(conn, f"SELECT TOP (0) * FROM {norm_view_fqn};")
    view_cols = list(_v0.columns)
    view_cols_u = {c.upper(): c for c in view_cols}
except Exception as e:
    st.error(f"Could not read NORM view columns: {e}")
    st.stop()


# ============================================================
# STEP 6 — FK Resolver (ONLY mapped FKs) + Parent Loaders
# Option A behavior:
#   - If ALL FK parents are satisfied => show success + set flag
#   - DO NOT st.stop() the page (so Promote UI below can still render)
# ============================================================

st.header("Step 6 — FK Resolver (parent loaders)")

import json
import getpass
from pathlib import Path

# ------------------------------------------------------------
# Created-by / loaded-by (audit defaults)
# ------------------------------------------------------------
created_by = (
    st.session_state.get("loaded_by")
    or st.session_state.get("user_name")
    or getpass.getuser()
    or "ppdm_loader"
)

# ------------------------------------------------------------
# Helpers (local so Step 6 is self-contained)
# ------------------------------------------------------------
def _strip_brackets(s: str) -> str:
    return (s or "").replace("[", "").replace("]", "").strip()

def _qident(name: str) -> str:
    n = _strip_brackets(name).replace("]", "]]")
    return f"[{n}]"

def _qfqn(schema: str, name: str) -> str:
    return f"{_qident(schema)}.{_qident(name)}"

def _require_schema_qualified(fqn: str, label: str = "object") -> tuple[str, str]:
    fqn = _strip_brackets(fqn or "")
    if "." not in fqn:
        raise ValueError(f"{label} must be schema-qualified like dbo.name, got: {fqn}")
    a, b = fqn.split(".", 1)
    return a.strip(), b.strip()

def _norm_fqn_sql(fqn: str) -> str:
    s, n = _require_schema_qualified(fqn, "norm_view_fqn")
    return _qfqn(s, n)

def _s6_sql_literal(val: str) -> str:
    v = (val or "").replace("'", "''")
    return f"N'{v}'"

def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame() if df is None else df.copy()

def _ss_df(key: str) -> pd.DataFrame:
    v = st.session_state.get(key)
    return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

# ------------------------------------------------------------
# FK introspection (local, DB-driven)
# ------------------------------------------------------------
@dataclass
class FKInfo:
    fk_name: str
    child_schema: str
    child_table: str
    parent_schema: str
    parent_table: str
    pairs: list[tuple[str, str]]  # (child_col, parent_col)

def introspect_fk_by_child_col(conn, *, child_schema: str, child_table: str, child_col: str) -> FKInfo | None:
    """
    Returns FKInfo for the FK that includes (child_schema.child_table.child_col).
    If multiple FKs include that column, returns the first by fk_name (stable enough for Step 6).
    """
    sql = """
    ;WITH fks AS (
        SELECT
            fk.name AS fk_name,
            sch_child.name AS child_schema,
            t_child.name AS child_table,
            sch_parent.name AS parent_schema,
            t_parent.name AS parent_table,
            c_child.name AS child_col,
            c_parent.name AS parent_col,
            fkc.constraint_column_id
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
             ON fkc.constraint_object_id = fk.object_id
        JOIN sys.tables t_child
             ON t_child.object_id = fk.parent_object_id
        JOIN sys.schemas sch_child
             ON sch_child.schema_id = t_child.schema_id
        JOIN sys.columns c_child
             ON c_child.object_id = t_child.object_id
            AND c_child.column_id = fkc.parent_column_id
        JOIN sys.tables t_parent
             ON t_parent.object_id = fk.referenced_object_id
        JOIN sys.schemas sch_parent
             ON sch_parent.schema_id = t_parent.schema_id
        JOIN sys.columns c_parent
             ON c_parent.object_id = t_parent.object_id
            AND c_parent.column_id = fkc.referenced_column_id
        WHERE sch_child.name = ? AND t_child.name = ? AND c_child.name = ?
    )
    SELECT *
    FROM fks
    ORDER BY fk_name, constraint_column_id;
    """
    df = db.read_sql(conn, sql, params=[child_schema, child_table, child_col])
    if df is None or df.empty:
        return None

    fk_name = str(df.iloc[0]["fk_name"])
    child_schema = str(df.iloc[0]["child_schema"])
    child_table = str(df.iloc[0]["child_table"])
    parent_schema = str(df.iloc[0]["parent_schema"])
    parent_table = str(df.iloc[0]["parent_table"])

    pairs = []
    for _, r in df.iterrows():
        pairs.append((str(r["child_col"]), str(r["parent_col"])))

    return FKInfo(
        fk_name=fk_name,
        child_schema=child_schema,
        child_table=child_table,
        parent_schema=parent_schema,
        parent_table=parent_table,
        pairs=pairs,
    )

# ------------------------------------------------------------
# Parent-table column + PK introspection
# ------------------------------------------------------------
def _s6_get_table_columns(conn, schema: str, table: str) -> pd.DataFrame:
    sql = """
    SELECT
        c.name AS [name],
        c.is_nullable,
        c.is_computed,
        c.is_identity
    FROM sys.columns c
    JOIN sys.tables  t ON t.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    return df if df is not None else pd.DataFrame()

def _s6_get_pk_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
    SELECT c.name AS column_name
    FROM sys.indexes i
    JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id
    JOIN sys.columns c ON c.object_id = ic.object_id AND c.column_id = ic.column_id
    JOIN sys.tables  t ON t.object_id = i.object_id
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE i.is_primary_key = 1 AND s.name = ? AND t.name = ?
    ORDER BY ic.key_ordinal;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return df["column_name"].astype(str).tolist()

# ------------------------------------------------------------
# NORM view columns (so dropdowns + __NAT logic works reliably)
# ------------------------------------------------------------
def _s6_get_view_columns(conn, norm_view_fqn: str) -> list[str]:
    try:
        s, n = _require_schema_qualified(norm_view_fqn, "norm_view_fqn")
    except Exception:
        return []

    sql = """
    SELECT c.name AS col_name
    FROM sys.columns c
    JOIN sys.views v ON v.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = v.schema_id
    WHERE s.name = ? AND v.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[s, n])
    if df is None or df.empty:
        return []
    return df["col_name"].astype(str).tolist()

# ------------------------------------------------------------
# Missing SQL builder (by FK pairs)
# ------------------------------------------------------------
def _s6_build_missing_sql_for_fk(*, norm_view_sql: str, parent_fqn_sql: str, fk_pairs: list[tuple[str, str]], top_n: int,
                                view_cols_u: dict[str, str]) -> tuple[str, str]:
    proj_exprs = []
    req_preds = []
    join_preds = []
    order_cols = []

    for child_col, parent_col in fk_pairs:
        nat_col = f"{child_col}__NAT"

        # prefer NAT column if present in view
        if nat_col.upper() in view_cols_u:
            src_ref = f"v.{_qident(view_cols_u[nat_col.upper()])}"
        elif child_col.upper() in view_cols_u:
            src_ref = f"v.{_qident(view_cols_u[child_col.upper()])}"
        else:
            src_ref = f"v.{_qident(nat_col)}"  # will error later; we’ll catch earlier

        expr = f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {src_ref}))), N'')"
        proj_exprs.append(f"{expr} AS {_qident(parent_col)}")
        req_preds.append(f"{_qident(parent_col)} IS NOT NULL")
        order_cols.append(_qident(parent_col))

        join_preds.append(
            f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{_qident(parent_col)}))), N'') = s.{_qident(parent_col)}"
        )

    proj_sql = ",\n        ".join(proj_exprs)
    req_sql = " AND ".join(req_preds) or "1=1"
    join_sql = " AND ".join(join_preds) or "1=0"
    order_sql = ", ".join(order_cols) or "1"

    sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_sql}
    FROM {norm_view_sql} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {parent_fqn_sql} t
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
    FROM {norm_view_sql} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {parent_fqn_sql} t
          WHERE {join_sql}
      )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

    return sql_sample, sql_count

# ------------------------------------------------------------
# Mapping grid defaults + required validation
# ------------------------------------------------------------
def _s6_default_parent_mapping_df(
    *,
    parent_cols_df: pd.DataFrame,
    parent_pk_cols: list[str],
    view_cols: list[str],
    fk_pairs_child_to_parent: list[tuple[str, str]],
    loaded_map: dict | None,
) -> pd.DataFrame:
    pk_u = {c.upper() for c in (parent_pk_cols or [])}
    fk_parent_u = {p.upper() for (_, p) in (fk_pairs_child_to_parent or [])}
    view_u = {c.upper(): c for c in view_cols}

    rows = []
    for _, r in parent_cols_df.iterrows():
        col = str(r["name"])
        is_pk = "Y" if col.upper() in pk_u else ""
        not_null = "Y" if int(r.get("is_nullable", 1) or 1) == 0 else ""
        is_computed = int(r.get("is_computed", 0) or 0)
        is_identity = int(r.get("is_identity", 0) or 0)

        include = "N" if (is_computed or is_identity) else ("Y" if (col.upper() in pk_u or col.upper() in fk_parent_u) else "N")

        # default mapping based on FK pairs
        src_default = ""
        for child_col, parent_col in (fk_pairs_child_to_parent or []):
            if parent_col.upper() == col.upper():
                nat = f"{child_col}__NAT"
                if nat.upper() in view_u:
                    src_default = view_u[nat.upper()]
                elif child_col.upper() in view_u:
                    src_default = view_u[child_col.upper()]
                break

        rows.append(
            {
                "include": include,
                "is_pk": is_pk,
                "target_column": col,
                "not_null": not_null,
                "source_column": src_default,
                "constant_value": "",
            }
        )

    df = pd.DataFrame(rows)

    # overlay saved mapping if present
    if loaded_map and isinstance(loaded_map, dict):
        grid_rows = loaded_map.get("grid_rows") or []
        if isinstance(grid_rows, list) and grid_rows:
            lm = pd.DataFrame(grid_rows)
            if "target_column" in lm.columns:
                lm_u = {str(tc).upper(): row for tc, row in zip(lm["target_column"].astype(str), lm.to_dict(orient="records"))}
                out = []
                for row in df.to_dict(orient="records"):
                    tc_u = str(row["target_column"]).upper()
                    if tc_u in lm_u:
                        rr = lm_u[tc_u]
                        row["include"] = str(rr.get("include", row["include"]) or row["include"])
                        row["source_column"] = str(rr.get("source_column", row["source_column"]) or row["source_column"])
                        row["constant_value"] = str(rr.get("constant_value", row["constant_value"]) or row["constant_value"])
                    out.append(row)
                df = pd.DataFrame(out)

    return df

def _s6_validate_required_columns(parent_cols_df: pd.DataFrame, edited_df: pd.DataFrame) -> list[str]:
    # NOT NULL + insertable must be supplied (unless auto-defaults)
    not_null = set(
        parent_cols_df.loc[parent_cols_df["is_nullable"].fillna(1).astype(int).eq(0), "name"]
        .astype(str).tolist()
    )
    insertable = set(
        parent_cols_df.loc[
            (parent_cols_df["is_computed"].fillna(0).astype(int).eq(0))
            & (parent_cols_df["is_identity"].fillna(0).astype(int).eq(0)),
            "name",
        ].astype(str).tolist()
    )
    required = {c.upper() for c in not_null.intersection(insertable)}

    auto_defaults = {
        "ROW_CREATED_BY",
        "ROW_CHANGED_BY",
        "ROW_CREATED_DATE",
        "ROW_CHANGED_DATE",
        "ROW_EFFECTIVE_DATE",
        "PPDM_GUID",
    }

    ed = edited_df.copy()
    ed["include"] = ed["include"].fillna("Y").astype(str).str.upper()
    ed["target_column"] = ed["target_column"].fillna("").astype(str).str.strip()
    ed["source_column"] = ed["source_column"].fillna("").astype(str).str.strip()
    ed["constant_value"] = ed["constant_value"].fillna("").astype(str)

    provided = {}
    for _, r in ed.iterrows():
        if r["include"] != "Y":
            continue
        tc = r["target_column"]
        if not tc:
            continue
        provided[tc.upper()] = (r["source_column"] != "") or (str(r["constant_value"]).strip() != "")

    missing = []
    for col_u in sorted(required):
        if col_u in auto_defaults:
            continue
        if not provided.get(col_u, False):
            missing.append(col_u)
    return missing

# ------------------------------------------------------------
# Seed SQL builder (INSERT missing rows; TEST=ROLLBACK / APPLY=COMMIT)
# ------------------------------------------------------------
def _s6_build_seed_insert_sql(
    *,
    norm_view_sql: str,
    parent_schema: str,
    parent_table: str,
    parent_cols_df: pd.DataFrame,
    parent_pk_cols: list[str],
    mapping_rows: pd.DataFrame,
    created_by: str,
    test_mode: bool,
    view_cols_u: dict[str, str],
) -> str:
    pc = parent_cols_df.copy()
    pc["name_u"] = pc["name"].astype(str).str.upper()
    insertable_u = set(
        pc.loc[
            (pc["is_computed"].fillna(0).astype(int).eq(0))
            & (pc["is_identity"].fillna(0).astype(int).eq(0)),
            "name_u",
        ].astype(str).tolist()
    )

    ed = mapping_rows.copy()
    ed["include"] = ed["include"].fillna("Y").astype(str).str.upper()
    ed["target_column"] = ed["target_column"].fillna("").astype(str).str.strip()
    ed["source_column"] = ed["source_column"].fillna("").astype(str).str.strip()
    ed["constant_value"] = ed["constant_value"].fillna("").astype(str)

    rows = []
    for _, r in ed.iterrows():
        tc = r["target_column"]
        if not tc:
            continue
        if r["include"] != "Y":
            continue
        if tc.upper() not in insertable_u:
            continue
        rows.append(r.to_dict())

    if not rows:
        raise ValueError("No insertable columns selected (include=Y).")

    parent_fqn_sql = _qfqn(parent_schema, parent_table)

    # keys: PK if present else all included cols
    key_cols = parent_pk_cols[:] if parent_pk_cols else [r["target_column"] for r in rows]

    sel_parts = []
    for r in rows:
        tc = r["target_column"]
        sc = r.get("source_column", "")
        cv = r.get("constant_value", "")

        u = tc.upper()
        if u == "PPDM_GUID":
            expr = "NEWID()"
        elif u == "ROW_CREATED_BY":
            expr = _s6_sql_literal(created_by)
        elif u == "ROW_CHANGED_BY":
            expr = _s6_sql_literal(created_by)
        elif u in {"ROW_CREATED_DATE", "ROW_CHANGED_DATE", "ROW_EFFECTIVE_DATE"}:
            expr = "SYSUTCDATETIME()"
        elif sc:
            if sc.upper() not in view_cols_u:
                raise ValueError(f"Mapped source column '{sc}' not found in NORM view.")
            expr = f"v.{_qident(view_cols_u[sc.upper()])}"
        elif str(cv).strip() != "":
            expr = _s6_sql_literal(str(cv))
        else:
            expr = "NULL"

        sel_parts.append(f"{expr} AS {_qident(tc)}")

    sel_sql = ",\n        ".join(sel_parts)
    ins_cols_sql = ", ".join(_qident(r["target_column"]) for r in rows)

    nn_sql = " AND ".join([f"s.{_qident(kc)} IS NOT NULL" for kc in key_cols]) or "1=1"
    join_sql = " AND ".join([f"t.{_qident(kc)} = s.{_qident(kc)}" for kc in key_cols]) or "1=0"

    end_tx = "ROLLBACK TRANSACTION;" if test_mode else "COMMIT TRANSACTION;"

    sql = f"""
BEGIN TRANSACTION;

;WITH src AS (
    SELECT DISTINCT
        {sel_sql}
    FROM {norm_view_sql} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {nn_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {parent_fqn_sql} t
          WHERE {join_sql}
      )
)
INSERT INTO {parent_fqn_sql} ({ins_cols_sql})
SELECT {", ".join([_qident(r["target_column"]) for r in rows])}
FROM missing;

SELECT @@ROWCOUNT AS rows_inserted;

{end_tx}
""".strip()

    return sql

# ------------------------------------------------------------
# Step 6 state defaults (Option A needs these)
# ------------------------------------------------------------
st.session_state.setdefault("s6_no_missing_parents", False)
st.session_state.setdefault("s6_scan_df", pd.DataFrame())

# ------------------------------------------------------------
# Read norm_view_fqn + view columns
# ------------------------------------------------------------
norm_view_fqn = (st.session_state.get("norm_view_fqn") or st.session_state.get("norm_view_name") or "").strip()
if not norm_view_fqn:
    st.warning("No normalized view found (norm_view_fqn). Build the NORM view in Step 5 first.")
    st.session_state["s6_no_missing_parents"] = True
else:
    # Refresh view columns from DB (reliable; avoids NameErrors later)
    view_cols = _s6_get_view_columns(conn, norm_view_fqn)
    view_cols_u = {str(c).upper(): str(c) for c in view_cols}

# ============================================================
# 6A — Identify FK columns to check (ONLY mapped + Treat-as-FK)
# ============================================================
primary_map_df = st.session_state.get("primary_map_df")
if primary_map_df is None or primary_map_df.empty:
    st.warning("No mapping found (primary_map_df). Complete Step 4 first.")
    st.session_state["s6_no_missing_parents"] = True
else:
    m = primary_map_df.copy()
    for c in ("column_name", "source_column", "constant_value", "treat_as_fk"):
        if c not in m.columns:
            m[c] = ""

    m["column_name"] = m["column_name"].fillna("").astype(str).str.strip()
    m["source_column"] = m["source_column"].fillna("").astype(str).str.strip()
    m["constant_value"] = m["constant_value"].fillna("").astype(str).str.strip()
    m["treat_as_fk"] = m["treat_as_fk"].fillna(False).astype(bool)

    mapped_fk_cols = (
        m.loc[
            (m["treat_as_fk"] == True)
            & (m["column_name"] != "")
            & ((m["source_column"] != "") | (m["constant_value"] != "")),
            "column_name",
        ]
        .astype(str)
        .tolist()
    )
    mapped_fk_cols = [c for c in mapped_fk_cols if c]

    st.caption(f"FK columns checked (Treat-as-FK + mapped): {len(mapped_fk_cols)}")

    if not mapped_fk_cols:
        st.info("No mapped Treat-as-FK columns found. Nothing to scan in Step 6.")
        st.session_state["s6_no_missing_parents"] = True

# ============================================================
# 6B — Scan missing parents for mapped FK columns
# ============================================================
scan_rows = []
# SAFE: never evaluates a DataFrame in boolean context
scan_df = st.session_state.get("s6_scan_df")
if scan_df is None or not isinstance(scan_df, pd.DataFrame):
    scan_df = pd.DataFrame()


top_n = st.number_input("Sample rows per FK (TOP N)", 10, 50000, 2000, 100, key="s6_topn")
scan_now = st.button("Scan missing FK parents", type="primary", key="s6_scan_now")

if (not st.session_state.get("s6_no_missing_parents", False)) and mapped_fk_cols and norm_view_fqn:
    if scan_now or scan_df.empty:
        norm_view_sql = _norm_fqn_sql(norm_view_fqn)

        for child_fk_col in mapped_fk_cols:
            fkinfo = introspect_fk_by_child_col(
                conn,
                child_schema=child_schema,
                child_table=child_table,
                child_col=child_fk_col,
            )
            if fkinfo is None:
                continue

            # Ensure required source columns exist in view (prefer __NAT)
            missing_cols = []
            for (cc, _) in fkinfo.pairs:
                nat = f"{cc}__NAT"
                if nat.upper() not in view_cols_u and cc.upper() not in view_cols_u:
                    missing_cols.append(nat)

            parent_schema = fkinfo.parent_schema
            parent_table = fkinfo.parent_table
            parent_fqn_sql = _qfqn(parent_schema, parent_table)

            sql_sample, sql_count = _s6_build_missing_sql_for_fk(
                norm_view_sql=norm_view_sql,
                parent_fqn_sql=parent_fqn_sql,
                fk_pairs=fkinfo.pairs,
                top_n=int(top_n),
                view_cols_u=view_cols_u,
            )

            missing_total = None
            err = ""

            if missing_cols:
                err = f"NORM view missing expected columns: {missing_cols}"
            else:
                try:
                    cnt_df = db.read_sql(conn, sql_count)
                    missing_total = int(cnt_df.iloc[0]["missing_total"]) if (cnt_df is not None and not cnt_df.empty) else 0
                except Exception as e:
                    err = str(e)

            scan_rows.append(
                {
                    "fk_name": fkinfo.fk_name,
                    "child_column_trigger": child_fk_col,
                    "child_cols": ", ".join([c for (c, _) in fkinfo.pairs]),
                    "parent_schema": parent_schema,
                    "parent_table": parent_table,
                    "parent_cols": ", ".join([p for (_, p) in fkinfo.pairs]),
                    "missing_total": missing_total,
                    "error": err,
                    "sql_count": sql_count,
                    "sql_sample": sql_sample,
                }
            )

        scan_df = pd.DataFrame(scan_rows)
        if not scan_df.empty:
            scan_df["missing_total_sort"] = (
                pd.to_numeric(scan_df["missing_total"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
            scan_df = scan_df.sort_values(
                ["missing_total_sort", "parent_schema", "parent_table"],
                ascending=[False, True, True],
            ).drop(columns=["missing_total_sort"])

        st.session_state["s6_scan_df"] = scan_df

scan_df = st.session_state.get("s6_scan_df")
if not isinstance(scan_df, pd.DataFrame):
    scan_df = pd.DataFrame()


if scan_df.empty:
    st.warning("No FK metadata found for mapped FK columns (or scan returned nothing).")
    # Option A: do NOT stop page; just mark as satisfied so Promote can still show
    st.session_state["s6_no_missing_parents"] = True
else:
    st.subheader("FK missing-parent scan (by FK)")
    with st.expander("FK scan table", expanded=False):
        st.dataframe(
            scan_df[
                ["fk_name", "child_column_trigger", "child_cols", "parent_schema", "parent_table", "parent_cols", "missing_total", "error"]
            ],
            hide_index=True,
            width="stretch",
        )

# ============================================================
# 6B.1 — Parent summary + parent pick (Option A)
# ============================================================
parents_df = pd.DataFrame()
if not scan_df.empty:
    parents_df = (
        scan_df.copy()
        .groupby(["parent_schema", "parent_table"], as_index=False)
        .agg(
            missing_total=("missing_total", "sum"),
            fk_count=("fk_name", "count"),
        )
    )
    parents_df["missing_total"] = (
        pd.to_numeric(parents_df["missing_total"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    parents_df = parents_df.sort_values(
        ["missing_total", "parent_schema", "parent_table"],
        ascending=[False, True, True],
    )
    parents_df["label"] = parents_df.apply(
        lambda r: f"{r['parent_schema']}.{r['parent_table']}  (missing≈{int(r['missing_total'])}, fks={int(r['fk_count'])})",
        axis=1,
    )

with st.expander("Parent summary (grouped)", expanded=False):
    if parents_df is None or parents_df.empty:
        st.info("No parent summary available.")
    else:
        st.dataframe(
            parents_df[["parent_schema", "parent_table", "missing_total", "fk_count"]],
            hide_index=True,
            width="stretch",
        )

no_missing = (parents_df is None or parents_df.empty or int(parents_df["missing_total"].sum()) == 0)
st.session_state["s6_no_missing_parents"] = bool(no_missing)

if st.session_state["s6_no_missing_parents"]:
    st.success("No missing FK parents detected. You should be able to Promote.")
    ps, pt = None, None
else:
    # NOTE: Only one selectbox with this key anywhere in Step 6.
    pick_parent = st.selectbox(
        "Pick a parent table to seed",
        parents_df["label"].tolist(),
        key=f"s6_parent_pick__{child_schema}.{child_table}",
    )
    ps, pt = pick_parent.split("  (missing≈", 1)[0].split(".", 1)

# ============================================================
# 6D — FK rows driving this selected parent (from scan)
# ============================================================
fk_rows = pd.DataFrame()
fk_pairs_default: list[tuple[str, str]] = []

if ps and pt:
    fk_rows = scan_df.loc[(scan_df["parent_schema"] == ps) & (scan_df["parent_table"] == pt)].copy()

    st.subheader(f"FKs referencing this parent: {ps}.{pt}")
    with st.expander("FKs referencing selected parent", expanded=False):
        st.dataframe(
            _safe_df(fk_rows)[["fk_name", "child_column_trigger", "child_cols", "parent_cols", "missing_total", "error"]],
            hide_index=True,
            width="stretch",
        )

    try:
        trig = str(fk_rows.iloc[0].get("child_column_trigger") or "")
        fkinfo0 = introspect_fk_by_child_col(conn, child_schema=child_schema, child_table=child_table, child_col=trig)
        fk_pairs_default = fkinfo0.pairs if fkinfo0 is not None else []
    except Exception:
        fk_pairs_default = []

# ============================================================
# 6E — Introspect parent columns + PK + mapping grid (behind expanders)
# ============================================================
parent_cols_df = pd.DataFrame()
parent_pk_cols: list[str] = []
parent_introspect_error = ""

if ps and pt:
    try:
        parent_cols_df = _s6_get_table_columns(conn, ps, pt)
        parent_pk_cols = _s6_get_pk_columns(conn, ps, pt)
    except Exception as e:
        parent_introspect_error = str(e)

with st.expander("Parent table structure (debug)", expanded=False):
    if not ps or not pt:
        st.info("No parent selected (no missing parents, or scan not run).")
    elif parent_introspect_error:
        st.error(f"Could not introspect parent table {ps}.{pt}")
        st.code(parent_introspect_error)
    elif parent_cols_df.empty:
        st.warning("No parent column metadata found.")
    else:
        st.caption(f"Parent PK: {', '.join(parent_pk_cols) if parent_pk_cols else '(not detected)'}")
        st.dataframe(
            parent_cols_df[["name", "is_nullable", "is_computed", "is_identity"]],
            hide_index=True,
            width="stretch",
        )

# If we can't build a mapping grid, skip seeding UI (but DO NOT stop the page)
if ps and pt and (not parent_cols_df.empty) and (not parent_introspect_error):
    # ---- map save/load path ----
    map_dir = Path(
        st.session_state.get("s6_map_dir")
        or r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\ppdm-39-seed-packs\fk_seed_maps"
    )
    st.session_state["s6_map_dir"] = str(map_dir)

    def _s6_map_path(map_dir: Path, schema: str, table: str) -> Path:
        map_dir.mkdir(parents=True, exist_ok=True)
        return map_dir / f"{schema}.{table}.fkmap.json"

    def _s6_load_map(map_dir: Path, schema: str, table: str) -> dict | None:
        p = _s6_map_path(map_dir, schema, table)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _s6_save_map(map_dir: Path, schema: str, table: str, payload: dict) -> None:
        p = _s6_map_path(map_dir, schema, table)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded_map = _s6_load_map(map_dir, ps, pt)

    mapping_df = _s6_default_parent_mapping_df(
        parent_cols_df=parent_cols_df,
        parent_pk_cols=parent_pk_cols,
        view_cols=view_cols,
        fk_pairs_child_to_parent=fk_pairs_default,
        loaded_map=loaded_map,
    )

    st.subheader("Parent insert mapping (target columns ⇐ NORM view columns / constants)")
    st.caption("Tip: keep structure and SQL behind expanders; the grid is the main UI.")

    editor_cols = ["include", "is_pk", "target_column", "not_null", "source_column", "constant_value"]
    col_config = {
        "include": st.column_config.SelectboxColumn("Include", options=["Y", "N"], width="small"),
        "is_pk": st.column_config.TextColumn("PK", disabled=True, width="small"),
        "target_column": st.column_config.TextColumn("Target Column", disabled=True),
        "not_null": st.column_config.TextColumn("NOT NULL", disabled=True, width="small"),
        "source_column": st.column_config.SelectboxColumn("Source Column (NORM view)", options=[""] + view_cols),
        "constant_value": st.column_config.TextColumn("Constant (optional)"),
    }

    edited = st.data_editor(
        mapping_df[editor_cols],
        hide_index=True,
        width="stretch",
        column_config=col_config,
        key=f"s6_parent_mapping_editor__{ps}.{pt}",
    )

    b0, b1, b2 = st.columns([1, 1, 2])
    do_save = b0.button("Save mapping", type="primary", key=f"s6_save_map__{ps}.{pt}")
    do_test = b1.button("TEST seed (ROLLBACK)", key=f"s6_test_seed__{ps}.{pt}")
    do_apply = b2.button("APPLY seed (COMMIT)", key=f"s6_apply_seed__{ps}.{pt}")

    if do_save:
        payload = {
            "table_schema": ps,
            "table_name": pt,
            "table_fqn": f"{ps}.{pt}",
            "pk_columns": parent_pk_cols,
            "grid_rows": edited.to_dict(orient="records"),
        }
        _s6_save_map(map_dir, ps, pt, payload)
        st.success(f"Saved mapping: {str(_s6_map_path(map_dir, ps, pt))}")

    # ============================================================
    # 6F — Missing-detection SQL (debug)
    # ============================================================
    with st.expander("Missing-detection SQL (debug)", expanded=False):
        if fk_rows is not None and not fk_rows.empty:
            first_row = fk_rows.iloc[0].to_dict()
            st.caption("sql_count")
            st.code(str(first_row.get("sql_count", "")), language="sql")
            st.caption("sql_sample")
            st.code(str(first_row.get("sql_sample", "")), language="sql")
        else:
            st.info("No FK rows available for this parent.")

    # ============================================================
    # 6G — Execute seed (TEST/APPLY)
    # ============================================================
    if do_test or do_apply:
        try:
            if not norm_view_fqn:
                raise ValueError("norm_view_fqn is not set. Build the NORM view in Step 5 first.")

            _require_schema_qualified(norm_view_fqn, "norm_view_fqn")
            norm_view_sql = _norm_fqn_sql(norm_view_fqn)  # [dbo].[stg_v_norm_...]

            missing_required = _s6_validate_required_columns(parent_cols_df, edited)
            if missing_required:
                raise ValueError(
                    "Cannot seed: parent has NOT NULL columns with no mapping/constant: "
                    + ", ".join(missing_required)
                )

            # ====================================================
            # DEBUG — parent-of-parent FK violation (BLOCKING SEED)
            # ====================================================
            with st.expander("🔍 Debug: parent FK violations (blocking seed)", expanded=False):
                try:
                    dbg_sql = f"""
                    SELECT DISTINCT
                        su.STRAT_NAME_SET_ID
                    FROM dbo.strat_unit su
                    LEFT JOIN dbo.strat_name_set sns
                        ON su.STRAT_NAME_SET_ID = sns.STRAT_NAME_SET_ID
                    WHERE su.STRAT_NAME_SET_ID IS NOT NULL
                      AND sns.STRAT_NAME_SET_ID IS NULL
                    ORDER BY su.STRAT_NAME_SET_ID;
                    """

                    dbg_df = db.read_sql(conn, dbg_sql)

                    if dbg_df.empty:
                        st.success("✅ No missing STRAT_NAME_SET_ID values. Parent FK is satisfied.")
                    else:
                        st.error("❌ Missing STRAT_NAME_SET_ID values (this blocks seeding):")
                        st.dataframe(dbg_df, hide_index=True, width="stretch")
                        st.caption(
                            "These values MUST exist in dbo.strat_name_set "
                            "before dbo.strat_unit can be seeded."
                        )

                except Exception as dbg_e:
                    st.warning(f"Debug query failed: {dbg_e}")

            # ====================================================
            # ACTUAL SEED SQL
            # ====================================================
            sql_seed = _s6_build_seed_insert_sql(
                norm_view_sql=norm_view_sql,
                parent_schema=ps,
                parent_table=pt,
                parent_cols_df=parent_cols_df,
                parent_pk_cols=parent_pk_cols,
                mapping_rows=edited,
                created_by=created_by,
                test_mode=bool(do_test),
                view_cols_u=view_cols_u,
            )

            with st.expander("Generated seed SQL (debug)", expanded=False):
                st.code(sql_seed, language="sql")

            out = db.read_sql(conn, sql_seed)
            rows_ins = int(out.iloc[0]["rows_inserted"]) if (out is not None and not out.empty) else 0

            if do_test:
                st.success(f"TEST seed complete (rolled back). Would insert {rows_ins} row(s) into {ps}.{pt}.")
            else:
                st.success(f"APPLY seed complete. Inserted {rows_ins} row(s) into {ps}.{pt}.")
                st.info("Now click 'Scan missing FK parents' again to refresh what remains.")

        except Exception as e:
            st.error(f"Seeding failed: {e}")
            st.info(
                "If this fails with an FK conflict ON THE PARENT, it means the parent depends on another table. "
                "Seed the upstream parent first, then retry."
            )


else:
    # no mapping/seed UI rendered; keep page alive
    pass



# ============================================================
# STEP 7 — Promote (insert new OR merge/update)
#   (PK-aware + derived keys + audit defaults)
# ============================================================
st.header("Step 7 — Promote to target")

# ------------------------------------------------------------------
# Derived PK registry (interval / pick style tables)
# Key: lowercase "schema.table"
# ------------------------------------------------------------------
DERIVED_PK_REGISTRY = {
    "dbo.well_zone_interval": {
        "derived": {
            "INTERVAL_ID": {
                "len": 20,
                "inputs": ["UWI", "SOURCE", "ZONE_ID", "ZONE_SOURCE", "TOP_MD", "BASE_MD"],
            },
        }
    },
    "dbo.zone_interval": {
        "derived": {
            "INTERVAL_ID": {
                "len": 20,
                "inputs": ["ZONE_ID", "SOURCE", "TOP_MD", "BASE_MD"],
            }
        }
    },
    "dbo.strat_well_section": {
        "derived": {
            "INTERP_ID": {
                "len": 20,
                "inputs": ["UWI", "STRAT_NAME_SET_ID", "STRAT_UNIT_ID", "SOURCE", "PICK_DEPTH"],
            }
        }
    },
    "dbo.well_node_strat_unit": {
        "derived": {
            "STRAT_UNIT_SHA1": {
                "len": 20,
                "inputs": ["NODE_ID", "STRAT_NAME_SET_ID", "STRAT_UNIT_ID", "SOURCE"],
            }
        }
    },
    "dbo.well_dir_srvy_station": {
        "derived": {
        }
    }
}

# -----------------------------
# Target selection
# -----------------------------
target_fqn = st.text_input("Target table (schema.table)", value=primary_fqn, key="promote_target_fqn")
if "." not in target_fqn:
    st.error("Enter target table like dbo.well_zone_interval")
    st.stop()

tgt_schema, tgt_table = target_fqn.split(".", 1)
target_full = _qfqn(tgt_schema, tgt_table)

# -----------------------------
# Promote mode
# -----------------------------
# Prefer the session value you already use in Step 5/6:
norm_view_fqn = (
    st.session_state.get("norm_view_fqn")
    or st.session_state.get("norm_view_name")
    or ""
).strip()

if not norm_view_fqn:
    st.error("No normalized view found. Build the NORM view in Step 5 first.")
    st.stop()

# Ensure schema-qualified and normalize to bracketed SQL-safe form
nv_schema, nv_name = _require_schema_qualified(norm_view_fqn, "norm_view_fqn")
norm_view = f"[{nv_schema}].[{nv_name}]"   # ✅ now Promote can use norm_view

promote_mode_key = f"promote_mode__{tgt_schema}.{tgt_table}".lower()
promote_update_key = f"promote_update_style__{tgt_schema}.{tgt_table}".lower()
promote_mode = st.radio(
    "Promote mode",
    ["Insert new only", "Merge/Update existing"],
    index=0,
    horizontal=True,
    key="promote_mode",
)

update_style = "Fill-only"
if promote_mode == "Merge/Update existing":
    update_style = st.radio(
        "Update style",
        ["Fill-only", "Overwrite"],
        index=0,
        horizontal=True,
        help="Fill-only updates target only when target is blank. Overwrite updates target when source has a value.",
        key="promote_update_style",
    )

# -----------------------------
# Read columns
# -----------------------------
try:
    view_cols = db.read_sql(conn, f"SELECT TOP (0) * FROM {norm_view};").columns.tolist()
    tgt_cols  = db.read_sql(conn, f"SELECT TOP (0) * FROM {target_full};").columns.tolist()
except Exception as e:
    st.error(f"Could not read columns for promote: {e}")
    st.stop()

view_cols_u = {c.upper(): c for c in view_cols}
tgt_cols_u  = {c.upper(): c for c in tgt_cols}

# intersection (what we can pull directly from v.*)
insertable_from_view = [c for c in view_cols if c.upper() in tgt_cols_u]

st.caption(f"Columns available from NORM view → target: {len(insertable_from_view)}")
with st.expander("Intersection columns", expanded=False):
    st.code(", ".join(insertable_from_view) if insertable_from_view else "(none)")

# -----------------------------
# PK columns (composite-safe)
# -----------------------------
pk_cols = fetch_pk_columns(conn, schema=tgt_schema, table=tgt_table) or []
if not pk_cols:
    st.error("Could not detect PK columns for target. Step 7 requires a PK for safe promote.")
    st.stop()

st.caption(f"Target PK: {', '.join(pk_cols)}")

# -----------------------------
# Audit + GUID defaults (insert-time and optional merge-time)
# -----------------------------
LOADED_BY = "Perry M Stokes"
who = (LOADED_BY or "").replace("'", "''")

# NOTE: For merge/update, I recommend only ROW_CHANGED_* (not ROW_CREATED_*)
AUDIT_DEFAULTS_INSERT = {
    "ROW_CREATED_BY":     lambda: f"N'{who}'",
    "ROW_CHANGED_BY":     lambda: f"N'{who}'",
    "ROW_CREATED_DATE":   lambda: "SYSUTCDATETIME()",
    "ROW_CHANGED_DATE":   lambda: "SYSUTCDATETIME()",
    "ROW_EFFECTIVE_DATE": lambda: "SYSUTCDATETIME()",
}
AUDIT_DEFAULTS_UPDATE = {
    "ROW_CHANGED_BY":   lambda: f"N'{who}'",
    "ROW_CHANGED_DATE": lambda: "SYSUTCDATETIME()",
}

# -----------------------------
# Derived key support
# -----------------------------
def _sql_trim(alias: str, col: str) -> str:
    return f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {alias}.{_qident(col)}))), N'')"

def _derived_expr(alias: str, inputs: list[str], out_len: int) -> str:
    parts = [f"COALESCE({_sql_trim(alias, c)}, N'')" for c in inputs] or ["N''"]
    concat = " + N'|' + ".join(parts)
    return (
        "LEFT(CONVERT(varchar(40), HASHBYTES('SHA1', CONVERT(varbinary(max), "
        f"{concat})), 2), {int(out_len)})"
    )

tkey = f"{tgt_schema}.{tgt_table}".lower()
derived_cfg = DERIVED_PK_REGISTRY.get(tkey, {})
derived_map = (derived_cfg.get("derived", {}) or {})

derived_cols_active: dict[str, dict] = {}
for dcol, cfg in derived_map.items():
    dcol_u = dcol.upper()
    # derive if target has it AND PK uses it
    if dcol_u in tgt_cols_u and dcol in pk_cols:
        derived_cols_active[dcol] = cfg

derived_u = {c.upper() for c in derived_cols_active.keys()}

if derived_cols_active:
    st.caption("Derived PK enabled: " + ", ".join(derived_cols_active.keys()))
else:
    st.caption("Derived PK: none (or not configured).")

# Validate derived inputs exist in view
for dcol, cfg in derived_cols_active.items():
    missing_inputs = [c for c in cfg.get("inputs", []) if c.upper() not in view_cols_u]
    if missing_inputs:
        st.error(
            f"Derived column {dcol} configured with inputs not in NORM view: {missing_inputs}. "
            "Fix mapping / rebuild view or adjust DERIVED_PK_REGISTRY."
        )
        st.stop()

# CROSS APPLY if derived keys needed
cross_apply_sql = ""
if derived_cols_active:
    bits = []
    for dcol, cfg in derived_cols_active.items():
        expr = _derived_expr("v", cfg.get("inputs", []), int(cfg.get("len", 20)))
        bits.append(f"{expr} AS {_qident(dcol)}")
    cross_apply_sql = "CROSS APPLY (SELECT " + ", ".join(bits) + ") h"

# -----------------------------
# Build final insert column list:
# - start with insertable_from_view
# - ensure ALL PK cols exist via view or derived
# - add PPDM_GUID + audit cols (if exist on target)
# -----------------------------
final_cols = list(insertable_from_view)

# Ensure PK cols are included (view or derived)
for pk in pk_cols:
    pku = pk.upper()
    if pku in view_cols_u and pku in tgt_cols_u:
        col_real = view_cols_u[pku]
        if col_real not in final_cols:
            final_cols.append(col_real)
    elif pku in derived_u:
        if pk not in final_cols:
            final_cols.append(pk)
    else:
        st.error(
            f"PK column '{pk}' is not available in NORM view and not derived. "
            f"Cannot safely promote into {target_fqn}."
        )
        st.stop()

# Add PPDM_GUID if target has it and not already included
if "PPDM_GUID" in tgt_cols_u and "PPDM_GUID" not in {c.upper() for c in final_cols}:
    final_cols.append("PPDM_GUID")

# Add audit cols (insert defaults) if target has them
for a in ["ROW_CREATED_BY", "ROW_CREATED_DATE", "ROW_CHANGED_BY", "ROW_CHANGED_DATE", "ROW_EFFECTIVE_DATE"]:
    if a in tgt_cols_u and a not in {c.upper() for c in final_cols}:
        final_cols.append(a)

st.caption(f"Final column list (used for INSERT; MERGE updates only non-PK): {len(final_cols)}")
with st.expander("Final columns", expanded=False):
    st.code(", ".join(final_cols))

# -----------------------------
# SELECT expression per column (INSERT)
# -----------------------------
def _select_expr_insert(col: str) -> str:
    u = col.upper()

    if u in derived_u:
        return f"h.{_qident(col)}"
    if u in AUDIT_DEFAULTS_INSERT:
        return AUDIT_DEFAULTS_INSERT[u]()
    if u == "PPDM_GUID":
        return "CONVERT(nvarchar(36), NEWID())"
    if u in view_cols_u:
        return f"v.{_qident(view_cols_u[u])}"
    return "NULL"

# -----------------------------
# PK predicates (existence)
#   - v.pk for normal
#   - h.pk for derived
# -----------------------------
pk_not_null_preds = []
exists_preds = []
on_preds = []

for pk in pk_cols:
    pku = pk.upper()
    if pku in derived_u:
        pk_not_null_preds.append(f"h.{_qident(pk)} IS NOT NULL")
        exists_preds.append(f"t.{_qident(pk)} = h.{_qident(pk)}")
        on_preds.append(f"t.{_qident(pk)} = h.{_qident(pk)}")
    else:
        pk_not_null_preds.append(f"v.{_qident(pk)} IS NOT NULL")
        exists_preds.append(f"t.{_qident(pk)} = v.{_qident(pk)}")
        on_preds.append(f"t.{_qident(pk)} = v.{_qident(pk)}")

pk_not_null_sql = " AND ".join(pk_not_null_preds)
exists_sql = " AND ".join(exists_preds)
on_sql = " AND ".join(on_preds)

# ============================================================
# INSERT-ONLY Promote UI (single, safe, no duplicates)
#   - builds sql_count_insert/sql_insert BEFORE using them
#   - guards if upstream vars aren't ready yet
# ============================================================

_required = [
    "final_cols",
    "norm_view",
    "target_full",
    "pk_not_null_sql",
    "exists_sql",
    "cross_apply_sql",
    "_select_expr_insert",
    "_qident",
]
_missing = [n for n in _required if n not in locals()]

if _missing:
    with st.expander("🧾 Insert-only SQL (debug)", expanded=False):
        st.info("Insert-only SQL not available yet. Missing: " + ", ".join(_missing))
else:
    # 1) Build SQL strings FIRST
    col_sql_insert = ", ".join(_qident(c) for c in final_cols)
    sel_sql_insert = ", ".join(f"{_select_expr_insert(c)} AS {_qident(c)}" for c in final_cols)

    sql_count_insert = f"""
SET NOCOUNT ON;
SELECT COUNT(*) AS would_insert
FROM {norm_view} v
{cross_apply_sql}
WHERE {pk_not_null_sql}
  AND NOT EXISTS (
      SELECT 1
      FROM {target_full} t
      WHERE {exists_sql}
  );
""".strip()

    sql_insert = f"""
SET NOCOUNT ON;
BEGIN TRAN;

INSERT INTO {target_full} ({col_sql_insert})
SELECT {sel_sql_insert}
FROM {norm_view} v
{cross_apply_sql}
WHERE {pk_not_null_sql}
  AND NOT EXISTS (
      SELECT 1
      FROM {target_full} t
      WHERE {exists_sql}
  );

SELECT @@ROWCOUNT AS rows_inserted;

COMMIT TRAN;
""".strip()

    # 2) Buttons
    c1, c2 = st.columns([1, 1])
    do_count = c1.button("Count would-insert rows", key=f"count_insert__{target_full}")
    do_insert = c2.button("Run INSERT-only", type="primary", key=f"run_insert__{target_full}")

    if do_count:
        try:
            df = db.read_sql(conn, sql_count_insert)
            n = int(df.iloc[0]["would_insert"]) if (df is not None and not df.empty) else 0
            st.info(f"Would insert: {n:,} row(s).")
        except Exception as e:
            st.error(f"Count failed: {e}")

    if do_insert:
        try:
            db.exec_sql(conn, sql_insert)
            st.success("INSERT-only completed.")
        except Exception as e:
            st.error(f"Insert failed: {e}")

    # 3) Debug expander (last)
    with st.expander("🧾 Insert-only SQL (debug)", expanded=False):
        st.caption("Would insert (count)")
        st.code(sql_count_insert, language="sql")

        st.caption("Insert SQL")
        st.code(sql_insert, language="sql")


# ============================================================
# MERGE/UPDATE SQL (FIXED aliasing)
# - ON clause must reference only t.<col> and src.<col>
# - derived cols are computed inside USING and surfaced as src.<derived_col>
# ============================================================

# Build a USING projection that includes:
#   1) all view columns (v.<col>) as their target names
#   2) derived PK columns (computed) as their column names
# This ensures ON can reference src.<pk> for both normal and derived.

using_proj = []

# include all view columns that exist in the target (and/or are used for updates/inserts)
# (we can just include all view_cols; it's fine)
for c in view_cols:
    using_proj.append(f"v.{_qident(c)} AS {_qident(c)}")

# add derived cols as actual columns in src
for dcol, cfg in derived_cols_active.items():
    expr = _derived_expr("v", cfg.get("inputs", []), int(cfg.get("len", 20)))
    using_proj.append(f"{expr} AS {_qident(dcol)}")

using_proj_sql = ",\n        ".join(using_proj)

# USING subquery: v is only visible INSIDE this subquery
using_sql = f"""
(
    SELECT
        {using_proj_sql}
    FROM {norm_view} v
    WHERE {pk_not_null_sql}
) AS src
""".strip()

# ON clause: reference ONLY t.<pk> and src.<pk>
on_parts = []
for pk in pk_cols:
    on_parts.append(f"t.{_qident(pk)} = src.{_qident(pk)}")
on_sql_merge = " AND ".join(on_parts)

# UPDATE SET: use src.<col> only (NOT v.<col>)
set_lines = []
pk_u = {p.upper() for p in pk_cols}

def _src_expr(col: str) -> str:
    u = col.upper()
    if u in AUDIT_DEFAULTS_UPDATE:
        return AUDIT_DEFAULTS_UPDATE[u]()          # literal expressions OK
    if u == "PPDM_GUID":
        return f"t.{_qident(col)}"                 # don't update guid
    return f"src.{_qident(col)}"

def _is_blank(expr: str) -> str:
    return f"({expr} IS NULL OR NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {expr}))), N'') IS NULL)"

def _has_value(expr: str) -> str:
    return f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {expr}))), N'') IS NOT NULL"

# updatable columns = target intersection minus PKs minus RID
updatable_cols = [c for c in insertable_from_view if c.upper() not in pk_u and c.upper() != "RID"]

# add audit update cols if present
for a in ["ROW_CHANGED_BY", "ROW_CHANGED_DATE"]:
    if a in tgt_cols_u and a not in {c.upper() for c in updatable_cols}:
        updatable_cols.append(a)

for c in updatable_cols:
    u = c.upper()
    tgt_expr = f"t.{_qident(c)}"
    src_expr = _src_expr(c)

    if u in {"ROW_CHANGED_BY", "ROW_CHANGED_DATE"}:
        set_lines.append(f"{tgt_expr} = {src_expr}")
        continue

    src_has = _has_value(src_expr)
    if update_style == "Fill-only":
        cond = f"{_is_blank(tgt_expr)} AND {src_has}"
        set_lines.append(f"{tgt_expr} = CASE WHEN {cond} THEN {src_expr} ELSE {tgt_expr} END")
    else:
        set_lines.append(f"{tgt_expr} = CASE WHEN {src_has} THEN {src_expr} ELSE {tgt_expr} END")

set_sql = ",\n    ".join(set_lines) if set_lines else f"t.{_qident(pk_cols[0])} = t.{_qident(pk_cols[0])}"

# INSERT VALUES list must also reference src.* (and derived cols already exist on src)
def _insert_value_expr(col: str) -> str:
    u = col.upper()
    if u in AUDIT_DEFAULTS_INSERT:
        return AUDIT_DEFAULTS_INSERT[u]()
    if u == "PPDM_GUID":
        return "CONVERT(nvarchar(36), NEWID())"
    return f"src.{_qident(col)}"

sql_preview_merge = f"""
SET NOCOUNT ON;
SELECT
  SUM(CASE WHEN EXISTS (
      SELECT 1 FROM {target_full} t WHERE { " AND ".join([f"t.{_qident(pk)} = src.{_qident(pk)}" for pk in pk_cols]) }
  ) THEN 1 ELSE 0 END) AS would_match,
  SUM(CASE WHEN NOT EXISTS (
      SELECT 1 FROM {target_full} t WHERE { " AND ".join([f"t.{_qident(pk)} = src.{_qident(pk)}" for pk in pk_cols]) }
  ) THEN 1 ELSE 0 END) AS missing_in_target
FROM {using_sql};
""".strip()

sql_merge = f"""
SET NOCOUNT ON;

MERGE {target_full} AS t
USING {using_sql}
ON {on_sql_merge}
WHEN MATCHED THEN
    UPDATE SET
    {set_sql}
WHEN NOT MATCHED BY TARGET THEN
    INSERT ({col_sql_insert})
    VALUES ({", ".join([_insert_value_expr(c) for c in final_cols])});
""".strip()


def fetch_station_signature(conn, schema: str, table: str):
    """
    Returns (parent_schema, parent_table, parent_keys, seq_col) if target looks like a station table:
      PK = (FK cols to one parent) + (one extra col)
    Else returns None.
    """
    pk_cols = fetch_pk_columns(conn, schema=schema, table=table) or []
    if not pk_cols:
        return None

    fk_rows = introspect_all_fks_for_child_table(conn, child_schema=schema, child_table=table)

    if fk_rows is None:
        return None

    # If DataFrame, convert to list[dict]
    if hasattr(fk_rows, "to_dict"):
        if fk_rows.empty:
            return None
        fk_rows = fk_rows.to_dict(orient="records")

    if not fk_rows:
        return None

    if fk_rows is None:
        return None

    # If DataFrame, convert to list[dict]
    if hasattr(fk_rows, "to_dict"):
        fk_rows = fk_rows.to_dict(orient="records")

    if not fk_rows:
        return None

    fks = {}
    for r in fk_rows:
        fk_name = r.get("fk_name") or r.get("FK_NAME") or r.get("constraint_name") or r.get("CONSTRAINT_NAME")
        if not fk_name:
            continue

        parent_schema = r.get("parent_schema") or r.get("PARENT_SCHEMA") or r.get("fk_table_schema") or r.get("PARENT_TABLE_SCHEMA")
        parent_table  = r.get("parent_table")  or r.get("PARENT_TABLE")  or r.get("fk_table_name")   or r.get("PARENT_TABLE_NAME")
        child_col     = r.get("child_col")     or r.get("CHILD_COL")     or r.get("fk_column_name")  or r.get("COLUMN_NAME")

        if not child_col or not parent_table:
            continue

        d = fks.setdefault(fk_name, {
            "parent_schema": parent_schema or "dbo",
            "parent_table":  parent_table,
            "child_cols": [],
        })
        d["child_cols"].append(child_col)

    pk_set = {c.upper() for c in pk_cols}

    for info in fks.values():
        child_cols = info["child_cols"]
        if not child_cols:
            continue

        fk_set = {c.upper() for c in child_cols}
        if not fk_set.issubset(pk_set):
            continue

        extras = [c for c in pk_cols if c.upper() not in fk_set]
        if len(extras) != 1:
            continue

        return (info["parent_schema"], info["parent_table"], child_cols, extras[0])

    return None

def choose_station_order_cols(view_cols: list[str]) -> list[str]:
    prefer = ["STATION_MD", "MD", "MEASURED_DEPTH", "DEPTH", "TVD", "STATION_TVD", "STATION_TVDSS", "RID"]
    view_u = {c.upper(): c for c in view_cols}
    chosen = [view_u[p] for p in prefer if p in view_u]
    return chosen or []
    
station_mode = locals().get("station_mode", False)
station_seq_col = locals().get("station_seq_col", None)
st.caption(f"Station mode: {'ON' if station_mode else 'OFF'} (seq col: {station_seq_col or '(none)'})")

# ============================================================
# PROMOTE (Insert-only + Merge) with:
#   ✅ composite PK safe
#   ✅ hash-derived PK support (DERIVED_PK_REGISTRY)
#   ✅ generic PPDM "station-table" sequencing support (ROW_NUMBER)
#      - detects PK = (FK parent keys) + (1 seq col)
#      - generates seq col in a CTE ("src")
#      - gates on parent existence to avoid FK failures
# ============================================================

# -----------------------------
# Promote mode
# -----------------------------
norm_view_fqn = (
    st.session_state.get("norm_view_fqn")
    or st.session_state.get("norm_view_name")
    or ""
).strip()

if not norm_view_fqn:
    st.error("No normalized view found. Build the NORM view in Step 5 first.")
    st.stop()

nv_schema, nv_name = _require_schema_qualified(norm_view_fqn, "norm_view_fqn")
norm_view = f"[{nv_schema}].[{nv_name}]"
promote_mode_key = f"promote_mode__{tgt_schema}.{tgt_table}".lower()
promote_update_key = f"promote_update_style__{tgt_schema}.{tgt_table}".lower()

promote_mode = st.radio(
    "Promote mode",
    ["Insert new only", "Merge/Update existing"],
    index=0,
    horizontal=True,
    key=promote_mode_key,
)

update_style = "Fill-only"
if promote_mode == "Merge/Update existing":
    update_style = st.radio(
        "Update style",
        ["Fill-only", "Overwrite"],
        index=0,
        horizontal=True,
        help="Fill-only updates target only when target is blank. Overwrite updates target when source has a value.",
        key=promote_update_key,
    )


# -----------------------------
# Read columns
# -----------------------------
try:
    view_cols = db.read_sql(conn, f"SELECT TOP (0) * FROM {norm_view};").columns.tolist()
    tgt_cols  = db.read_sql(conn, f"SELECT TOP (0) * FROM {target_full};").columns.tolist()
except Exception as e:
    st.error(f"Could not read columns for promote: {e}")
    st.stop()

view_cols_u = {c.upper(): c for c in view_cols}
tgt_cols_u  = {c.upper(): c for c in tgt_cols}

insertable_from_view = [c for c in view_cols if c.upper() in tgt_cols_u]

st.caption(f"Columns available from NORM view → target: {len(insertable_from_view)}")
with st.expander("Intersection columns", expanded=False):
    st.code(", ".join(insertable_from_view) if insertable_from_view else "(none)")

# -----------------------------
# PK columns (composite-safe)
# -----------------------------
pk_cols = fetch_pk_columns(conn, schema=tgt_schema, table=tgt_table) or []
if not pk_cols:
    st.error("Could not detect PK columns for target. Step 7 requires a PK for safe promote.")
    st.stop()

st.caption(f"Target PK: {', '.join(pk_cols)}")

# -----------------------------
# Audit + GUID defaults
# -----------------------------
LOADED_BY = "Perry M Stokes"
who = (LOADED_BY or "").replace("'", "''")

AUDIT_DEFAULTS_INSERT = {
    "ROW_CREATED_BY":     lambda: f"N'{who}'",
    "ROW_CHANGED_BY":     lambda: f"N'{who}'",
    "ROW_CREATED_DATE":   lambda: "SYSUTCDATETIME()",
    "ROW_CHANGED_DATE":   lambda: "SYSUTCDATETIME()",
    "ROW_EFFECTIVE_DATE": lambda: "SYSUTCDATETIME()",
}
AUDIT_DEFAULTS_UPDATE = {
    "ROW_CHANGED_BY":   lambda: f"N'{who}'",
    "ROW_CHANGED_DATE": lambda: "SYSUTCDATETIME()",
}

# -----------------------------
# Helper: derived-hash expressions (existing)
# -----------------------------
def _sql_trim(alias: str, col: str) -> str:
    return f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {alias}.{_qident(col)}))), N'')"

def _derived_expr(alias: str, inputs: list[str], out_len: int) -> str:
    parts = [f"COALESCE({_sql_trim(alias, c)}, N'')" for c in inputs] or ["N''"]
    concat = " + N'|' + ".join(parts)
    return (
        "LEFT(CONVERT(varchar(40), HASHBYTES('SHA1', CONVERT(varbinary(max), "
        f"{concat})), 2), {int(out_len)})"
    )

def choose_station_order_cols(view_cols: list[str]) -> list[str]:
    prefer = [
        # directional / station-ish
        "STATION_MD", "MD", "MEASURED_DEPTH", "DEPTH",
        "STATION_TVD", "STATION_TVDSS", "TVD",

        # strat-ish
        "TOP_MD", "BASE_MD",
        "TOP_TVD", "BASE_TVD",
        "TOP_DEPTH", "BASE_DEPTH",

        # universal fallback
        "RID",
    ]
    vu = {c.upper(): c for c in view_cols}
    return [vu[p] for p in prefer if p in vu]


    
# -----------------------------
# Station-table detection (helper)
# -----------------------------
station_mode = False
station_parent_schema = None
station_parent_table = None
station_parent_keys = []
station_seq_col = None
station_order_cols = []

station_sig = fetch_station_signature(conn, schema=tgt_schema, table=tgt_table)


station_dec = plan_station_mode(
    signature=station_sig,
    view_cols=view_cols,
    pk_cols=pk_cols,
)


st.caption(f"Station mode: {'ON' if station_dec.enabled else 'OFF'} — {station_dec.reason}")

if station_sig:
    station_parent_schema = station_sig.parent_schema
    station_parent_table  = station_sig.parent_table
    station_parent_keys   = station_sig.parent_keys or []
    station_seq_col       = station_sig.seq_col
    station_order_cols    = station_dec.order_cols or []

station_mode = bool(station_dec.enabled)

st.caption(f"Station mode: {'ON' if station_dec.enabled else 'OFF'} — {station_dec.reason}")
# -----------------------------
# Derived key support (hash-based registry)
# -----------------------------
tkey = f"{tgt_schema}.{tgt_table}".lower()
derived_cfg = DERIVED_PK_REGISTRY.get(tkey, {})
derived_map = (derived_cfg.get("derived", {}) or {})

derived_cols_active: dict[str, dict] = {}
for dcol, cfg in derived_map.items():
    dcol_u = dcol.upper()
    if dcol_u in tgt_cols_u and dcol in pk_cols:
        derived_cols_active[dcol] = cfg

# If station mode, NEVER hash-derive the station seq col (we generate it with ROW_NUMBER)
if station_mode and station_seq_col:
    derived_cols_active.pop(station_seq_col, None)

derived_u = {c.upper() for c in derived_cols_active.keys()}

if derived_cols_active:
    st.caption("Derived PK enabled: " + ", ".join(derived_cols_active.keys()))
else:
    st.caption("Derived PK: none (or not configured).")

# Validate derived inputs exist in view
for dcol, cfg in derived_cols_active.items():
    missing_inputs = [c for c in cfg.get("inputs", []) if c.upper() not in view_cols_u]
    if missing_inputs:
        st.error(
            f"Derived column {dcol} configured with inputs not in NORM view: {missing_inputs}. "
            "Fix mapping / rebuild view or adjust DERIVED_PK_REGISTRY."
        )
        st.stop()

# CROSS APPLY for derived cols (non-station mode uses h.<col>; station mode inlines into src)
cross_apply_sql = ""
derived_select_cols_sql = ""  # only used in station CTE so derived cols are visible as columns
if derived_cols_active:
    bits = []
    sel_bits = []
    for dcol, cfg in derived_cols_active.items():
        expr = _derived_expr("v", cfg.get("inputs", []), int(cfg.get("len", 20)))
        bits.append(f"{expr} AS {_qident(dcol)}")
        sel_bits.append(f"h.{_qident(dcol)} AS {_qident(dcol)}")
    cross_apply_sql = "CROSS APPLY (SELECT " + ", ".join(bits) + ") h"
    derived_select_cols_sql = ", " + ", ".join(sel_bits)

# -----------------------------
# Build final insert column list:
# - start with insertable_from_view
# - ensure ALL PK cols exist via view, derived, or station seq
# - add PPDM_GUID + audit cols
# -----------------------------
final_cols = list(insertable_from_view)

for pk in pk_cols:
    pku = pk.upper()

    if station_mode and station_seq_col and pku == station_seq_col.upper():
        if pk not in final_cols:
            final_cols.append(pk)
        continue

    if pku in view_cols_u and pku in tgt_cols_u:
        col_real = view_cols_u[pku]
        if col_real not in final_cols:
            final_cols.append(col_real)
        continue

    if pku in derived_u:
        if pk not in final_cols:
            final_cols.append(pk)
        continue

    st.error(
        f"PK column '{pk}' is not available in NORM view and not derived (and not station-sequenced). "
        f"Cannot safely promote into {target_fqn}."
    )
    st.stop()

if "PPDM_GUID" in tgt_cols_u and "PPDM_GUID" not in {c.upper() for c in final_cols}:
    final_cols.append("PPDM_GUID")

for a in ["ROW_CREATED_BY", "ROW_CREATED_DATE", "ROW_CHANGED_BY", "ROW_CHANGED_DATE", "ROW_EFFECTIVE_DATE"]:
    if a in tgt_cols_u and a not in {c.upper() for c in final_cols}:
        final_cols.append(a)

st.caption(f"Final column list (used for INSERT; MERGE updates only non-PK): {len(final_cols)}")
with st.expander("Final columns", expanded=False):
    st.code(", ".join(final_cols))

# -----------------------------
# SELECT expression per column (INSERT)
# -----------------------------
def _select_expr_insert(col: str) -> str:
    u = col.upper()

    # station seq col comes from src v.[seq_col]
    if station_mode and station_seq_col and u == station_seq_col.upper():
        return f"v.{_qident(station_seq_col)}"

    # derived hash cols:
    if u in derived_u:
        # non-station: CROSS APPLY alias is h
        if not station_mode:
            return f"h.{_qident(col)}"
        # station: derived cols were projected into src as real columns
        return f"v.{_qident(col)}"

    if u in AUDIT_DEFAULTS_INSERT:
        return AUDIT_DEFAULTS_INSERT[u]()
    if u == "PPDM_GUID":
        return "CONVERT(nvarchar(36), NEWID())"
    if u in view_cols_u:
        return f"v.{_qident(view_cols_u[u])}"
    return "NULL"

# -----------------------------
# PK predicates + EXISTS predicates
# -----------------------------
pk_not_null_preds = []
exists_preds = []

for pk in pk_cols:
    pku = pk.upper()

    if station_mode and station_seq_col and pku == station_seq_col.upper():
        pk_not_null_preds.append(f"v.{_qident(station_seq_col)} IS NOT NULL")
        exists_preds.append(f"t.{_qident(pk)} = v.{_qident(station_seq_col)}")
        continue

    if pku in derived_u:
        if station_mode:
            pk_not_null_preds.append(f"v.{_qident(pk)} IS NOT NULL")
            exists_preds.append(f"t.{_qident(pk)} = v.{_qident(pk)}")
        else:
            pk_not_null_preds.append(f"h.{_qident(pk)} IS NOT NULL")
            exists_preds.append(f"t.{_qident(pk)} = h.{_qident(pk)}")
        continue

    pk_not_null_preds.append(f"v.{_qident(pk)} IS NOT NULL")
    exists_preds.append(f"t.{_qident(pk)} = v.{_qident(pk)}")

pk_not_null_sql = " AND ".join(pk_not_null_preds)
exists_sql = " AND ".join(exists_preds)

# -----------------------------
# Build column SQL (INSERT)
# -----------------------------
col_sql_insert = ", ".join(_qident(c) for c in final_cols)
sel_sql_insert = ", ".join(f"{_select_expr_insert(c)} AS {_qident(c)}" for c in final_cols)

# -----------------------------
# Station CTE (optional)
# - generates station_seq_col with ROW_NUMBER()
# - gates on parent existence to prevent FK errors
# - optionally projects derived hash cols into src as real columns
# -----------------------------
station_cte_sql = ""
from_sql_insert = f"FROM {norm_view} v\n{cross_apply_sql}".rstrip()

if station_mode and station_seq_col:
    parent_full = f"[{station_parent_schema}].[{station_parent_table}]"
    parent_keys_sql = ", ".join([f"v.{_qident(k)}" for k in station_parent_keys])

    # ORDER BY: use best station axis if possible, else tie-break stable-ish
    if station_order_cols:
        first = station_order_cols[0]
        rest  = station_order_cols[1:]
        order_parts = [
            f"CASE WHEN TRY_CONVERT(float, v.{_qident(first)}) IS NULL THEN 1 ELSE 0 END",
            f"TRY_CONVERT(float, v.{_qident(first)})",
        ] + [f"v.{_qident(c)}" for c in rest]
        order_sql = ", ".join(order_parts)
    else:
        # fallback to first parent key (last resort)
        order_sql = f"v.{_qident(station_parent_keys[0])}"

    parent_exists_sql = " AND ".join([f"h.{_qident(k)} = v.{_qident(k)}" for k in station_parent_keys])
    parent_not_null_sql = " AND ".join([f"v.{_qident(k)} IS NOT NULL" for k in station_parent_keys])

    station_cte_sql = f"""
;WITH src AS (
    SELECT
        v.*{derived_select_cols_sql},
        CAST(ROW_NUMBER() OVER (
            PARTITION BY {parent_keys_sql}
            ORDER BY {order_sql}
        ) AS int) AS {_qident(station_seq_col)}
    FROM {norm_view} v
    {cross_apply_sql}
    WHERE {parent_not_null_sql}
      AND EXISTS (
          SELECT 1
          FROM {parent_full} h
          WHERE {parent_exists_sql}
      )
)
""".strip()

    # In station mode, outer queries read FROM src v (no CROSS APPLY outside)
    from_sql_insert = "FROM src v"

# ============================================================
# INSERT-ONLY SQL
# ============================================================
sql_count_insert = f"""
SET NOCOUNT ON;
{station_cte_sql}
SELECT COUNT(*) AS would_insert
{from_sql_insert}
WHERE {pk_not_null_sql}
  AND NOT EXISTS (
      SELECT 1
      FROM {target_full} t
      WHERE {exists_sql}
  );
""".strip()

sql_insert = f"""
SET NOCOUNT ON;
BEGIN TRAN;

{station_cte_sql}
INSERT INTO {target_full} ({col_sql_insert})
SELECT {sel_sql_insert}
{from_sql_insert}
WHERE {pk_not_null_sql}
  AND NOT EXISTS (
      SELECT 1
      FROM {target_full} t
      WHERE {exists_sql}
  );

SELECT @@ROWCOUNT AS rows_inserted;

COMMIT TRAN;
""".strip()

# ============================================================
# MERGE/UPDATE SQL (FIXED aliasing)
# - ON clause references only t.<col> and src.<col>
# - derived cols surfaced as src.<derived_col>
# - station seq col surfaced as src.<seq_col> via station CTE
# ============================================================

# Build a USING projection that includes:
#   1) all view columns (v.<col>) as themselves
#   2) derived cols (non-station: computed via expr; station: already present in src)
#   3) station seq col (station: present in src)
using_proj = []
for c in view_cols:
    using_proj.append(f"v.{_qident(c)} AS {_qident(c)}")

# Derived cols: if non-station, compute expressions in USING; if station, they already exist in src
if derived_cols_active:
    if station_mode:
        for dcol in derived_cols_active.keys():
            using_proj.append(f"v.{_qident(dcol)} AS {_qident(dcol)}")
    else:
        for dcol, cfg in derived_cols_active.items():
            expr = _derived_expr("v", cfg.get("inputs", []), int(cfg.get("len", 20)))
            using_proj.append(f"{expr} AS {_qident(dcol)}")

# Station seq col
if station_mode and station_seq_col:
    using_proj.append(f"v.{_qident(station_seq_col)} AS {_qident(station_seq_col)}")

using_proj_sql = ",\n        ".join(using_proj)

# USING source: from norm_view normally; from src in station mode
using_prefix = ""
using_from = f"FROM {norm_view} v\n{cross_apply_sql}".rstrip()

if station_mode and station_seq_col:
    using_prefix = station_cte_sql
    using_from = "FROM src v"  # v is the src alias here; derived cols + seq col exist

using_sql = f"""
{using_prefix}
(
    SELECT
        {using_proj_sql}
    {using_from}
    WHERE {pk_not_null_sql}
) AS src
""".strip()

# ON clause: match on PK columns
on_parts = []
for pk in pk_cols:
    on_parts.append(f"t.{_qident(pk)} = src.{_qident(pk)}")
on_sql_merge = " AND ".join(on_parts)

pk_u = {p.upper() for p in pk_cols}

def _src_expr(col: str) -> str:
    u = col.upper()
    if u in AUDIT_DEFAULTS_UPDATE:
        return AUDIT_DEFAULTS_UPDATE[u]()
    if u == "PPDM_GUID":
        return f"t.{_qident(col)}"
    return f"src.{_qident(col)}"

def _is_blank(expr: str) -> str:
    return f"({expr} IS NULL OR NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {expr}))), N'') IS NULL)"

def _has_value(expr: str) -> str:
    return f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), {expr}))), N'') IS NOT NULL"

updatable_cols = [c for c in insertable_from_view if c.upper() not in pk_u and c.upper() != "RID"]
for a in ["ROW_CHANGED_BY", "ROW_CHANGED_DATE"]:
    if a in tgt_cols_u and a not in {c.upper() for c in updatable_cols}:
        updatable_cols.append(a)

set_lines = []
for c in updatable_cols:
    u = c.upper()
    tgt_expr = f"t.{_qident(c)}"
    src_expr = _src_expr(c)

    if u in {"ROW_CHANGED_BY", "ROW_CHANGED_DATE"}:
        set_lines.append(f"{tgt_expr} = {src_expr}")
        continue

    src_has = _has_value(src_expr)
    if update_style == "Fill-only":
        cond = f"{_is_blank(tgt_expr)} AND {src_has}"
        set_lines.append(f"{tgt_expr} = CASE WHEN {cond} THEN {src_expr} ELSE {tgt_expr} END")
    else:
        set_lines.append(f"{tgt_expr} = CASE WHEN {src_has} THEN {src_expr} ELSE {tgt_expr} END")

set_sql = ",\n    ".join(set_lines) if set_lines else f"t.{_qident(pk_cols[0])} = t.{_qident(pk_cols[0])}"

def _insert_value_expr(col: str) -> str:
    u = col.upper()
    if u in AUDIT_DEFAULTS_INSERT:
        return AUDIT_DEFAULTS_INSERT[u]()
    if u == "PPDM_GUID":
        return "CONVERT(nvarchar(36), NEWID())"
    return f"src.{_qident(col)}"

sql_preview_merge = f"""
SET NOCOUNT ON;
SELECT
  SUM(CASE WHEN EXISTS (
      SELECT 1 FROM {target_full} t WHERE {" AND ".join([f"t.{_qident(pk)} = src.{_qident(pk)}" for pk in pk_cols])}
  ) THEN 1 ELSE 0 END) AS would_match,
  SUM(CASE WHEN NOT EXISTS (
      SELECT 1 FROM {target_full} t WHERE {" AND ".join([f"t.{_qident(pk)} = src.{_qident(pk)}" for pk in pk_cols])}
  ) THEN 1 ELSE 0 END) AS missing_in_target
FROM {using_sql};
""".strip()

sql_merge = f"""
SET NOCOUNT ON;

MERGE {target_full} AS t
USING {using_sql}
ON {on_sql_merge}
WHEN MATCHED THEN
    UPDATE SET
    {set_sql}
WHEN NOT MATCHED BY TARGET THEN
    INSERT ({col_sql_insert})
    VALUES ({", ".join([_insert_value_expr(c) for c in final_cols])});
""".strip()

# -----------------------------
# Buttons + execution
# -----------------------------
c1, c2 = st.columns([1, 1])

if promote_mode == "Insert new only":
    if c1.button("Preview would-insert count", key="promote_preview_insert"):
        try:
            df = db.read_sql(conn, sql_count_insert)
            st.dataframe(_safe_df(df), hide_index=True, width="stretch")
        except Exception as e:
            st.error(f"Count failed: {e}")

    if c2.button("Promote now (insert new only)", type="primary", key="promote_do_insert"):
        try:
            db.exec_sql(conn, sql_insert)
            st.success("Insert promote completed.")
        except Exception as e:
            st.error(f"Promote failed: {e}")

    with st.expander("SQL (insert-only)", expanded=False):
        st.code(sql_count_insert, language="sql")
        st.code(sql_insert, language="sql")

else:
    if c1.button("Preview match vs missing", key="promote_preview_merge"):
        try:
            df = db.read_sql(conn, sql_preview_merge)
            st.dataframe(_safe_df(df), hide_index=True, width="stretch")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    if c2.button("Promote now (merge/update)", type="primary", key="promote_do_merge"):
        try:
            db.exec_sql(conn, sql_merge)
            st.success("Merge/update promote completed.")
        except Exception as e:
            st.error(f"Promote failed: {e}")

    with st.expander("SQL (merge/update)", expanded=False):
        st.code(sql_preview_merge, language="sql")
        st.code(sql_merge, language="sql")



