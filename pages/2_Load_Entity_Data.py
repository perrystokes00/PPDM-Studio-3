# pages/2_Load_Entity_Data.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import ppdm_loader.db as db
from ppdm_loader.stage import save_upload, stage_bulk_insert, DELIM_MAP
from ppdm_loader.normalize import build_primary_norm_view_sql
from ppdm_loader.fk_introspect import introspect_fk_by_child_col
from ppdm_loader.fk_suggest import suggest_fk_candidates_step4

from common.ui import sidebar_connect, require_connection
from ppdm_loader.child_insert import fetch_fk_map, fetch_pk_columns, build_insert_new_sql_generic


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

        prev = db.read_sql(conn, "SELECT TOP (25) * FROM stg.v_raw_with_rid ORDER BY RID;")
        st.subheader("stg.v_raw_with_rid (top 25)")
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
        st.code(view_name)

        prev = db.read_sql(conn, f"SELECT TOP (25) * FROM {view_name} ORDER BY RID;")
        st.subheader("NORM view preview (top 25)")
        st.dataframe(_safe_df(prev), hide_index=True, width="stretch")

    except Exception as e:
        st.error(f"Build NORM view failed: {e}")

norm_view = st.session_state.get("norm_view_name")
if not norm_view:
    st.info("Build the NORM view above to enable Step 6/7.")
    st.stop()


# ============================================================
# STEP 6 — FK QC (missing reference values) (pass-through friendly)
# ============================================================
st.header("Step 6 — FK QC (missing reference values)")

primary_map_df = st.session_state.get("primary_map_df")
treat_fk_cols = (
    primary_map_df.loc[primary_map_df.get("treat_as_fk") == True, "column_name"]
    .astype(str)
    .tolist()
)
treat_fk_cols = [c for c in treat_fk_cols if c.strip()]

cA, cB = st.columns([2, 2])
with cA:
    top_n = st.number_input("Show top N missing", 10, 50000, 2000, 100, key="step6_topn")
with cB:
    st.caption("QC only. (Seeding stays off by default.)")

if not treat_fk_cols:
    st.info("No FK columns marked. Tick 'Treat as FK' in Step 4 if you want FK QC.")
else:
    for child_fk_col in treat_fk_cols:
        fkinfo = introspect_fk_by_child_col(
            conn,
            child_schema=child_schema,
            child_table=child_table,
            child_col=child_fk_col,
        )

        if fkinfo is None:
            with st.expander(f"⚠️ {child_fk_col} — No FK found in metadata", expanded=False):
                st.warning("This column is marked Treat-as-FK but SQL Server FK metadata did not find a relationship.")
            continue

        parent_label = f"{fkinfo.parent_schema}.{fkinfo.parent_table}"
        title = f"FK: {child_fk_col} → {parent_label}  ({fkinfo.fk_name})"

        with st.expander(title, expanded=False):
            st.dataframe(pd.DataFrame(fkinfo.pairs, columns=["child_col", "parent_col"]), hide_index=True, width="stretch")

            # NOTE: expects NORM view has <child_col>__NAT for FK-marked columns
            # If you hit "Invalid column name <X>__NAT", it means that column wasn't included in the view.
            # For now, just untick Treat-as-FK for that column (pass-through).
            parent_fqn = _qfqn(fkinfo.parent_schema, fkinfo.parent_table)

            proj_exprs = []
            req_preds = []
            join_preds = []
            order_cols = []

            for child_col, parent_col in fkinfo.pairs:
                nat_col = child_col + "__NAT"
                expr = f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_qident(nat_col)}))), N'')"
                proj_exprs.append(f"{expr} AS {_qident(parent_col)}")
                req_preds.append(f"{_qident(parent_col)} IS NOT NULL")
                order_cols.append(_qident(parent_col))
                join_preds.append(
                    f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{_qident(parent_col)}))), N'') = s.{_qident(parent_col)}"
                )

            proj_sql = ",\n        ".join(proj_exprs)
            req_sql = " AND ".join(req_preds)
            join_sql = " AND ".join(join_preds)
            order_sql = ", ".join(order_cols)

            sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_sql}
    FROM {norm_view} v
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
    FROM {norm_view} v
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

            c1, c2 = st.columns([1, 2])
            if c1.button("Compute missing", key=f"step6_missing_{child_fk_col}", type="primary"):
                try:
                    miss_df = db.read_sql(conn, sql_sample)
                    cnt_df = db.read_sql(conn, sql_count)
                    missing_total = int(cnt_df.iloc[0]["missing_total"]) if (cnt_df is not None and not cnt_df.empty) else 0

                    st.dataframe(_safe_df(miss_df), hide_index=True, width="stretch")
                    st.caption(f"Missing shown: {len(miss_df)} | Missing total: {missing_total}")
                except Exception as e:
                    st.error(f"Missing-QC failed: {e}")
                    st.info("If error mentions __NAT, untick Treat-as-FK for that column and rebuild the NORM view.")

            with c2.expander("SQL", expanded=False):
                st.code(sql_sample, language="sql")
                st.code(sql_count, language="sql")

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
            "DEPTH_OBS_NO": {
                "len": 20,
                "inputs": ["UWI", "SURVEY_ID", "SOURCE", "STATION_ID", "STATION_MD"],
            }
        }
    },
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
# INSERT-ONLY SQL (current behavior)
# ============================================================
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
""".strip()

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


# -----------------------------
# Buttons
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


