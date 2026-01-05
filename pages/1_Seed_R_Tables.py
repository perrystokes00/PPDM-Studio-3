# pages/1_Seed_R_Tables.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

import ppdm_loader.db as db
from common.ui import sidebar_connect, require_connection
from ppdm_loader.seed_generic import (
    MapRow,
    fetch_table_columns,
    preview_missing_by_pk,
    build_src_frame_from_mapping,
    seed_missing_rows,
)

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Seed R Tables", layout="wide")
sidebar_connect(page_prefix="seedr")
conn = require_connection()

DEFAULT_LOADED_BY = "Perry M Stokes"

# Default registry paths (adjust if yours differ)
REG_PPDM39 = Path(r"C:\Users\perry\OneDrive\Documents\PPDM\ETL-4\schema_registry\catalog\ppdm_39_schema_domain.json")
REG_LITE = Path(r"C:\Users\perry\OneDrive\Documents\PPDM\ETL-4\schema_registry\catalog\ppdm_lite_schema_catalog.json")


# ============================================================
# Small helpers
# ============================================================
def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if (df is not None and isinstance(df, pd.DataFrame) and not df.empty) else pd.DataFrame()


def _db_name(conn) -> str:
    try:
        d = db.read_sql(conn, "SELECT DB_NAME() AS db;")
        if d is not None and not d.empty:
            return str(d.iloc[0]["db"])
    except Exception:
        pass
    return ""


def _pick_registry_path() -> Path:
    model = (st.session_state.get("ppdm_model") or st.session_state.get("ppdm_version") or "").lower()
    if "lite" in model:
        return REG_LITE
    return REG_PPDM39


@st.cache_data(show_spinner=False)
def _load_registry_df(json_path: str) -> pd.DataFrame:
    p = Path(json_path)
    if not p.exists():
        return pd.DataFrame(columns=["schema", "table", "column", "is_pk"])

    obj = json.loads(p.read_text(encoding="utf-8"))

    # Accept both shapes:
    #  (A) wrapper dict: { "ppdm_39_schema_domain": [rows] } etc.
    #  (B) raw list: [rows]
    if isinstance(obj, dict):
        data = (
            obj.get("ppdm_39_schema_domain")
            or obj.get("ppdm_lite_schema_domain")
            or obj.get("schema_domain")
            or obj.get("rows")
            or obj.get("data")
            or []
        )
    elif isinstance(obj, list):
        data = obj
    else:
        data = []

    rows: list[dict[str, Any]] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                "schema": (r.get("table_schema") or "dbo"),
                "table": r.get("table_name") or "",
                "column": r.get("column_name") or "",
                "is_pk": str(r.get("is_primary_key") or "").strip().upper() in {"YES", "Y", "TRUE", "1"},
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["schema", "table", "column", "is_pk"])

    df = df[df["table"].astype(str).str.len() > 0]
    df = df[df["column"].astype(str).str.len() > 0]
    return df


@st.cache_data(show_spinner=False)
def _list_r_tables(reg: pd.DataFrame) -> list[str]:
    t = reg[["schema", "table"]].drop_duplicates()
    tl = t["table"].astype(str).str.lower()
    t = t[tl.str.startswith("r_") | tl.str.startswith("ra_")]
    out = [f"{r.schema}.{r.table}" for r in t.itertuples(index=False)]
    return sorted(out, key=lambda s: s.lower())


@st.cache_data(show_spinner=False)
def _table_columns_from_reg(reg: pd.DataFrame, schema: str, table: str) -> list[str]:
    df = reg[(reg["schema"] == schema) & (reg["table"] == table)]
    return df["column"].astype(str).tolist()


@st.cache_data(show_spinner=False)
def _pk_columns_from_reg(reg: pd.DataFrame, schema: str, table: str) -> list[str]:
    df = reg[(reg["schema"] == schema) & (reg["table"] == table) & (reg["is_pk"] == True)]
    return df["column"].astype(str).tolist()


def _read_uploaded_table(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = (uploaded.name or "").lower()

    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded, dtype=str).fillna("")
        # default csv/txt
        return pd.read_csv(uploaded, dtype=str, keep_default_na=False, engine="python").fillna("")
    except Exception as e:
        st.error(f"Could not read upload: {e}")
        return pd.DataFrame()


def _load_seed_json(json_file) -> dict[str, Any]:
    if json_file is None:
        return {}
    try:
        raw = json_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj if isinstance(obj, (dict, list)) else {}
    except Exception as e:
        st.error(f"Could not parse seed JSON: {e}")
        return {}


def _extract_rows_for_table(seed_obj: Any, target_fqn: str) -> list[dict[str, Any]]:
    """
    Accepts:
      - dict mapping table->rows
      - dict wrapper with "tables": {...}
      - list of rows (assumes it is already for the table)
    Tries keys: exact "dbo.r_x", "r_x", case-insensitive.
    """
    if seed_obj is None:
        return []

    if isinstance(seed_obj, list):
        # assume list already rows
        rows = [r for r in seed_obj if isinstance(r, dict)]
        return rows

    if not isinstance(seed_obj, dict):
        return []

    # unwrap common wrappers
    if "tables" in seed_obj and isinstance(seed_obj["tables"], dict):
        seed_obj = seed_obj["tables"]

    want = target_fqn.strip()
    want_l = want.lower()
    want_no_schema = want.split(".", 1)[-1].lower()

    # direct matches
    for k in (want, want_l, want_no_schema):
        if k in seed_obj and isinstance(seed_obj[k], list):
            return [r for r in seed_obj[k] if isinstance(r, dict)]

    # case-insensitive search
    for k, v in seed_obj.items():
        if not isinstance(k, str):
            continue
        kl = k.lower().strip()
        if kl == want_l or kl == want_no_schema:
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]

    return []


# ============================================================
# UI
# ============================================================
st.title("Seed R Tables (fast registry + JSON seed option)")

with st.expander("🔎 Debug: connection sanity", expanded=False):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(_safe_df(who), hide_index=True, width="stretch")

# ---- registry picker ----
cR1, cR2 = st.columns([2, 3])
with cR1:
    reg_default = _pick_registry_path()
    reg_path_txt = st.text_input("Schema registry JSON path", value=str(reg_default), key="seedr_reg_path")
with cR2:
    st.caption("Registry drives table list + PK detection (fast).")

reg_df = _load_registry_df(reg_path_txt)
if reg_df.empty:
    st.error(f"Registry not found/empty: {reg_path_txt}")
    st.stop()

r_tables = _list_r_tables(reg_df)
if not r_tables:
    st.error("No r_/ra_ tables found in registry.")
    st.stop()

# ---- select target table ----
dbn = _db_name(conn)
target_fqn = st.selectbox("Target R/RA table", r_tables, key="seedr_target_fqn")

# reset derived state when target changes (fixes stale UI)
ss = st.session_state
target_sig = {"target_fqn": target_fqn, "db": dbn}
if ss.get("seedr_target_sig") != target_sig:
    ss["seedr_target_sig"] = target_sig
    for k in ["seedr_src_df", "seedr_map_df", "seedr_missing_df", "seedr_missing_total", "seedr_insert_df"]:
        ss.pop(k, None)

t_schema, t_table = target_fqn.split(".", 1)
t_cols = _table_columns_from_reg(reg_df, t_schema, t_table)
pk_cols = _pk_columns_from_reg(reg_df, t_schema, t_table)

cA, cB = st.columns([2, 3])
with cA:
    st.caption("Target PK (from registry)")
    st.code(", ".join(pk_cols) if pk_cols else "(none detected)")
with cB:
    st.caption("Target columns (from registry)")
    st.code(", ".join(t_cols) if t_cols else "(none)")

if not pk_cols:
    st.error("PK not detected for this table in the registry JSON. Seeding requires PK for safe missing checks.")
    st.stop()

st.divider()

# ============================================================
# STEP 1 — Load source (JSON seed OR file upload)
# ============================================================
st.header("Step 1 — Provide source rows (JSON seed OR upload file)")

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Option A — Load from JSON seed file (recommended)")
    seed_json_up = st.file_uploader(
        "Seed JSON file",
        type=["json"],
        key="seedr_seed_json_upload",
        help="Expected: { 'dbo.r_x': [ {...}, ... ] } or { 'tables': { 'dbo.r_x': [...] } }",
    )
    if st.button("Use seed JSON for this table", type="primary", key="seedr_use_seed_json"):
        seed_obj = _load_seed_json(seed_json_up)
        rows = _extract_rows_for_table(seed_obj, target_fqn)
        if not rows:
            st.error(f"No rows found in seed JSON for key '{target_fqn}' (or '{t_table}').")
        else:
            df = pd.DataFrame(rows)
            # normalize to strings
            for col in df.columns:
                df[col] = df[col].astype(str)
            ss["seedr_src_df"] = df
            st.success(f"Loaded {len(df)} rows from seed JSON for {target_fqn}.")

with c2:
    st.subheader("Option B — Upload CSV/TXT/XLSX (fallback)")
    up = st.file_uploader("Drop file here", type=["csv", "txt", "xlsx", "xls"], key="seedr_src_upload")
    if st.button("Use uploaded file", key="seedr_use_upload", type="secondary"):
        df = _read_uploaded_table(up)
        if df.empty:
            st.error("No rows read from upload.")
        else:
            ss["seedr_src_df"] = df
            st.success(f"Loaded {len(df)} rows from upload.")

src_df = ss.get("seedr_src_df")
if src_df is None or not isinstance(src_df, pd.DataFrame) or src_df.empty:
    st.info("Load source rows using Option A or Option B to continue.")
    st.stop()

st.caption("Source preview")
st.dataframe(src_df.head(25), hide_index=True, width="stretch")

src_cols = [str(c) for c in src_df.columns.tolist()]

st.divider()

# ============================================================
# STEP 2 — Mapping grid (target cols ← source cols)
# ============================================================
st.header("Step 2 — Map source columns → target columns")

# build/refresh default mapping if needed
map_sig = {
    "target_fqn": target_fqn,
    "src_cols": tuple([c.upper() for c in src_cols]),
}
if ss.get("seedr_map_sig") != map_sig or "seedr_map_df" not in ss:
    src_u = {c.upper(): c for c in src_cols}
    rows = []
    for tgt in t_cols:
        guess = src_u.get(str(tgt).upper(), "")
        rows.append(
            {
                "use": True if (guess or str(tgt).upper() in {p.upper() for p in pk_cols}) else False,
                "target_column": tgt,
                "source_column": guess,
                "constant_value": "",
                "transform": "trim",
            }
        )
    ss["seedr_map_df"] = pd.DataFrame(rows)[
        ["use", "target_column", "source_column", "constant_value", "transform"]
    ]
    ss["seedr_map_sig"] = map_sig

with st.form("seedr_map_form", clear_on_submit=False, border=True):
    edited = st.data_editor(
        ss["seedr_map_df"],
        width="stretch",
        num_rows="fixed",
        column_config={
            "use": st.column_config.CheckboxColumn("Use"),
            "target_column": st.column_config.TextColumn("Target column", disabled=True),
            "source_column": st.column_config.SelectboxColumn(
                "Source column (dropdown)", options=[""] + src_cols
            ),
            "constant_value": st.column_config.TextColumn("Constant (optional)"),
            "transform": st.column_config.SelectboxColumn("Transform", options=["none", "trim", "upper"]),
        },
        key="seedr_map_editor",
    )
    apply_map = st.form_submit_button("Apply mapping", type="primary")

if apply_map:
    ss["seedr_map_df"] = edited.copy()
    st.success("Mapping saved.")

map_df = ss["seedr_map_df"]

st.divider()

# ============================================================
# STEP 3 — Build insert_df from mapping
# ============================================================
st.header("Step 3 — Build insert set (from mapping)")

mapping: list[MapRow] = []
for r in map_df.to_dict(orient="records"):
    mapping.append(
        MapRow(
            target_column=str(r.get("target_column") or "").strip(),
            use=bool(r.get("use", False)),
            source_column=str(r.get("source_column") or "").strip(),
            constant_value=str(r.get("constant_value") or ""),
            transform=str(r.get("transform") or "trim"),
        )
    )

insert_df = build_src_frame_from_mapping(src_df, mapping)
if insert_df.empty:
    st.warning("Resulting insert_df is empty. Check mapping and source data.")
    st.stop()

ss["seedr_insert_df"] = insert_df
st.caption("Insert DF preview (first 25)")
st.dataframe(insert_df.head(25), hide_index=True, width="stretch")

missing_pk = [c for c in pk_cols if c not in insert_df.columns]
if missing_pk:
    st.error(f"Insert DF is missing PK column(s): {missing_pk}. Map them before seeding.")
    st.stop()

st.divider()

# ============================================================
# STEP 4 — Preview missing (by PK)
# ============================================================
st.header("Step 4 — Preview missing PK rows (what will be inserted)")

top_n = st.number_input("Show top N missing", 10, 50000, 500, 50, key="seedr_topn")

if st.button("Compute missing", type="primary", key="seedr_compute_missing"):
    try:
        miss_df, miss_total = preview_missing_by_pk(
            conn,
            target_schema=t_schema,
            target_table=t_table,
            pk_cols=pk_cols,
            src_df=insert_df,
            top_n=int(top_n),
        )
        ss["seedr_missing_df"] = miss_df
        ss["seedr_missing_total"] = miss_total
    except Exception as e:
        st.error(f"Missing preview failed: {e}")

miss_df = ss.get("seedr_missing_df")
miss_total = ss.get("seedr_missing_total")

if isinstance(miss_total, int):
    st.caption(f"Missing total: {miss_total}")

if isinstance(miss_df, pd.DataFrame) and not miss_df.empty:
    st.dataframe(miss_df, hide_index=True, width="stretch")

st.divider()

# ============================================================
# STEP 5 — Seed missing (insert only missing)
# ============================================================
st.header("Step 5 — Seed missing rows")

loaded_by = st.text_input("Loaded by", value=DEFAULT_LOADED_BY, key="seedr_loaded_by")

if st.button("Seed missing now", type="primary", key="seedr_seed_now"):
    try:
        # IMPORTANT: seed_generic checks actual target columns (so no bogus PPDM_GUID/audit cols)
        inserted = seed_missing_rows(
            conn,
            target_schema=t_schema,
            target_table=t_table,
            pk_cols=pk_cols,
            insert_df=insert_df,
            loaded_by=loaded_by,
        )
        st.success(f"Seed completed. Inserted rows: {inserted}")
        # refresh missing preview after insert
        ss.pop("seedr_missing_df", None)
        ss.pop("seedr_missing_total", None)
    except Exception as e:
        st.error(f"Seed failed: {e}")

with st.expander("Notes", expanded=False):
    st.write(
        "- This page uses the schema-registry JSON for fast table+PK detection.\n"
        "- Seeding inserts **only missing PK tuples** (safe, idempotent).\n"
        "- PPDM_GUID/audit columns are only injected if they exist on the actual target table."
    )
