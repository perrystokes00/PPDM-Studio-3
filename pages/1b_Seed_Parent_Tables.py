# pages/1b_Seed_Parent_Tables.py
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import ppdm_loader.db as db
from ppdm_loader.seed_generic import (
    MapRow,
    fetch_pk_columns,
    fetch_table_columns,
    build_src_frame_from_mapping,
    preview_missing_by_pk,
    seed_missing_rows,
)

from common.ui import sidebar_connect, require_connection


st.set_page_config(page_title="Seed Parent Tables (Generic)", layout="wide")

sidebar_connect(page_prefix="seedp")
conn = require_connection()

st.title("Seed Parent Tables (Generic, composite PK aware)")
st.caption("Use this when parent tables have composite keys (zone, strat_unit, etc.). No silent inserts.")


with st.expander("🔎 Debug: connection sanity", expanded=False):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(who, hide_index=True, width="stretch")


# ------------------------------------------------------------
# Source: upload file
# ------------------------------------------------------------
st.header("Step 1 — Source file")
up = st.file_uploader("Drop CSV/TXT here", type=["csv", "txt"], key="seedp_upload")
sep = st.selectbox("Delimiter", [",", "\t", "|", ";"], index=0, key="seedp_sep")

if not up:
    st.stop()

df = pd.read_csv(up, sep=sep, dtype=str, keep_default_na=False)
df.columns = [str(c).strip().upper() for c in df.columns]

st.subheader("Preview")
st.dataframe(df.head(50), hide_index=True, width="stretch")
st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")


# ------------------------------------------------------------
# Target table selection (simple text input for MVP)
# ------------------------------------------------------------
st.header("Step 2 — Target parent table")
target_fqn = st.text_input("Target table (schema.table)", value="dbo.zone", key="seedp_target")
if "." not in target_fqn:
    st.error("Enter like dbo.zone")
    st.stop()

t_schema, t_table = target_fqn.split(".", 1)

pk_cols = fetch_pk_columns(conn, t_schema, t_table)
t_cols = fetch_table_columns(conn, t_schema, t_table)

if not t_cols:
    st.error(f"Could not read columns for {target_fqn}")
    st.stop()

if not pk_cols:
    st.error("No PK detected. MVP requires a primary key (composite OK).")
    st.stop()

st.success(f"PK columns: {', '.join(pk_cols)}")
st.caption(f"Target columns: {len(t_cols)}")


# ------------------------------------------------------------
# Mapping grid
# ------------------------------------------------------------
st.header("Step 3 — Mapping")
st.caption("Map source file columns → target columns (use source or constant). PK columns must be populated.")

# default mapping rows: include PK cols + a few common columns if present
default_targets = []
for c in pk_cols:
    default_targets.append(c)
for c in ["ZONE_NAME", "LONG_NAME", "SHORT_NAME", "ACTIVE_IND", "REMARK", "SOURCE_DOCUMENT_ID"]:
    if c in t_cols and c not in default_targets:
        default_targets.append(c)

# build initial df once
ss = st.session_state
sig = (target_fqn, tuple(df.columns), tuple(pk_cols))
if ss.get("seedp_sig") != sig:
    rows = []
    src_u = {c.upper(): c for c in df.columns}
    for tgt in default_targets:
        guess = src_u.get(tgt.upper(), "")
        rows.append(
            dict(
                use=True,
                target_column=tgt,
                source_column=guess,
                constant_value="",
                transform="trim",
            )
        )
    ss["seedp_map_df"] = pd.DataFrame(rows)
    ss["seedp_sig"] = sig

map_df = ss["seedp_map_df"]

with st.form("seedp_map_form", clear_on_submit=False, border=True):
    edited = st.data_editor(
        map_df,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "use": st.column_config.CheckboxColumn("Use"),
            "target_column": st.column_config.TextColumn("Target column", disabled=True),
            "source_column": st.column_config.SelectboxColumn("Source column", options=[""] + df.columns.tolist()),
            "constant_value": st.column_config.TextColumn("Constant"),
            "transform": st.column_config.SelectboxColumn("Transform", options=["none", "trim", "upper"]),
        },
        key="seedp_map_editor",
    )
    apply_btn = st.form_submit_button("Apply mapping", type="primary")

if apply_btn:
    ss["seedp_map_df"] = edited
    st.success("Mapping saved.")


# Validate PK coverage
edited = ss["seedp_map_df"]
mrows: list[MapRow] = []
for _, r in edited.iterrows():
    mrows.append(
        MapRow(
            target_column=str(r.get("target_column", "")).strip(),
            use=bool(r.get("use", False)),
            source_column=str(r.get("source_column", "")).strip(),
            constant_value=str(r.get("constant_value", "")).strip(),
            transform=str(r.get("transform", "none")).strip(),
        )
    )

# Build src df for PK check
src_df = build_src_frame_from_mapping(df, mrows)

missing_pk = [c for c in pk_cols if c not in src_df.columns]
if missing_pk:
    st.error(f"Mapping is missing PK columns: {', '.join(missing_pk)}")
    st.stop()


# ------------------------------------------------------------
# Missing preview + seed
# ------------------------------------------------------------
st.header("Step 4 — Preview missing + seed")
top_n = st.number_input("Show top N missing", min_value=10, max_value=50000, value=2000, step=100, key="seedp_topn")

c1, c2 = st.columns([1, 1])
compute = c1.button("Compute missing", type="primary", key="seedp_compute")
seed = c2.button("Seed missing now", type="secondary", key="seedp_seed")

if compute:
    miss_df, miss_total = preview_missing_by_pk(
        conn,
        target_schema=t_schema,
        target_table=t_table,
        pk_cols=pk_cols,
        src_df=src_df,
        top_n=int(top_n),
    )
    ss["seedp_missing_df"] = miss_df
    ss["seedp_missing_total"] = miss_total

miss_df = ss.get("seedp_missing_df")
miss_total = ss.get("seedp_missing_total")

if isinstance(miss_df, pd.DataFrame):
    st.subheader("Missing PK tuples")
    st.dataframe(miss_df, hide_index=True, width="stretch")
    st.caption(f"Missing shown: {len(miss_df):,} | Missing total: {int(miss_total or 0):,}")

loaded_by = st.text_input("Loaded by (audit columns)", value="Perry M Stokes", key="seedp_loaded_by")

if seed:
    inserted = seed_missing_rows(
        conn,
        target_schema=t_schema,
        target_table=t_table,
        pk_cols=pk_cols,
        insert_df=src_df,
        loaded_by=loaded_by,
    )
    st.success(f"Seed completed. Inserted {inserted:,} row(s).")
    # re-check missing
    miss_df2, miss_total2 = preview_missing_by_pk(
        conn,
        target_schema=t_schema,
        target_table=t_table,
        pk_cols=pk_cols,
        src_df=src_df,
        top_n=int(top_n),
    )
    ss["seedp_missing_df"] = miss_df2
    ss["seedp_missing_total"] = miss_total2
    st.subheader("Missing after seeding")
    st.dataframe(miss_df2, hide_index=True, width="stretch")
    st.caption(f"Missing shown: {len(miss_df2):,} | Missing total: {int(miss_total2 or 0):,}")
