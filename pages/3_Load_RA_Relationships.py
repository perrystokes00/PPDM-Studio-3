# pages/3_Load_RA_Relationships.py
from __future__ import annotations
import streamlit as st
from common.ui import sidebar_connect, require_connection
from common.db_helpers import list_ra_tables

st.set_page_config(page_title="Load ra_ relationships", layout="wide")
sidebar_connect(page_prefix="ra")
conn = require_connection()
sidebar_connect()


st.title("Load ra_ Relationship Tables (Match & Map)")

st.info(
    "ra_ tables usually represent relationships and often have composite keys. "
    "These almost always require a Match & Map grid (you pick multiple input columns per key)."
)

schema = st.text_input("Schema", "dbo", key="ra::schema")
ra_tables = [""] + list_ra_tables(conn, schema=schema)

target = st.selectbox("Pick an ra_ table", ra_tables, key="ra::target")
if target:
    st.write("Next: show PK columns + mapping grid for each key column (coming next).")
