import streamlit as st
import pandas as pd
from pathlib import Path
import json

import ppdm_loader.db as db

st.set_page_config(page_title="PPDM Studio — Data Explorer", layout="wide")
st.title("Data Explorer (mini-SSMS)")

conn = st.session_state.get("conn")
if conn is None:
    st.warning("Not connected. Use the sidebar to connect first.")
    st.stop()

PDF_DIR = Path("docs/schema_pdfs")
BOOKMARKS_PATH = PDF_DIR / "bookmarks.json"
bookmarks = {}
if BOOKMARKS_PATH.exists():
    try:
        bookmarks = json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
        if not isinstance(bookmarks, dict):
            bookmarks = {}
    except Exception:
        bookmarks = {}

# -----------------------------
# Helpers
# -----------------------------
def _qident(name: str) -> str:
    return f"[{name.replace(']', ']]')}]"

def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"

def list_schemas(conn) -> list[str]:
    sql = """
    SELECT s.name AS schema_name
    FROM sys.schemas s
    WHERE s.name NOT IN ('sys', 'INFORMATION_SCHEMA')
    ORDER BY s.name;
    """
    return db.read_sql(conn, sql)["schema_name"].tolist()

def list_tables(conn, schema: str) -> list[str]:
    sql = """
    SELECT t.name AS table_name
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ?
    ORDER BY t.name;
    """
    return db.read_sql(conn, sql, params=[schema])["table_name"].tolist()

def fetch_columns(conn, schema: str, table: str) -> pd.DataFrame:
    sql = """
    SELECT
      c.column_id,
      c.name AS column_name,
      ty.name AS data_type,
      CASE
        WHEN ty.name IN ('nvarchar','nchar') THEN c.max_length/2
        WHEN ty.name IN ('varchar','char','varbinary','binary') THEN c.max_length
        ELSE c.max_length
      END AS max_length,
      c.precision,
      c.scale,
      c.is_nullable
    FROM sys.columns c
    JOIN sys.types ty ON ty.user_type_id = c.user_type_id
    JOIN sys.tables t ON t.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id;
    """
    return db.read_sql(conn, sql, params=[schema, table])

def fetch_pk_cols(conn, schema: str, table: str) -> list[str]:
    sql = """
    SELECT c.name AS column_name
    FROM sys.indexes i
    JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id
    JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id
    JOIN sys.tables t ON t.object_id=i.object_id
    JOIN sys.schemas s ON s.schema_id=t.schema_id
    WHERE i.is_primary_key=1 AND s.name=? AND t.name=?
    ORDER BY ic.key_ordinal;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    return df["column_name"].tolist() if not df.empty else []

def fetch_fk_summary(conn, schema: str, table: str) -> pd.DataFrame:
    sql = """
    SELECT
      fk.name AS fk_name,
      s2.name AS parent_schema,
      t2.name AS parent_table
    FROM sys.foreign_keys fk
    JOIN sys.tables t1 ON t1.object_id = fk.parent_object_id
    JOIN sys.schemas s1 ON s1.schema_id = t1.schema_id
    JOIN sys.tables t2 ON t2.object_id = fk.referenced_object_id
    JOIN sys.schemas s2 ON s2.schema_id = t2.schema_id
    WHERE s1.name = ? AND t1.name = ?
    ORDER BY fk.name;
    """
    return db.read_sql(conn, sql, params=[schema, table])

def row_count(conn, schema: str, table: str) -> int:
    sql = f"SELECT COUNT_BIG(*) AS n FROM {_qfqn(schema, table)};"
    return int(db.read_sql(conn, sql).iloc[0]["n"])

def preview_rows(conn, schema: str, table: str, top_n: int, order_col: str | None) -> pd.DataFrame:
    fqn = _qfqn(schema, table)
    order_sql = f" ORDER BY {_qident(order_col)}" if order_col else ""
    sql = f"SELECT TOP ({int(top_n)}) * FROM {fqn}{order_sql};"
    return db.read_sql(conn, sql)

def _find_bookmark_for_table(table_fqn: str):
    """
    Return (pdf_name, page) if bookmarks.json contains any entry with table == table_fqn.
    """
    tfqn = (table_fqn or "").strip().lower()
    for pdf_name, sec_map in (bookmarks or {}).items():
        if not isinstance(sec_map, dict):
            continue
        for _, item in sec_map.items():
            if not isinstance(item, dict):
                continue
            t = (item.get("table") or "").strip().lower()
            if t == tfqn:
                try:
                    return pdf_name, int(item.get("page", 1) or 1)
                except Exception:
                    return pdf_name, 1
    return None, None

def _is_readonly_select(sql: str) -> bool:
    s = (sql or "").strip().lstrip("(")
    s_u = s.upper()
    # Disallow anything that isn't SELECT/WITH leading
    if not (s_u.startswith("SELECT") or s_u.startswith("WITH")):
        return False
    bad = ["INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "TRUNCATE", "CREATE", "EXEC", "GRANT", "REVOKE"]
    return not any(b in s_u for b in bad)

def _enforce_top_cap(sql: str, cap: int) -> str:
    # If user didn't specify TOP, we wrap
    s = (sql or "").strip().rstrip(";")
    s_u = s.upper()
    if " TOP " in s_u or "TOP(" in s_u:
        return s  # user already specified TOP
    return f"SELECT TOP ({int(cap)}) * FROM ({s}) AS q"

# -----------------------------
# Initial target from ERD viewer (link)
# -----------------------------
target_from_erd = (st.session_state.get("explorer_target_fqn") or "").strip()

schemas = list_schemas(conn)
schema_default = "dbo" if "dbo" in schemas else schemas[0]

# If ERD passed a table, use it to set defaults
if target_from_erd and "." in target_from_erd:
    s_in, t_in = target_from_erd.split(".", 1)
    if s_in in schemas:
        schema_default = s_in

schema = st.selectbox("Schema", schemas, index=schemas.index(schema_default) if schema_default in schemas else 0)

tables = list_tables(conn, schema)
table_default = tables[0] if tables else ""
if target_from_erd and "." in target_from_erd:
    s_in, t_in = target_from_erd.split(".", 1)
    if s_in == schema and t_in in tables:
        table_default = t_in

table = st.selectbox("Table", tables, index=tables.index(table_default) if table_default in tables else 0)

top_n = st.selectbox("Preview rows", [10, 25, 50, 100, 250, 1000], index=1)

# Clear linked target after applying (so it doesn't keep overriding)
st.session_state["explorer_target_fqn"] = ""

cols_df = fetch_columns(conn, schema, table)
pk_cols = fetch_pk_cols(conn, schema, table)
fk_df = fetch_fk_summary(conn, schema, table)

order_col = pk_cols[0] if pk_cols else ("RID" if "RID" in cols_df["column_name"].tolist() else None)

table_fqn = f"{schema}.{table}"
pdf_name, pdf_page = _find_bookmark_for_table(table_fqn)

cA, cB, cC = st.columns([2, 1, 1])
with cA:
    st.caption(f"Target: **{table_fqn}**")
with cB:
    st.caption(f"PK: {', '.join(pk_cols) if pk_cols else '(none)'}")
with cC:
    if pdf_name:
        if st.button(f"Jump to ERD ({pdf_name} p{pdf_page})", type="primary"):
            st.query_params["pdf"] = pdf_name
            st.query_params["page"] = str(pdf_page)
            st.query_params["zoom"] = "150"
            st.switch_page("pages/97_Schema_PDF_Viewer.py")
    else:
        st.caption("No ERD bookmark for this table")

left, right = st.columns([1.2, 1])

with left:
    with st.expander("Columns", expanded=True):
        df_show = cols_df.copy()
        df_show["PK"] = df_show["column_name"].str.upper().isin({c.upper() for c in pk_cols})
        df_show["NULLABLE"] = df_show["is_nullable"].map({True: "YES", False: "NO"})
        df_show = df_show[["column_id","column_name","data_type","max_length","precision","scale","NULLABLE","PK"]]
        st.dataframe(df_show, hide_index=True, use_container_width=True)

with right:
    with st.expander("FK Parents", expanded=True):
        if fk_df.empty:
            st.caption("(none)")
        else:
            st.dataframe(fk_df, hide_index=True, use_container_width=True)

with st.expander("Row count", expanded=False):
    try:
        n = row_count(conn, schema, table)
        st.info(f"Rows: {n:,}")
    except Exception as e:
        st.error(f"Count failed: {e}")

st.subheader("Preview")
try:
    df_prev = preview_rows(conn, schema, table, top_n=top_n, order_col=order_col)
    st.dataframe(df_prev, hide_index=True, use_container_width=True)
except Exception as e:
    st.error(f"Preview failed: {e}")

# -----------------------------
# Advanced: read-only SQL
# -----------------------------
with st.expander("Advanced (read-only SQL)", expanded=False):
    st.caption("Rules: SELECT/WITH only. No INSERT/UPDATE/DELETE/MERGE/etc. Results are capped.")
    cap = st.selectbox("Result cap (TOP)", [50, 100, 250, 500, 1000, 5000], index=2, key="sql_cap")
    default_sql = f"SELECT TOP (100) * FROM {_qfqn(schema, table)};"
    sql_text = st.text_area("SQL", value=default_sql, height=160, key="sql_text")

    c1, c2 = st.columns([1, 1])
    run = c1.button("Run SQL (read-only)", type="primary")
    clear = c2.button("Reset to default")

    if clear:
        st.session_state["sql_text"] = default_sql
        st.rerun()

    if run:
        if not _is_readonly_select(sql_text):
            st.error("Blocked: SQL must start with SELECT or WITH and must not contain DDL/DML keywords.")
        else:
            try:
                final_sql = _enforce_top_cap(sql_text, int(cap))
                df = db.read_sql(conn, final_sql)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.code(final_sql, language="sql")
            except Exception as e:
                st.error(f"SQL failed: {e}")
