# common/db_helpers.py
from __future__ import annotations

import pandas as pd
from ppdm_loader.db import read_sql

def list_tables_like(conn, *, schema: str | None = None, pattern: str = "%") -> pd.DataFrame:
    sql = """
    SELECT s.name AS schema_name, t.name AS table_name
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE (? IS NULL OR s.name = ?)
      AND t.name LIKE ?
    ORDER BY s.name, t.name;
    """
    return read_sql(conn, sql, params=[schema, schema, pattern])

def list_r_tables(conn, schema: str | None = None) -> list[str]:
    df = list_tables_like(conn, schema=schema, pattern="r[_]%")
    if df is None or df.empty:
        return []
    return [f"{r['schema_name']}.{r['table_name']}" for _, r in df.iterrows()]

def list_ra_tables(conn, schema: str | None = None) -> list[str]:
    df = list_tables_like(conn, schema=schema, pattern="ra[_]%")
    if df is None or df.empty:
        return []
    return [f"{r['schema_name']}.{r['table_name']}" for _, r in df.iterrows()]
