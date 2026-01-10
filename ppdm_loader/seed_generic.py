# ppdm_loader/seed_generic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

import ppdm_loader.db as db
import json

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
_AUDIT_COLS = {
    "RID",
    "PPDM_GUID",
    "ROW_CREATED_BY", "ROW_CREATED_DATE",
    "ROW_CHANGED_BY", "ROW_CHANGED_DATE",
    "ROW_EFFECTIVE_DATE", "ROW_EXPIRY_DATE",
    "ROW_EXPIRY_DATE",
    "ROW_QUALITY",
    "REMARK",
}

def qident(name: str) -> str:
    return "[" + (name or "").replace("]", "]]") + "]"

def qfqn(schema: str, table: str) -> str:
    return f"{qident(schema)}.{qident(table)}"

def normalize_df_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
        out[c] = out[c].replace({"": None, "nan": None, "None": None})
    return out

def fetch_table_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
    SELECT c.name AS column_name
    FROM sys.schemas s
    JOIN sys.tables  t ON t.schema_id = s.schema_id
    JOIN sys.columns c ON c.object_id = t.object_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return [str(x).strip() for x in df["column_name"].tolist()]

def fetch_pk_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
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
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return [str(x).strip() for x in df["COLUMN_NAME"].tolist() if str(x).strip()]

def table_has_column(conn, schema: str, table: str, col: str) -> bool:
    cols = {c.upper() for c in fetch_table_columns(conn, schema, table)}
    return (col or "").strip().upper() in cols


@dataclass(frozen=True)
class MapRow:
    target_column: str
    use: bool
    source_column: str
    constant_value: str
    transform: str  # none|trim|upper


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    t = (transform or "none").strip().lower()
    s = series
    if t == "none":
        return s
    if t == "trim":
        return s.astype(str).str.strip()
    if t == "upper":
        return s.astype(str).str.strip().str.upper()
    return s


def build_src_frame_from_mapping(
    df_in: pd.DataFrame,
    mapping: list[MapRow],
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns = chosen target columns.
    All values are strings/None (caller can cast later if needed).
    """
    df = normalize_df_strings(df_in)

    out_cols: dict[str, pd.Series] = {}
    in_cols_upper = {c.upper(): c for c in df.columns}

    for m in mapping:
        if not m.use:
            continue

        tgt = m.target_column.strip()
        if not tgt:
            continue

        if m.source_column:
            src_key = m.source_column.strip().upper()
            src_real = in_cols_upper.get(src_key)
            if src_real is None:
                # source col not present, treat as null series
                s = pd.Series([None] * len(df))
            else:
                s = df[src_real]
                s = apply_transform(s, m.transform)
        else:
            # constant
            const = (m.constant_value or "").strip()
            s = pd.Series([const if const != "" else None] * len(df))

        # normalize empties to None
        s = s.replace({"": None, "nan": None, "None": None})
        out_cols[tgt] = s

    out = pd.DataFrame(out_cols)
    # drop rows where all selected cols are null
    if not out.empty:
        out = out.dropna(how="all")
    return out

def preview_missing_by_pk(
    conn,
    *,
    target_schema: str,
    target_table: str,
    pk_cols: list[str],
    items: list[dict[str, Any]] | None = None,
    src_df: pd.DataFrame | None = None,
    top_n: int = 500,
) -> tuple[pd.DataFrame, int]:
    # Allow either items or src_df
    if items is None:
        if src_df is None or src_df.empty:
            return pd.DataFrame(), 0
        # Only keep pk columns
        missing = [c for c in pk_cols if c not in src_df.columns]
        if missing:
            raise ValueError(f"src_df missing PK columns: {missing}")
        items = src_df[pk_cols].dropna().drop_duplicates().to_dict(orient="records")

    if not items:
        return pd.DataFrame(), 0

    # ... then proceed with the OPENJSON-based SQL (the safe CTE pattern)


    payload = json.dumps(items, ensure_ascii=False)
    tgt = qfqn(target_schema, target_table)

    # OPENJSON schema: include ONLY pk_cols (nvarchar is fine for comparison)
    with_cols = ",\n        ".join(f"{c} nvarchar(4000) '$.{c}'" for c in pk_cols)

    # src projection
    proj_cols = ",\n        ".join(
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), j.{qident(c)}))), N'') AS {qident(c)}"
        for c in pk_cols
    )

    # predicates
    req_pred = " AND ".join(f"s.{qident(c)} IS NOT NULL" for c in pk_cols)
    join_pred = " AND ".join(
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{qident(c)}))), N'') = s.{qident(c)}"
        for c in pk_cols
    )

    order_by = ", ".join(qident(c) for c in pk_cols)

    sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_cols}
    FROM OPENJSON(?)
    WITH (
        {with_cols}
    ) j
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_pred}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_pred}
      )
)
SELECT TOP ({int(top_n)}) *
FROM missing
ORDER BY {order_by};
""".strip()

    sql_count = f"""
;WITH src AS (
    SELECT DISTINCT
        {proj_cols}
    FROM OPENJSON(?)
    WITH (
        {with_cols}
    ) j
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_pred}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_pred}
      )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

    df_sample = db.read_sql(conn, sql_sample, params=[payload])
    if df_sample is None or df_sample.empty:
        df_sample = pd.DataFrame()

    df_count = db.read_sql(conn, sql_count, params=[payload])
    missing_total = int(df_count.iloc[0]["missing_total"]) if (df_count is not None and not df_count.empty) else 0

    return df_sample, missing_total

def seed_missing_rows(
    conn,
    *,
    target_schema: str,
    target_table: str,
    pk_cols: list[str],
    insert_df: pd.DataFrame,
    loaded_by: str = "Perry M Stokes",
) -> int:
    """
    Insert missing PK tuples + selected non-PK mapped columns.
    Auto-populates PPDM_GUID + audit columns if present on target table and not provided.

    Returns an accurate inserted row count (does NOT rely on cursor.rowcount).
    """
    if insert_df is None or insert_df.empty:
        return 0
    if not pk_cols:
        raise ValueError("Target table has no PK (or PK not detected). MVP requires PK.")
    for c in pk_cols:
        if c not in insert_df.columns:
            raise ValueError(f"Insert DF missing PK column: {c}")

    # distinct rows (based on all columns provided)
    insert_df = insert_df.dropna(subset=pk_cols).drop_duplicates()

    import json
    items = insert_df.to_dict(orient="records")
    payload = json.dumps(items, ensure_ascii=False)

    tgt = qfqn(target_schema, target_table)
    tgt_cols = {c.upper() for c in fetch_table_columns(conn, target_schema, target_table)}

    # columns from user DF that exist on target
    user_cols = [c for c in insert_df.columns if c.upper() in tgt_cols]

    # defaults we may inject (only if present on target and not in user_cols)
    user_cols_u = {c.upper() for c in user_cols}
    who = (loaded_by or "").replace("'", "''")

    default_cols: list[str] = []

    def _add_default(col: str) -> None:
        if col.upper() in tgt_cols and col.upper() not in user_cols_u:
            default_cols.append(col)

    _add_default("PPDM_GUID")
    _add_default("ROW_CREATED_BY")
    _add_default("ROW_CREATED_DATE")
    _add_default("ROW_CHANGED_BY")
    _add_default("ROW_CHANGED_DATE")
    _add_default("ROW_EFFECTIVE_DATE")

    # final insert column order (user cols first, then defaults)
    insert_cols = user_cols + default_cols

    # JSON-provided columns (OPENJSON) should NOT include defaults
    default_u = {c.upper() for c in default_cols}
    json_cols = [c for c in user_cols if c.upper() not in default_u]  # user_cols only

    if not json_cols:
        raise ValueError("No JSON columns available to seed. Map at least the PK columns from the source.")

    # OPENJSON schema only for json_cols
    with_cols = ",\n        ".join(f"{c} nvarchar(4000) '$.{c}'" for c in json_cols)

    # INSERT select expressions (defaults injected here)
    sel_exprs: list[str] = []
    for c in insert_cols:
        u = c.upper()
        if u == "PPDM_GUID":
            sel_exprs.append(f"CONVERT(nvarchar(36), NEWID()) AS {qident(c)}")
        elif u in {"ROW_CREATED_BY", "ROW_CHANGED_BY"}:
            sel_exprs.append(f"N'{who}' AS {qident(c)}")
        elif u in {"ROW_CREATED_DATE", "ROW_CHANGED_DATE", "ROW_EFFECTIVE_DATE"}:
            sel_exprs.append(f"SYSUTCDATETIME() AS {qident(c)}")
        else:
            sel_exprs.append(f"s.{qident(c)}")

    # join predicate for PK tuple
    join_pred = " AND ".join(
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{qident(c)}))), N'') = s.{qident(c)}"
        for c in pk_cols
    )
    req_pred = " AND ".join(f"s.{qident(c)} IS NOT NULL" for c in pk_cols)

    col_list = ", ".join(qident(c) for c in insert_cols)
    sel_list = ", ".join(sel_exprs)

    # IMPORTANT: return inserted rowcount explicitly
    sql = f"""
SET NOCOUNT ON;

;WITH src AS (
    SELECT DISTINCT
        {", ".join([qident(c) for c in json_cols])}
    FROM OPENJSON(?)
    WITH (
        {with_cols}
    ) j
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {req_pred}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_pred}
      )
)
INSERT INTO {tgt} ({col_list})
SELECT {sel_list}
FROM missing s;

SELECT CAST(@@ROWCOUNT AS int) AS inserted_rows;
""".strip()

    cur = conn.cursor()
    cur.execute(sql, (payload,))

    # Fetch the explicit count (reliable even when pyodbc rowcount = -1)
    row = cur.fetchone()
    inserted = int(row[0]) if row and row[0] is not None else 0

    conn.commit()
    return inserted





