# common/seeding.py
from __future__ import annotations

from typing import Any
from ppdm_loader.db import read_sql
import ppdm_loader.db as db

def _quote_ident(name: str) -> str:
    name = (name or "").replace("]", "]]")
    return f"[{name}]"

def _fqn(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"

def _sql_lit(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v).replace("'", "''")
    return f"N'{s}'"

def missing_distinct_report(
    conn,
    *,
    view_name: str,
    view_col: str,
    target_schema: str,
    target_table: str,
    target_col: str,
    top_sample: int = 25,
) -> dict:
    """
    Compare DISTINCT normalized values in view(view_col) vs target(target_col).
    Returns counts + STRING_AGG sample of missing values.
    """
    tgt_fqn = _fqn(target_schema, target_table)
    vcol = _quote_ident(view_col)
    tcol = _quote_ident(target_col)

    sql = f"""
;WITH src AS (
    SELECT DISTINCT NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{vcol}))), N'') AS val
    FROM {view_name} v
),
src_clean AS (
    SELECT val FROM src WHERE val IS NOT NULL
),
missing AS (
    SELECT val FROM src_clean
    EXCEPT
    SELECT NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{tcol}))), N'')
    FROM {tgt_fqn} t
)
SELECT
  (SELECT COUNT(*) FROM src_clean) AS src_distinct,
  (SELECT COUNT(*) FROM (
        SELECT DISTINCT NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{tcol}))), N'') AS v
        FROM {tgt_fqn} t
   ) x WHERE x.v IS NOT NULL) AS tgt_distinct,
  (SELECT COUNT(*) FROM missing) AS missing_count,
  (SELECT STRING_AGG(val, N', ') WITHIN GROUP (ORDER BY val)
     FROM (SELECT TOP ({int(top_sample)}) val FROM missing ORDER BY val) s
  ) AS missing_sample;
""".strip()

    df = read_sql(conn, sql)
    if df is None or df.empty:
        return {"src_distinct": None, "tgt_distinct": None, "missing_count": None, "missing_sample": None}

    r = df.iloc[0].to_dict()
    for k in ("src_distinct", "tgt_distinct", "missing_count"):
        try:
            r[k] = int(r[k]) if r[k] is not None else None
        except Exception:
            pass
    return r


def seed_ref_table_from_view_distinct(
    *,
    conn,
    view_name: str,
    source_attr: str,
    ref_schema: str,
    ref_table: str,
    pk_col: str,
    defaults: dict | None = None,
    long_name_mode: str = "none",      # "none" | "code" | "column"
    long_name_from: str | None = None, # NORM col if mode="column"
) -> int:
    """
    Insert only missing PK values into ref table.
    Returns inserted_count.
    """
    defaults = defaults or {}
    long_name_mode = (long_name_mode or "none").strip().lower()
    if long_name_mode not in ("none", "code", "column"):
        raise ValueError(f"Unknown long_name_mode: {long_name_mode}")
    if long_name_mode == "column" and not long_name_from:
        raise ValueError("long_name_mode='column' requires long_name_from")

    tgt = _fqn(ref_schema, ref_table)
    pk = _quote_ident(pk_col)

    src_code_expr = f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_quote_ident(source_attr)}))), N'')"

    insert_cols = [pk_col]
    select_exprs = [f"m.{pk}"]

    if long_name_mode != "none":
        insert_cols.append("LONG_NAME")
        if long_name_mode == "code":
            select_exprs.append(f"m.{pk}")
        else:
            select_exprs.append("m.[LONG_NAME_VAL]")

    # add defaults
    for k, v in defaults.items():
        insert_cols.append(k)
        select_exprs.append(_sql_lit(v))

    col_list = ", ".join(_quote_ident(c) for c in insert_cols)
    sel_list = ", ".join(select_exprs)

    if long_name_mode == "column":
        ln_expr = f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_quote_ident(long_name_from)}))), N'')"
        sql = f"""
;WITH src0 AS (
    SELECT
        {src_code_expr} AS {pk},
        {ln_expr}       AS [LONG_NAME_VAL]
    FROM {view_name} v
    WHERE v.{_quote_ident(source_attr)} IS NOT NULL
),
src AS (
    SELECT {pk}, MIN([LONG_NAME_VAL]) AS [LONG_NAME_VAL]
    FROM src0
    WHERE {pk} IS NOT NULL
    GROUP BY {pk}
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE NOT EXISTS (
        SELECT 1 FROM {tgt} t WHERE t.{pk} = s.{pk}
    )
)
DECLARE @ins TABLE (x int);
INSERT INTO {tgt} ({col_list})
OUTPUT 1 INTO @ins(x)
SELECT {sel_list}
FROM missing m;

SELECT COUNT(*) AS inserted_count FROM @ins;
""".strip()
    else:
        sql = f"""
;WITH src AS (
    SELECT DISTINCT {src_code_expr} AS {pk}
    FROM {view_name} v
    WHERE v.{_quote_ident(source_attr)} IS NOT NULL
),
src2 AS (
    SELECT {pk} FROM src WHERE {pk} IS NOT NULL
),
missing AS (
    SELECT s.*
    FROM src2 s
    WHERE NOT EXISTS (
        SELECT 1 FROM {tgt} t WHERE t.{pk} = s.{pk}
    )
)
DECLARE @ins TABLE (x int);
INSERT INTO {tgt} ({col_list})
OUTPUT 1 INTO @ins(x)
SELECT {sel_list}
FROM missing m;

SELECT COUNT(*) AS inserted_count FROM @ins;
""".strip()

    df = read_sql(conn, sql)
    if df is None or df.empty:
        db.exec_sql(conn, sql)
        return 0
    return int(df.iloc[0]["inserted_count"])
