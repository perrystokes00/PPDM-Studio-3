# ppdm_loader/parent_autoseed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import ppdm_loader.db as db
from ppdm_loader.fk_introspect import FKInfo


# -----------------------
# quoting + helpers
# -----------------------
def _qident(name: str) -> str:
    return "[" + (name or "").replace("]", "]]") + "]"


def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def _upper_set(df: pd.DataFrame, col: str) -> set[str]:
    if df is None or df.empty or col not in df.columns:
        return set()
    return set(df[col].astype(str).str.upper().tolist())


def _fetch_parent_columns(conn, schema: str, table: str) -> pd.DataFrame:
    sql = """
    SELECT
        c.name AS column_name,
        c.is_nullable,
        t.name AS type_name,
        c.max_length
    FROM sys.columns c
    JOIN sys.tables tb ON tb.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = tb.schema_id
    JOIN sys.types t ON t.user_type_id = c.user_type_id
    WHERE s.name = ? AND tb.name = ?
    ORDER BY c.column_id;
    """
    return db.read_sql(conn, sql, params=[schema, table])


def _fetch_parent_pk_columns(conn, schema: str, table: str) -> list[str]:
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


# -----------------------
# assessment
# -----------------------
@dataclass
class AutoSeedAssessment:
    can_auto_seed: bool
    reason: str
    required_extra_cols: list[str]


def assess_parent_auto_seed(conn, fk: FKInfo) -> AutoSeedAssessment:
    """
    Auto-seed is allowed if:
      - all FK parent key columns are provided by the FK itself (obviously)
      - any other NOT NULL columns on the parent are either:
          - identity OR has default constraint OR is computed OR is rowguid-ish we fill (PPDM_GUID)
          - OR in a small set we can safely default (ACTIVE_IND)
    We keep this conservative to avoid inserting garbage.
    """
    parent_cols = _fetch_parent_columns(conn, fk.parent_schema, fk.parent_table)
    if parent_cols is None or parent_cols.empty:
        return AutoSeedAssessment(False, "Could not introspect parent columns.", [])

    parent_schema = fk.parent_schema
    parent_table = fk.parent_table

    # identity/default/computed detection from sys.columns + defaults
    sql2 = """
    SELECT
        c.name AS column_name,
        c.is_nullable,
        c.is_identity,
        c.is_computed,
        dc.definition AS default_definition
    FROM sys.columns c
    JOIN sys.tables tb ON tb.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = tb.schema_id
    LEFT JOIN sys.default_constraints dc
      ON dc.parent_object_id = c.object_id
     AND dc.parent_column_id = c.column_id
    WHERE s.name = ? AND tb.name = ?;
    """
    meta = db.read_sql(conn, sql2, params=[parent_schema, parent_table])
    if meta is None or meta.empty:
        return AutoSeedAssessment(False, "Could not introspect parent defaults/identity.", [])

    parent_fk_cols = {p.upper() for _, p in fk.pairs}

    required_extras: list[str] = []
    for _, r in meta.iterrows():
        col = str(r["column_name"])
        cu = col.upper()
        is_nullable = int(r["is_nullable"]) == 1
        is_identity = int(r["is_identity"]) == 1
        is_computed = int(r["is_computed"]) == 1
        has_default = (r.get("default_definition") is not None) and (str(r.get("default_definition")).strip() != "")

        if cu in parent_fk_cols:
            continue  # FK provides it

        if is_nullable or is_identity or is_computed or has_default:
            continue

        # Allow a couple PPDM conventions
        if cu == "PPDM_GUID":
            continue
        if cu == "ACTIVE_IND":
            continue

        required_extras.append(col)

    if required_extras:
        return AutoSeedAssessment(
            can_auto_seed=False,
            reason="Parent has NOT NULL columns without defaults beyond FK keys.",
            required_extra_cols=required_extras,
        )

    return AutoSeedAssessment(True, "OK to auto-seed using FK keys (+ optional defaults).", [])


# -----------------------
# SQL generation
# -----------------------
@dataclass
class AutoSeedSQL:
    sql_missing_sample: str
    sql_missing_count: str
    sql_seed_insert: str


def build_autoseed_sql(
    *,
    norm_view_fqn: str,
    fk: FKInfo,
    top_n: int = 2000,
    default_active_ind: Optional[str] = "Y",
) -> AutoSeedSQL:
    """
    Uses norm_view_fqn and expects NAT columns in view: <child_col>__NAT.
    For each fk pair (child_col -> parent_col), we read v.[child_col__NAT]
    and compare to parent key columns.
    """
    v = norm_view_fqn
    tgt = _qfqn(fk.parent_schema, fk.parent_table)

    # build key expressions
    key_select = []
    key_not_null = []
    join_pred = []
    order_cols = []

    for i, (child_col, parent_col) in enumerate(fk.pairs, start=1):
        nat_col = f"{child_col}__NAT"
        alias = f"k{i}"

        key_select.append(
            f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_qident(nat_col)}))), N'') AS {alias}"
        )
        key_not_null.append(f"{alias} IS NOT NULL")
        join_pred.append(
            f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), t.{_qident(parent_col)}))), N'') = s.{alias}"
        )
        order_cols.append(alias)

    key_select_sql = ",\n        ".join(key_select)
    not_null_sql = " AND ".join(key_not_null)
    join_sql = " AND ".join(join_pred)
    order_sql = ", ".join(order_cols)

    # Missing sample
    sql_missing_sample = f"""
;WITH src AS (
    SELECT DISTINCT
        {key_select_sql}
    FROM {v} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {not_null_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_sql}
      )
)
SELECT TOP ({int(top_n)}) *
FROM missing
ORDER BY {order_sql};
""".strip()

    # Missing count
    sql_missing_count = f"""
;WITH src AS (
    SELECT DISTINCT
        {key_select_sql}
    FROM {v} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {not_null_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_sql}
      )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

    # Seed insert: only key cols + optional ACTIVE_IND + PPDM_GUID
    insert_cols = [_qident(parent_col) for _, parent_col in fk.pairs]
    select_cols = [f"s.k{i}" for i in range(1, len(fk.pairs) + 1)]

    # optional defaults (safe)
    if default_active_ind is not None:
        insert_cols.append("[ACTIVE_IND]")
        vact = str(default_active_ind).replace("'", "''")
        select_cols.append(f"N'{vact}'")

    insert_cols.append("[PPDM_GUID]")
    select_cols.append("CONVERT(nvarchar(36), NEWID())")

    insert_cols_sql = ", ".join(insert_cols)
    select_cols_sql = ", ".join(select_cols)

    sql_seed_insert = f"""
;WITH src AS (
    SELECT DISTINCT
        {key_select_sql}
    FROM {v} v
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE {not_null_sql}
      AND NOT EXISTS (
          SELECT 1
          FROM {tgt} t
          WHERE {join_sql}
      )
)
INSERT INTO {tgt} ({insert_cols_sql})
SELECT {select_cols_sql}
FROM missing s;
""".strip()

    return AutoSeedSQL(
        sql_missing_sample=sql_missing_sample,
        sql_missing_count=sql_missing_count,
        sql_seed_insert=sql_seed_insert,
    )


# -----------------------
# execution helpers (no pandas hazards)
# -----------------------
def run_select_df(conn, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    return db.read_sql(conn, sql, params=params)


def run_insert(conn, sql: str, params: list[Any] | None = None) -> int:
    cur = conn.cursor()
    try:
        cur.execute(sql, params or [])
        inserted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        return int(inserted)
    finally:
        try:
            cur.close()
        except Exception:
            pass
