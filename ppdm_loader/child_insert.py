# ppdm_loader/child_insert.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ppdm_loader.db as db


@dataclass
class FkMap:
    fk_name: str
    parent_schema: str
    parent_table: str
    pairs: List[Tuple[str, str]]  # (child_col, parent_col)


def fetch_fk_map(conn, *, child_schema: str, child_table: str) -> List[FkMap]:
    sql = """
    SELECT
        fk.name AS fk_name,
        sch_p.name AS parent_schema,
        tab_p.name AS parent_table,
        col_c.name AS child_col,
        col_p.name AS parent_col,
        fkc.constraint_column_id AS ord
    FROM sys.foreign_key_columns fkc
    JOIN sys.foreign_keys fk
      ON fk.object_id = fkc.constraint_object_id
    JOIN sys.tables tab_c
      ON tab_c.object_id = fkc.parent_object_id
    JOIN sys.schemas sch_c
      ON sch_c.schema_id = tab_c.schema_id
    JOIN sys.tables tab_p
      ON tab_p.object_id = fkc.referenced_object_id
    JOIN sys.schemas sch_p
      ON sch_p.schema_id = tab_p.schema_id
    JOIN sys.columns col_c
      ON col_c.object_id = fkc.parent_object_id AND col_c.column_id = fkc.parent_column_id
    JOIN sys.columns col_p
      ON col_p.object_id = fkc.referenced_object_id AND col_p.column_id = fkc.referenced_column_id
    WHERE sch_c.name = ?
      AND tab_c.name = ?
    ORDER BY fk.name, fkc.constraint_column_id;
    """
    df = db.read_sql(conn, sql, params=[child_schema, child_table])
    if df is None or df.empty:
        return []

    out: dict[tuple[str, str, str], FkMap] = {}
    for _, r in df.iterrows():
        key = (str(r["fk_name"]), str(r["parent_schema"]), str(r["parent_table"]))
        if key not in out:
            out[key] = FkMap(
                fk_name=key[0],
                parent_schema=key[1],
                parent_table=key[2],
                pairs=[],
            )
        out[key].pairs.append((str(r["child_col"]), str(r["parent_col"])))
    return list(out.values())


def fetch_pk_columns(conn, *, schema: str, table: str) -> List[str]:
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


def _qident(name: str) -> str:
    return "[" + (name or "").replace("]", "]]") + "]"


def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def build_insert_new_sql_generic(
    *,
    conn,
    target_schema: str,
    target_table: str,
    norm_view: str,
    insert_cols: List[str],
    # key preference: PK columns if present, else fallback
    fallback_key_cols: Optional[List[str]] = None,
    # optional FK enforcement (child tables): only insert rows with existing parent
    fk_name: Optional[str] = None,
    # optional audit defaults (only applied if columns exist in insert_cols)
    loaded_by: Optional[str] = None,
) -> tuple[str, str]:
    """
    Returns (sql_count, sql_insert) for "insert new only", but:
      - Uses PK columns (composite-aware) if present
      - Otherwise uses fallback_key_cols
      - Optionally enforces FK parent existence via JOIN on a chosen FK
      - Optionally populates audit columns if present (as expressions in SELECT)
    """

    target_full = _qfqn(target_schema, target_table)

    pk_cols = fetch_pk_columns(conn, schema=target_schema, table=target_table)
    key_cols = pk_cols[:] if pk_cols else (fallback_key_cols[:] if fallback_key_cols else [])

    if not key_cols:
        # absolute last resort: first column (better than crashing)
        key_cols = [insert_cols[0]]

    # Optional FK join (for child tables)
    join_sql = ""
    if fk_name:
        fks = fetch_fk_map(conn, child_schema=target_schema, child_table=target_table)
        fk = next((x for x in fks if x.fk_name == fk_name), None)
        if fk:
            parent_full = _qfqn(fk.parent_schema, fk.parent_table)
            preds = []
            for child_col, parent_col in fk.pairs:
                # We can only join on columns that we are inserting (exist in norm view intersection)
                if child_col not in insert_cols:
                    # If FK col isn't being inserted, join isn't possible -> skip FK enforcement
                    preds = []
                    break
                preds.append(f"p.{_qident(parent_col)} = v.{_qident(child_col)}")
            if preds:
                join_sql = f"JOIN {parent_full} p ON " + " AND ".join(preds)

    # SELECT expressions (support audit defaults)
    who = (loaded_by or "").replace("'", "''")

    sel_exprs = []
    for c in insert_cols:
        u = c.upper()

        if who and u in {"ROW_CREATED_BY", "ROW_CHANGED_BY"}:
            sel_exprs.append(f"N'{who}' AS {_qident(c)}")

        elif u in {"ROW_CREATED_DATE", "ROW_CHANGED_DATE", "ROW_EFFECTIVE_DATE"}:
            sel_exprs.append(f"SYSUTCDATETIME() AS {_qident(c)}")

        elif u == "PPDM_GUID":
            sel_exprs.append(f"CONVERT(nvarchar(36), NEWID()) AS {_qident(c)}")

        else:
            sel_exprs.append(f"v.{_qident(c)}")


    col_sql = ", ".join(_qident(c) for c in insert_cols)
    sel_sql = ", ".join(sel_exprs)

    # Key predicate: composite-safe
    key_not_null = " AND ".join([f"v.{_qident(k)} IS NOT NULL" for k in key_cols])
    exists_pred = " AND ".join([f"t.{_qident(k)} = v.{_qident(k)}" for k in key_cols])

    sql_count = f"""
SET NOCOUNT ON;
SELECT COUNT(*) AS would_insert
FROM {norm_view} v
{join_sql}
WHERE {key_not_null}
  AND NOT EXISTS (SELECT 1 FROM {target_full} t WHERE {exists_pred});
""".strip()

    sql_insert = f"""
SET NOCOUNT ON;
INSERT INTO {target_full} ({col_sql})
SELECT {sel_sql}
FROM {norm_view} v
{join_sql}
WHERE {key_not_null}
  AND NOT EXISTS (SELECT 1 FROM {target_full} t WHERE {exists_pred});
""".strip()

    return sql_count, sql_insert
