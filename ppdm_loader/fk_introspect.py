# ppdm_loader/fk_introspect.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import ppdm_loader.db as db


@dataclass
class FKInfo:
    fk_name: str
    child_schema: str
    child_table: str
    parent_schema: str
    parent_table: str
    pairs: list[tuple[str, str]]  # [(child_col, parent_col), ...] ordered


def introspect_fk_by_child_col(
    conn,
    *,
    child_schema: str,
    child_table: str,
    child_col: str,
) -> Optional[FKInfo]:
    """
    Find the FK that involves child_schema.child_table.child_col.
    Returns composite pairs ordered by ordinal.
    """
    sql = r"""
    ;WITH fkcols AS (
        SELECT
            fk.name AS fk_name,
            sch_child.name AS child_schema,
            tab_child.name AS child_table,
            col_child.name AS child_column,
            sch_parent.name AS parent_schema,
            tab_parent.name AS parent_table,
            col_parent.name AS parent_column,
            fkc.constraint_column_id AS ordinal
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
            ON fkc.constraint_object_id = fk.object_id
        JOIN sys.tables tab_child
            ON tab_child.object_id = fk.parent_object_id
        JOIN sys.schemas sch_child
            ON sch_child.schema_id = tab_child.schema_id
        JOIN sys.columns col_child
            ON col_child.object_id = tab_child.object_id
           AND col_child.column_id = fkc.parent_column_id
        JOIN sys.tables tab_parent
            ON tab_parent.object_id = fk.referenced_object_id
        JOIN sys.schemas sch_parent
            ON sch_parent.schema_id = tab_parent.schema_id
        JOIN sys.columns col_parent
            ON col_parent.object_id = tab_parent.object_id
           AND col_parent.column_id = fkc.referenced_column_id
    )
    SELECT *
    FROM fkcols
    WHERE child_schema = ?
      AND child_table  = ?
      AND child_column = ?
    ORDER BY fk_name, ordinal;
    """
    df = db.read_sql(conn, sql, params=[child_schema, child_table, child_col])
    if df is None or df.empty:
        return None

    # If multiple FK names match, choose the one with most columns
    best_fk = None
    best_n = -1
    for fk_name, g in df.groupby("fk_name"):
        n = len(g)
        if n > best_n:
            best_n = n
            best_fk = (fk_name, g.sort_values("ordinal"))

    fk_name, g = best_fk
    first = g.iloc[0]
    pairs = [(r["child_column"], r["parent_column"]) for _, r in g.iterrows()]

    return FKInfo(
        fk_name=str(fk_name),
        child_schema=str(first["child_schema"]),
        child_table=str(first["child_table"]),
        parent_schema=str(first["parent_schema"]),
        parent_table=str(first["parent_table"]),
        pairs=pairs,
    )


def introspect_all_fks_for_child_table(
    conn,
    *,
    child_schema: str,
    child_table: str,
) -> pd.DataFrame:
    """
    Return ALL FK columns for a child table (one row per column pair).
    Useful for listing all FK options without guessing.
    """
    sql = r"""
    SELECT
        fk.name AS fk_name,
        sch_child.name AS child_schema,
        tab_child.name AS child_table,
        col_child.name AS child_column,
        sch_parent.name AS parent_schema,
        tab_parent.name AS parent_table,
        col_parent.name AS parent_column,
        fkc.constraint_column_id AS ordinal
    FROM sys.foreign_keys fk
    JOIN sys.foreign_key_columns fkc
        ON fkc.constraint_object_id = fk.object_id
    JOIN sys.tables tab_child
        ON tab_child.object_id = fk.parent_object_id
    JOIN sys.schemas sch_child
        ON sch_child.schema_id = tab_child.schema_id
    JOIN sys.columns col_child
        ON col_child.object_id = tab_child.object_id
       AND col_child.column_id = fkc.parent_column_id
    JOIN sys.tables tab_parent
        ON tab_parent.object_id = fk.referenced_object_id
    JOIN sys.schemas sch_parent
        ON sch_parent.schema_id = tab_parent.schema_id
    JOIN sys.columns col_parent
        ON col_parent.object_id = tab_parent.object_id
       AND col_parent.column_id = fkc.referenced_column_id
    WHERE sch_child.name = ?
      AND tab_child.name = ?
    ORDER BY fk.name, fkc.constraint_column_id;
    """
    df = db.read_sql(conn, sql, params=[child_schema, child_table])
    return df if df is not None else pd.DataFrame()
