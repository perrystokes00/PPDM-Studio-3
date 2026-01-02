# ppdm_loader/fk_suggest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import ppdm_loader.db as db


@dataclass(frozen=True)
class FKSuggestion:
    fk_name: str
    child_schema: str
    child_table: str
    parent_schema: str
    parent_table: str
    child_cols: list[str]   # ordered
    parent_cols: list[str]  # ordered
    score: int              # how many child FK cols are present in staged_cols
    reason: str


def _fetch_all_fks_for_child_table(conn, *, child_schema: str, child_table: str):
    """
    Returns rows for ALL FK columns for the given child table.
    One row per (FK, column ordinal).
    """
    sql = r"""
    SELECT
        fk.name AS fk_name,
        sch_child.name AS child_schema,
        tab_child.name AS child_table,
        sch_parent.name AS parent_schema,
        tab_parent.name AS parent_table,
        col_child.name AS child_column,
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
    return db.read_sql(conn, sql, params=[child_schema, child_table])


def suggest_fk_candidates(
    conn,
    *,
    child_schema: str,
    child_table: str,
    staged_cols: Sequence[str],
    require_all_cols_present: bool = False,
    top_n: int = 25,
) -> list[FKSuggestion]:
    """
    Rank FK relationships on the child table by how many FK columns are present
    in staged_cols.

    score = count(child_fk_cols ∩ staged_cols)

    If require_all_cols_present=True, only include FKs where ALL child FK columns
    are present in staged_cols.
    """
    df = _fetch_all_fks_for_child_table(conn, child_schema=child_schema, child_table=child_table)
    if df is None or df.empty:
        return []

    staged_u = {str(c).strip().upper() for c in staged_cols if str(c).strip()}

    out: list[FKSuggestion] = []
    for fk_name, g in df.groupby("fk_name"):
        g = g.sort_values("ordinal")

        child_cols = [str(x).strip() for x in g["child_column"].tolist()]
        parent_cols = [str(x).strip() for x in g["parent_column"].tolist()]
        parent_schema = str(g.iloc[0]["parent_schema"])
        parent_table = str(g.iloc[0]["parent_table"])

        hits = [c for c in child_cols if c.upper() in staged_u]
        score = len(hits)

        if require_all_cols_present and score != len(child_cols):
            continue

        if score == 0:
            reason = "0 FK columns present in staged file (low-confidence)."
        elif score == len(child_cols):
            reason = "All FK columns are present in staged file."
        else:
            reason = f"{score}/{len(child_cols)} FK columns present in staged file."

        out.append(
            FKSuggestion(
                fk_name=str(fk_name),
                child_schema=child_schema,
                child_table=child_table,
                parent_schema=parent_schema,
                parent_table=parent_table,
                child_cols=child_cols,
                parent_cols=parent_cols,
                score=score,
                reason=reason,
            )
        )

    out.sort(key=lambda x: (x.score, x.fk_name), reverse=True)
    return out[: int(top_n)]


def suggest_fk_candidates_step4(conn, *, child_schema: str, child_table: str) -> list[str]:
    """
    Simple helper for the "Auto-tick Treat-as-FK" button:
    returns the set of ALL child FK columns on this table (deduped, ordered).
    """
    df = _fetch_all_fks_for_child_table(conn, child_schema=child_schema, child_table=child_table)
    if df is None or df.empty:
        return []

    cols = [str(x).strip() for x in df["child_column"].tolist()]
    seen = set()
    out: list[str] = []
    for c in cols:
        cu = c.upper()
        if cu not in seen:
            out.append(c)
            seen.add(cu)
    return out


def apply_fk_suggestions_to_map_df(map_df, fk_cols: Sequence[str]):
    """
    Given a mapping DataFrame (from st.data_editor / session_state),
    set treat_as_fk=True for rows where column_name matches any fk_cols
    (case-insensitive). Keeps existing True values.
    """
    if map_df is None:
        return map_df

    fk_u = {str(c).strip().upper() for c in fk_cols if str(c).strip()}
    df = map_df.copy()

    # Ensure column exists
    if "treat_as_fk" not in df.columns:
        df["treat_as_fk"] = False

    def _tick(row):
        cn = str(row.get("column_name", "")).strip().upper()
        if cn in fk_u:
            return True
        return bool(row.get("treat_as_fk", False))

    df["treat_as_fk"] = df.apply(_tick, axis=1)
    return df


def apply_fk_autotick_to_mapping_df(map_df, fk_cols: Sequence[str]):
    """
    Backwards-compatible alias (some page code uses this name).
    """
    return apply_fk_suggestions_to_map_df(map_df, fk_cols)
