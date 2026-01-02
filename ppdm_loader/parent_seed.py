# ppdm_loader/parent_seed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import ppdm_loader.db as db
from ppdm_loader.fk_introspect import FKInfo


@dataclass
class ParentSeedAssessment:
    can_auto_seed: bool
    reason: str
    required_extra_cols: list[str]  # required cols beyond FK cols


def _quote_ident(name: str) -> str:
    name = (name or "").replace("]", "]]")
    return f"[{name}]"


def _fqn(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def parent_table_profile(conn, schema: str, table: str) -> pd.DataFrame:
    """
    Column profile with nullability + default.
    """
    sql = """
    SELECT
        c.name AS column_name,
        c.is_nullable,
        dc.definition AS default_definition
    FROM sys.schemas s
    JOIN sys.tables t ON t.schema_id = s.schema_id
    JOIN sys.columns c ON c.object_id = t.object_id
    LEFT JOIN sys.default_constraints dc
      ON dc.parent_object_id = t.object_id
     AND dc.parent_column_id = c.column_id
    WHERE s.name = ?
      AND t.name = ?
    ORDER BY c.column_id;
    """
    df = db.read_sql(conn, sql, params=[schema, table])
    return df if df is not None else pd.DataFrame()


def assess_parent_auto_seed(conn, fk: FKInfo) -> ParentSeedAssessment:
    """
    Decide if we can auto-insert parent rows using ONLY the FK columns (+ optional common defaults).
    Rule: any NOT NULL column without default that is not part of FK target columns => cannot auto-seed.
    """
    prof = parent_table_profile(conn, fk.parent_schema, fk.parent_table)
    if prof is None or prof.empty:
        return ParentSeedAssessment(False, "Parent table not found / no columns.", [])

    parent_fk_cols = {p.upper() for _, p in fk.pairs}

    required_extra = []
    for _, r in prof.iterrows():
        col = str(r["column_name"])
        col_u = col.upper()
        is_nullable = int(r["is_nullable"])
        has_default = r["default_definition"] is not None and str(r["default_definition"]).strip() != ""

        if col_u in parent_fk_cols:
            continue

        # allow common audit-ish cols to be nullable or defaulted;
        # but if they are NOT NULL with no default, we cannot auto.
        if is_nullable == 0 and not has_default:
            required_extra.append(col)

    if required_extra:
        return ParentSeedAssessment(
            can_auto_seed=False,
            reason="Parent has required (NOT NULL, no default) columns beyond the FK columns.",
            required_extra_cols=required_extra,
        )

    return ParentSeedAssessment(True, "Safe to auto-seed parent from FK key columns.", [])


def compute_missing_parent_keys(
    conn,
    *,
    norm_view_fqn: str,
    fk: FKInfo,
    top_n: int = 2000,
) -> tuple[pd.DataFrame, int]:
    """
    Server-side missing key tuples:
      src = distinct child key tuples from norm view
      missing = src where not exists in parent

    Returns (sample_df, missing_total).
    """
    child_cols = [c for c, _ in fk.pairs]
    parent_cols = [p for _, p in fk.pairs]

    child_sel = ", ".join(f"v.{_quote_ident(c)} AS {_quote_ident(c)}" for c in child_cols)
    not_null = " AND ".join(f"v.{_quote_ident(c)} IS NOT NULL AND LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_quote_ident(c)}))) <> N''" for c in child_cols)

    join_on = " AND ".join(
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), p.{_quote_ident(pcol)}))), N'') = "
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), s.{_quote_ident(ccol)}))), N'')"
        for (ccol, pcol) in fk.pairs
    )

    parent_fqn = _fqn(fk.parent_schema, fk.parent_table)

    sql_sample = f"""
;WITH src AS (
    SELECT DISTINCT {child_sel}
    FROM {norm_view_fqn} v
    WHERE {not_null}
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE NOT EXISTS (
        SELECT 1 FROM {parent_fqn} p
        WHERE {join_on}
    )
)
SELECT TOP ({int(top_n)}) *
FROM missing
ORDER BY 1;
""".strip()

    sql_count = f"""
;WITH src AS (
    SELECT DISTINCT {child_sel}
    FROM {norm_view_fqn} v
    WHERE {not_null}
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE NOT EXISTS (
        SELECT 1 FROM {parent_fqn} p
        WHERE {join_on}
    )
)
SELECT COUNT(*) AS missing_total
FROM missing;
""".strip()

    df_sample = db.read_sql(conn, sql_sample)
    if df_sample is None:
        df_sample = pd.DataFrame()

    df_count = db.read_sql(conn, sql_count)
    missing_total = 0
    if df_count is not None and not df_count.empty:
        missing_total = int(df_count.iloc[0]["missing_total"])

    return df_sample, missing_total


def seed_missing_parent_keys(
    conn,
    *,
    norm_view_fqn: str,
    fk: FKInfo,
    defaults: dict[str, Any] | None = None,  # e.g. {"ACTIVE_IND":"Y"}
) -> int:
    """
    Auto-insert missing parent keys using key tuples from NORM view.
    Only insert FK columns + optional defaults + optional PPDM_GUID if present.
    Returns inserted row count (best effort).
    """
    defaults = defaults or {}

    # profile to detect optional columns
    prof = parent_table_profile(conn, fk.parent_schema, fk.parent_table)
    cols = {str(c).upper() for c in (prof["column_name"].tolist() if prof is not None and not prof.empty else [])}

    parent_fk_cols = [p for _, p in fk.pairs]
    child_fk_cols = [c for c, _ in fk.pairs]

    parent_fqn = _fqn(fk.parent_schema, fk.parent_table)

    insert_cols = list(parent_fk_cols)
    select_exprs = [f"m.{_quote_ident(cc)}" for cc in child_fk_cols]

    # defaults
    for k, v in defaults.items():
        if not k:
            continue
        ku = str(k).strip().upper()
        if ku in cols and ku not in {c.upper() for c in insert_cols}:
            insert_cols.append(str(k).strip())
            if v is None:
                select_exprs.append("NULL")
            else:
                vv = str(v).replace("'", "''")
                select_exprs.append(f"N'{vv}'")

    # PPDM_GUID
    if "PPDM_GUID" in cols and "PPDM_GUID" not in {c.upper() for c in insert_cols}:
        insert_cols.append("PPDM_GUID")
        select_exprs.append("CONVERT(nvarchar(36), NEWID())")

    col_list = ", ".join(_quote_ident(c) for c in insert_cols)
    sel_list = ", ".join(select_exprs)

    # build missing CTE (same as compute)
    child_sel = ", ".join(f"v.{_quote_ident(c)} AS {_quote_ident(c)}" for c in child_fk_cols)
    not_null = " AND ".join(f"v.{_quote_ident(c)} IS NOT NULL AND LTRIM(RTRIM(CONVERT(nvarchar(4000), v.{_quote_ident(c)}))) <> N''" for c in child_fk_cols)

    join_on = " AND ".join(
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), p.{_quote_ident(pcol)}))), N'') = "
        f"NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(4000), m.{_quote_ident(ccol)}))), N'')"
        for (ccol, pcol) in fk.pairs
    )

    sql = f"""
;WITH src AS (
    SELECT DISTINCT {child_sel}
    FROM {norm_view_fqn} v
    WHERE {not_null}
),
missing AS (
    SELECT s.*
    FROM src s
    WHERE NOT EXISTS (
        SELECT 1 FROM {parent_fqn} p
        WHERE {join_on}
    )
)
INSERT INTO {parent_fqn} ({col_list})
SELECT {sel_list}
FROM missing m;
""".strip()

    cur = conn.cursor()
    cur.execute(sql)
    inserted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return int(inserted)
