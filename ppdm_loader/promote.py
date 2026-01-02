# ppdm_loader/promote.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ppdm_loader.db import read_sql, exec_sql


SYSTEM_USER = "Perry M Stokes"

# Audit/system columns we may auto-stamp (only if they exist in the target)
AUDIT_COLS = {
    "PPDM_GUID",
    "ROW_CREATED_BY",
    "ROW_CREATED_DATE",
    "ROW_CHANGED_BY",
    "ROW_CHANGED_DATE",
}

# ---- compatibility constant (app.py expects this) ----
PROMOTE_QC_TABLE = "stg.promote_qc"

# -----------------------------------------------------------------------------
# Promote plan + runner
# -----------------------------------------------------------------------------
@dataclass
class PromotePlan:
    target_fqn: str
    view_name: str
    merge_sql: str


def _qident(name: str) -> str:
    """Bracket-quote an identifier part safely."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("Empty identifier")
    return f"[{name.replace(']', ']]')}]"


def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def _escape_sql_literal(s: str) -> str:
    return (s or "").replace("'", "''")


def _colset(cols: Sequence[str]) -> set:
    return {str(c).strip().upper() for c in cols if str(c).strip()}


def _find_first_present(target_cols_upper: set, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c.upper() in target_cols_upper:
            return c
    return None


def _get_target_columns(conn, target_schema: str, target_table: str) -> pd.DataFrame:
    """
    Pull column list + PK info from SQL Server metadata.
    Keep promote.py standalone (do not depend on introspect module).
    """
    sql = """
    SELECT
        c.name AS column_name,
        t.name AS data_type,
        c.max_length AS max_length,
        c.is_nullable AS is_nullable,
        CASE WHEN i.is_primary_key = 1 THEN 'YES' ELSE 'NO' END AS is_primary_key
    FROM sys.columns c
    JOIN sys.types t
      ON c.user_type_id = t.user_type_id
    JOIN sys.objects o
      ON c.object_id = o.object_id
    LEFT JOIN sys.index_columns ic
      ON ic.object_id = c.object_id AND ic.column_id = c.column_id
    LEFT JOIN sys.indexes i
      ON i.object_id = ic.object_id AND i.index_id = ic.index_id AND i.is_primary_key = 1
    WHERE o.type = 'U'
      AND SCHEMA_NAME(o.schema_id) = ?
      AND o.name = ?
    ORDER BY c.column_id;
    """
    return read_sql(conn, sql, params=[target_schema, target_table])


def _get_pk_columns(conn, target_schema: str, target_table: str) -> List[str]:
    df = _get_target_columns(conn, target_schema, target_table)
    if df is None or df.empty:
        return []
    return (
        df[df["is_primary_key"].astype(str).str.upper().eq("YES")]["column_name"]
        .astype(str)
        .tolist()
    )


def _sql_guid_expr() -> str:
    # PPDM typically stores GUID as nvarchar(36/38). NEWID() is uniqueidentifier.
    return "CONVERT(nvarchar(36), NEWID())"


def build_promote_plan(
    conn,
    view_name: str,
    target_schema: str,
    target_table: str,
    mapped_cols: List[Tuple[str, str]],
    fk_map: Optional[Dict[str, Tuple[str, str, str]]] = None,
    treat_as_fk_cols: Optional[List[str]] = None,
    key_strategy: str = "hash_if_too_long",
    use_nat_suffix: str = "__NAT",
    use_valid_rid: bool = True,
    audit_user: str = SYSTEM_USER,
) -> PromotePlan:
    """
    Generates a MERGE statement:
      - src is the normalized view (optionally filtered to valid RID set)
      - join is on target PK columns
      - INSERT on NOT MATCHED
      - UPDATE on MATCHED

    Enhancements:
      - Auto-generate PPDM_GUID on INSERT if not provided/mapped
      - Auto-stamp ROW_CREATED_BY/DATE and ROW_CHANGED_BY/DATE (if columns exist)

    Critical fix:
      - Do NOT reference src.[PPDM_GUID] (or any column) unless it exists in the view.
      - If the target has PPDM_GUID but the view does not, inject:
            CAST(NULL AS nvarchar(38)) AS [PPDM_GUID]
        so MERGE can still safely reference s.[PPDM_GUID].
    """

    target_fqn = _qfqn(target_schema, target_table)

    # -----------------------------
    # Get target metadata
    # -----------------------------
    tgt_cols_df = _get_target_columns(conn, target_schema, target_table)
    if tgt_cols_df is None or tgt_cols_df.empty:
        raise ValueError(f"No columns found for {target_schema}.{target_table}")

    tgt_cols: List[str] = tgt_cols_df["column_name"].astype(str).tolist()
    tgt_cols_upper = _colset(tgt_cols)

    pk_cols = _get_pk_columns(conn, target_schema, target_table)
    if not pk_cols:
        raise ValueError(
            f"{target_schema}.{target_table} has no primary key. "
            f"Promote requires a PK for MERGE matching."
        )

    # -----------------------------
    # Get VIEW columns (what actually exists)
    # -----------------------------
    def _get_view_columns_upper(conn, view_name: str) -> set:
        # OBJECT_ID works with schema-qualified names (stg.v_seed_norm) and often with unqualified too.
        sql = """
        SELECT c.name AS column_name
        FROM sys.columns c
        WHERE c.object_id = OBJECT_ID(?);
        """
        df = read_sql(conn, sql, params=[view_name])
        if df is None or df.empty:
            return set()
        return {str(x).strip().upper() for x in df["column_name"].tolist() if str(x).strip()}

    view_cols_upper = _get_view_columns_upper(conn, view_name)

    # Normalize mapped_cols into dict: (target_col -> source_col in view)
    mapped_dict: Dict[str, str] = {}
    for t, s in mapped_cols:
        t = str(t or "").strip()
        s = str(s or "").strip()
        if not t:
            continue
        mapped_dict[t] = s  # view already has target-shaped cols (constants handled upstream)

    # IMPORTANT: Remove audit columns from "mapped targets"
    # Promote owns stamping these to avoid double-assign.
    mapped_dict = {k: v for k, v in mapped_dict.items() if k.strip().upper() not in AUDIT_COLS}

    # Determine which audit columns exist in target
    ppdm_guid_col = _find_first_present(tgt_cols_upper, ["PPDM_GUID"])
    row_created_by_col = _find_first_present(tgt_cols_upper, ["ROW_CREATED_BY"])
    row_created_date_col = _find_first_present(tgt_cols_upper, ["ROW_CREATED_DATE"])
    row_changed_by_col = _find_first_present(tgt_cols_upper, ["ROW_CHANGED_BY"])
    row_changed_date_col = _find_first_present(tgt_cols_upper, ["ROW_CHANGED_DATE"])

    # -----------------------------
    # Build SRC CTE SELECT list
    # -----------------------------
    src_select_cols: List[str] = []
    src_seen = set()

    def add_src_col_from_view(colname: str, null_type: str = "nvarchar(4000)"):
        """
        Add a column to the CTE:
          - If it exists in the view:  src.[COL]
          - If it does NOT exist:      CAST(NULL AS <null_type>) AS [COL]
        """
        c = str(colname).strip()
        if not c:
            return
        u = c.upper()
        if u in src_seen:
            return
        src_seen.add(u)

        if u in view_cols_upper:
            src_select_cols.append(f"src.{_qident(c)}")
        else:
            src_select_cols.append(f"CAST(NULL AS {null_type}) AS {_qident(c)}")

    # PKs always need to be present in the CTE output for MERGE joins
    for pk in pk_cols:
        add_src_col_from_view(pk, null_type="nvarchar(4000)")

    # Add mapped targets (only those that actually exist in target; if missing in view, inject NULL)
    for tgt_col in mapped_dict.keys():
        if tgt_col.upper() in tgt_cols_upper:
            # If target column is numeric/date, NULL typing doesn't really matter for NULL itself,
            # but nvarchar(4000) is safe for most of your normalized views.
            add_src_col_from_view(tgt_col, null_type="nvarchar(4000)")

    # Ensure PPDM_GUID is available to MERGE as s.[PPDM_GUID] if target has it
    if ppdm_guid_col:
        # Prefer view column if present, else inject NULL typed correctly
        add_src_col_from_view(ppdm_guid_col, null_type="nvarchar(38)")

    # Source FROM/WHERE
    src_from = f"FROM {view_name} AS src"
    src_where = ""
    if use_valid_rid:
        # NOTE: requires view to have RID column; we don't select RID necessarily, just filter by it.
        # If RID is missing from the view, validation/promote should be fixed upstream.
        src_where = "WHERE EXISTS (SELECT 1 FROM stg.valid_rid v WHERE v.RID = src.RID)"

    src_cte = f"""
WITH src_base AS (
    SELECT
        {", ".join(src_select_cols)}
    {src_from}
    {src_where}
)
""".strip()

    # -----------------------------
    # MERGE join on PK(s)
    # -----------------------------
    on_clause = " AND ".join([f"t.{_qident(pk)} = s.{_qident(pk)}" for pk in pk_cols])

    # -----------------------------
    # INSERT columns + values (dedupe defensively)
    # -----------------------------
    insert_cols: List[str] = []
    insert_vals: List[str] = []
    inserted_upper = set()

    def insert_pair(col: str, val_expr: str):
        cu = str(col).strip().upper()
        if not cu or cu in inserted_upper:
            return
        inserted_upper.add(cu)
        insert_cols.append(_qident(col))
        insert_vals.append(val_expr)

    # PKs always inserted
    for col in pk_cols:
        if col.upper() in tgt_cols_upper:
            insert_pair(col, f"s.{_qident(col)}")

    mapped_targets_upper = _colset(mapped_dict.keys())
    pk_upper = _colset(pk_cols)

    # Insert mapped cols + PPDM_GUID (special) but do NOT insert audit columns via mapping
    for col in tgt_cols:
        u = col.upper()
        if u in pk_upper:
            continue

        # PPDM_GUID special handling
        if ppdm_guid_col and u == ppdm_guid_col.upper():
            # s.[PPDM_GUID] is guaranteed to exist now (real or injected NULL)
            insert_pair(col, f"COALESCE(NULLIF(s.{_qident(col)}, N''), {_sql_guid_expr()})")
            continue

        # Never map-insert audit fields; Promote stamps them
        if u in AUDIT_COLS:
            continue

        # Insert if mapped
        if u in mapped_targets_upper:
            insert_pair(col, f"s.{_qident(col)}")

    # Stamp audit fields on INSERT (only if those columns exist)
    aud_user_lit = f"N'{_escape_sql_literal(audit_user)}'"
    if row_created_by_col:
        insert_pair(row_created_by_col, aud_user_lit)
    if row_created_date_col:
        insert_pair(row_created_date_col, "SYSUTCDATETIME()")
    if row_changed_by_col:
        insert_pair(row_changed_by_col, aud_user_lit)
    if row_changed_date_col:
        insert_pair(row_changed_date_col, "SYSUTCDATETIME()")

    if not insert_cols:
        raise ValueError("No insertable columns resolved (check mappings and target metadata).")

    # -----------------------------
    # UPDATE set list (dedupe defensively)
    # -----------------------------
    update_sets: List[str] = []
    updated_upper = set()

    def add_update_set(col: str, expr: str):
        cu = str(col).strip().upper()
        if not cu or cu in updated_upper:
            return
        updated_upper.add(cu)
        update_sets.append(f"t.{_qident(col)} = {expr}")

    # Update mapped columns only (excluding PKs, created_*, and audit fields handled below)
    for col in tgt_cols:
        u = col.upper()
        if u in pk_upper:
            continue

        # Never update created_* (typical audit practice)
        if row_created_by_col and u == row_created_by_col.upper():
            continue
        if row_created_date_col and u == row_created_date_col.upper():
            continue

        # PPDM_GUID: keep existing if set; otherwise set from src or generate
        if ppdm_guid_col and u == ppdm_guid_col.upper():
            # s.[PPDM_GUID] is guaranteed to exist now (real or injected NULL)
            add_update_set(
                col,
                f"COALESCE(NULLIF(t.{_qident(col)}, N''), NULLIF(s.{_qident(col)}, N''), {_sql_guid_expr()})",
            )
            continue

        # Do not update audit fields via mapping
        if u in {"ROW_CHANGED_BY", "ROW_CHANGED_DATE"}:
            continue

        if u in mapped_targets_upper:
            add_update_set(col, f"s.{_qident(col)}")

    # Stamp change fields on UPDATE (always if columns exist)
    if row_changed_by_col:
        add_update_set(row_changed_by_col, aud_user_lit)
    if row_changed_date_col:
        add_update_set(row_changed_date_col, "SYSUTCDATETIME()")

    update_clause = ""
    if update_sets:
        update_clause = "WHEN MATCHED THEN UPDATE SET\n        " + ",\n        ".join(update_sets)

    merge_sql = f"""
SET NOCOUNT ON;

{src_cte}

MERGE {target_fqn} AS t
USING src_base AS s
ON {on_clause}
WHEN NOT MATCHED BY TARGET THEN
    INSERT ({", ".join(insert_cols)})
    VALUES ({", ".join(insert_vals)})
{update_clause}
;
""".strip()

    return PromotePlan(
        target_fqn=f"{target_schema}.{target_table}",
        view_name=view_name,
        merge_sql=merge_sql,
    )



def run_promote(conn, plan: PromotePlan) -> None:
    exec_sql(conn, plan.merge_sql)


# -----------------------------------------------------------------------------
# QC helpers (simple + practical)
# -----------------------------------------------------------------------------
def read_promote_qc(conn, target_fqn: str, top_n: int = 20) -> pd.DataFrame:
    """
    Very lightweight QC summary:
      - rowcount in target
      - top N rows (arbitrary sample)
    """
    if "." in target_fqn:
        schema, table = target_fqn.split(".", 1)
    else:
        schema, table = "dbo", target_fqn

    cnt = read_sql(conn, f"SELECT COUNT_BIG(1) AS [rowcount] FROM {_qfqn(schema, table)};")
    samp = read_sql(conn, f"SELECT TOP ({int(top_n)}) * FROM {_qfqn(schema, table)};")

    out = pd.DataFrame(
        {"metric": ["rowcount"], "value": [int(cnt.iloc[0]["rowcount"]) if not cnt.empty else None]}
    )
    out.attrs["sample"] = samp
    return out


def promote_qc_adjacent(
    conn,
    view_name: str,
    target_schema: str,
    target_table: str,
    mapped_target_cols: List[str],
    use_valid_rid: bool = True,
    top_n: int = 200,
) -> pd.DataFrame:
    """
    Adjacent SRC vs TGT (limited set) using PK join.
    Dedupes columns so the CTE doesn't select the same column twice.
    """
    pk_cols = _get_pk_columns(conn, target_schema, target_table)
    if not pk_cols:
        return pd.DataFrame()

    tgt_fqn = _qfqn(target_schema, target_table)

    pk_upper = {c.strip().upper() for c in pk_cols if str(c).strip()}

    raw_show = [str(c).strip() for c in (mapped_target_cols or []) if str(c).strip()]
    raw_show = raw_show[:50]

    show_cols: List[str] = []
    seen = set()
    for c in raw_show:
        cu = c.upper()
        if cu in pk_upper:
            continue
        if cu in seen:
            continue
        seen.add(cu)
        show_cols.append(c)

    cte_cols = pk_cols + show_cols

    where = ""
    if use_valid_rid:
        where = "WHERE EXISTS (SELECT 1 FROM stg.valid_rid v WHERE v.RID = src.RID)"

    join_on = " AND ".join([f"t.{_qident(pk)} = s.{_qident(pk)}" for pk in pk_cols])

    sel: List[str] = []
    for pk in pk_cols:
        sel.append(f"s.{_qident(pk)} AS {_qident(pk)}")

    for c in show_cols[:20]:
        sel.append(f"s.{_qident(c)} AS [SRC_{c}]")
        sel.append(f"t.{_qident(c)} AS [TGT_{c}]")

    sql = f"""
WITH s AS (
    SELECT TOP ({int(top_n)})
        {", ".join([f"src.{_qident(c)}" for c in cte_cols])}
    FROM {view_name} src
    {where}
)
SELECT
    {", ".join(sel)}
FROM s
JOIN {tgt_fqn} t
  ON {join_on};
""".strip()

    return read_sql(conn, sql)
