# ppdm_loader/normalize.py
from __future__ import annotations

import re
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd


def _sql_literal(val: str) -> str:
    """Safe SQL unicode string literal."""
    s = "" if val is None else str(val)
    s = s.replace("'", "''")
    return f"N'{s}'"


def _is_string_type(dt: str) -> bool:
    t = (dt or "").lower()
    return any(x in t for x in ["varchar", "nvarchar", "char", "nchar", "text", "ntext"])


def _is_numeric_type(dt: str) -> bool:
    t = (dt or "").lower()
    return any(x in t for x in ["int", "bigint", "smallint", "tinyint", "numeric", "decimal", "float", "real", "money", "smallmoney"])


def _is_date_type(dt: str) -> bool:
    t = (dt or "").lower()
    return any(x in t for x in ["date", "datetime", "datetime2", "smalldatetime", "time"])


def _parse_char_len(dt: str) -> Optional[int]:
    """
    Parse nvarchar(240) / varchar(50) / nchar(10) etc.
    Returns max characters (not bytes). Returns None for MAX or unknown.
    """
    t = (dt or "").strip().lower()
    m = re.search(r"\((max|\d+)\)", t)
    if not m:
        return None
    g = m.group(1)
    if g == "max":
        return None
    try:
        return int(g)
    except Exception:
        return None


def _trim_expr(expr: str) -> str:
    return f"LTRIM(RTRIM(CAST({expr} AS nvarchar(4000))))"


def _cast_expr(base_expr: str, target_sql_type: str) -> str:
    """
    Cast/convert the base expr into the target type safely.
    - strings: trim (and leave as string)
    - numeric/date: TRY_CAST so view doesn't fail on dirty values
    - otherwise: leave as-is
    """
    dt = (target_sql_type or "").strip()
    if not dt:
        dt = "nvarchar(4000)"

    if _is_string_type(dt):
        return _trim_expr(base_expr)

    if _is_numeric_type(dt) or _is_date_type(dt):
        # safe cast, avoids blowing up the view on bad rows
        return f"TRY_CAST({_trim_expr(base_expr)} AS {dt})"

    # fallback
    return base_expr


def _cast_constant_expr(const_val: str, target_sql_type: str) -> str:
    """
    Cast constant to target type (safely).
    Use TRY_CAST for numeric/date; strings get trimmed.
    """
    dt = (target_sql_type or "").strip() or "nvarchar(4000)"
    lit = _sql_literal(const_val)

    if _is_string_type(dt):
        return _trim_expr(lit)

    if _is_numeric_type(dt) or _is_date_type(dt):
        return f"TRY_CAST({_trim_expr(lit)} AS {dt})"

    return f"CAST({lit} AS {dt})"


def build_primary_norm_view_sql(
    primary_schema: str,
    primary_table: str,
    cols_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    treat_as_fk_cols: List[str],
    pk_hash_enabled: bool = False,
    pk_hash_src: Optional[str] = None,
    use_nat_suffix: str = "__NAT",
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Build the normalized view SQL.

    Supports:
      - mapping_df.source_column (from stg.v_raw_with_rid)
      - mapping_df.constant_value (literal injected here)
      - treat_as_fk: emits both [COL] and [COL__NAT] in view
    """
    # View name always in stg schema
    view_name = f"stg.v_norm_{primary_schema}_{primary_table}".replace(".", "_")
    view_fqn = view_name

    df = mapping_df.copy()
    if "constant_value" not in df.columns:
        df["constant_value"] = ""

    df["source_column"] = df["source_column"].fillna("").astype(str).str.strip()
    df["constant_value"] = df["constant_value"].fillna("").astype(str).str.strip()
    df["column_name"] = df["column_name"].astype(str).str.strip()

    # Column meta lookup
    colmeta: Dict[str, Dict[str, str]] = {}
    if cols_df is not None and not cols_df.empty and "column_name" in cols_df.columns:
        for _, r in cols_df.iterrows():
            cn = str(r["column_name"]).strip()
            colmeta[cn.upper()] = {
                "data_type": str(r.get("data_type") or r.get("type_name") or ""),
            }

    select_exprs: List[str] = []
    maxlen_checks: List[Dict[str, Any]] = []

    # RID always
    select_exprs.append("v.RID AS [RID]")

    # Make FK treat-set case-insensitive
    treat_set = {str(c).strip().upper() for c in (treat_as_fk_cols or []) if str(c).strip()}

    for _, r in df.iterrows():
        tgt = str(r["column_name"]).strip()
        if not tgt:
            continue

        src = str(r.get("source_column") or "").strip()
        const = str(r.get("constant_value") or "").strip()

        dt = (colmeta.get(tgt.upper(), {}).get("data_type") or "").strip()

        if const:
            base_expr = _cast_constant_expr(const, dt)
            nat_base = _trim_expr(_sql_literal(const))
        elif src:
            base_expr = f"v.[{src}]"
            nat_base = _trim_expr(base_expr)
        else:
            # unmapped; skip
            continue

        # Cast safely according to target type
        expr_norm = _cast_expr(base_expr, dt)

        # Main column
        select_exprs.append(f"{expr_norm} AS [{tgt}]")

        # NAT shadow for FK columns (always trimmed string form)
        if tgt.upper() in treat_set:
            select_exprs.append(f"{nat_base} AS [{tgt}{use_nat_suffix}]")

        # Optional max-len checks for strings
        if _is_string_type(dt):
            max_chars = _parse_char_len(dt)
            if max_chars is not None:
                maxlen_checks.append(
                    {
                        "column_name": tgt,
                        "max_chars": max_chars,
                        "expr": f"LEN(CAST([{tgt}] AS nvarchar(4000))) > {max_chars}",
                    }
                )

    sql = f"""
SET NOCOUNT ON;

IF SCHEMA_ID('stg') IS NULL
    EXEC('CREATE SCHEMA stg');

CREATE OR ALTER VIEW {view_fqn} AS
SELECT
    {",\n    ".join(select_exprs)}
FROM stg.v_raw_with_rid AS v
;
""".strip()

    return sql, view_fqn, maxlen_checks
