# ppdm_loader/seed.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from ppdm_loader.db import read_sql, exec_sql


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------
def _object_id(conn, name: str) -> Optional[int]:
    df = read_sql(conn, "SELECT OBJECT_ID(?) AS oid", params=[name])
    if df is None or df.empty:
        return None
    v = df["oid"].iloc[0]
    return int(v) if v is not None else None


def _resolve_object_name(conn, name: str) -> Optional[str]:
    nm = name.strip()
    if _object_id(conn, nm) is not None:
        return nm

    if "." not in nm:
        for schema in ("stg", "dbo"):
            cand = f"{schema}.{nm}"
            if _object_id(conn, cand) is not None:
                return cand
    return None


def _view_column_set(conn, view_name: str) -> set[str]:
    obj = _resolve_object_name(conn, view_name)
    if not obj:
        return set()

    df = read_sql(
        conn,
        "SELECT c.name FROM sys.columns c WHERE c.object_id = OBJECT_ID(?)",
        params=[obj],
    )
    if df is None or df.empty:
        return set()

    return set(df["name"].astype(str).str.lower())


def _col_type_len_chars(conn, schema: str, table: str, col: str) -> tuple[str, Optional[int]]:
    df = read_sql(
        conn,
        """
        SELECT t.name AS type_name, c.max_length
        FROM sys.columns c
        JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID(?)
          AND c.name = ?
        """,
        params=[f"{schema}.{table}", col],
    )
    if df is None or df.empty:
        return ("", None)

    type_name = str(df["type_name"].iloc[0]).lower()
    max_len = int(df["max_length"].iloc[0])

    if type_name in ("nvarchar", "nchar"):
        return (type_name, None if max_len < 0 else max_len // 2)
    if type_name in ("varchar", "char"):
        return (type_name, None if max_len < 0 else max_len)

    return (type_name, None)


def _parent_pk_only_is_safe(conn, schema: str, table: str, pk: str) -> tuple[bool, str]:
    """
    True if inserting PK alone will not violate NOT NULL constraints.
    """
    df = read_sql(
        conn,
        """
        SELECT
            c.name,
            c.is_nullable,
            c.is_identity,
            c.is_computed,
            dc.object_id AS has_default
        FROM sys.columns c
        LEFT JOIN sys.default_constraints dc
          ON dc.parent_object_id = c.object_id
         AND dc.parent_column_id = c.column_id
        WHERE c.object_id = OBJECT_ID(?)
        """,
        params=[f"{schema}.{table}"],
    )
    if df is None or df.empty:
        return False, "Parent table not found"

    required = df[
        (df["name"].str.lower() != pk.lower())
        & (df["is_nullable"] == 0)
        & (df["is_identity"] == 0)
        & (df["is_computed"] == 0)
        & (df["has_default"].isna())
    ]

    if not required.empty:
        return False, "Other required columns without defaults"

    return True, "OK"


# -----------------------------------------------------------------------------
# Main seeding function
# -----------------------------------------------------------------------------
# =============================================================================
# COMPLETE UPDATED FUNCTION (DROP-IN REPLACEMENT)
# - Preserves your current behavior 1:1 by default
# - Adds OPTIONAL seed-catalog integration via resolve_seed_spec
# - Adds OPTIONAL static seeding (mode="static") when catalog says so
#
# You can paste this over your existing seed_parent_tables_from_view().
#
# New optional parameter:
#   resolve_seed_spec(parent_fqn, child_col, pk) -> dict/spec
#     Expected keys (all optional; sensible defaults applied):
#       mode: "from_view" | "static"
#       key_strategy: "hash_if_too_long" | "hash_always" | "raw"
#       pk_from: list[str]   # source cols used to build key_expr (default = [child_col])
#       defaults: dict[str, Any]  # constant values for extra parent cols (optional)
#       static_rows: list[dict]   # rows to insert when mode="static"
#
# If resolve_seed_spec is not provided, everything behaves exactly as before.
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple, Callable
import pandas as pd


def seed_parent_tables_from_view(
    conn,
    view_name: str,
    fk_map: Dict[str, Tuple[str, str, str]],
    treat_as_fk_cols: Optional[List[str]] = None,
    parent_mappings: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    key_strategy: str = "hash_if_too_long",
    use_nat_suffix: str = "__NAT",
    resolve_seed_spec: Optional[Callable[[str, str, str], Any]] = None,
) -> pd.DataFrame:
    """
    Seed FK parent tables from a normalized view.

    IMPORTANT RULES:
    - Only FK columns that are BOTH:
        * treat_as_fk == True
        * mapped in the primary grid
      are considered.
    - Parent tables are seeded with PK ONLY unless extra mappings are provided.
    - OPTIONAL: seed catalog integration via resolve_seed_spec().
      If catalog says mode="static", insert static rows (reference tables).
    """

    treat_as_fk_cols = treat_as_fk_cols or []
    parent_mappings = parent_mappings or {}

    view_cols = _view_column_set(conn, view_name)

    def _as_dict_spec(spec_obj: Any) -> Dict[str, Any]:
        """
        Accept either dict-like or dataclass-like spec.
        """
        if spec_obj is None:
            return {}
        if isinstance(spec_obj, dict):
            return spec_obj
        # dataclass / object with attributes
        out = {}
        for k in ("mode", "key_strategy", "pk_from", "defaults", "static_rows"):
            if hasattr(spec_obj, k):
                out[k] = getattr(spec_obj, k)
        return out

    def _seed_static_rows(ps: str, pt: str, rows: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Minimal static seeding: inserts rows if missing by first key column.
        Returns (success, message).
        """
        if not rows:
            return True, "SKIP: no static rows"

        # stable union of columns
        cols = list(rows[0].keys())
        for r in rows[1:]:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)

        if not cols:
            return True, "SKIP: static rows had no columns"

        key_col = cols[0]  # existence check
        col_list = ", ".join(f"[{c}]" for c in cols)

        def _lit(v: Any) -> str:
            if v is None:
                return "NULL"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return str(v)
            return "N'" + str(v).replace("'", "''") + "'"

        values_lines = []
        for r in rows:
            vals = [_lit(r.get(c)) for c in cols]
            values_lines.append("(" + ", ".join(vals) + ")")

        values_sql = ",\n            ".join(values_lines)

        sql = f"""
        SET NOCOUNT ON;

        INSERT INTO [{ps}].[{pt}] ({col_list})
        SELECT v.*
        FROM (VALUES
            {values_sql}
        ) AS v({col_list})
        WHERE NOT EXISTS (
            SELECT 1
            FROM [{ps}].[{pt}] t
            WHERE t.[{key_col}] = v.[{key_col}]
        );
        """

        exec_sql(conn, sql)
        return True, "SEEDED (static)"

    report_rows: list[dict] = []

    for child_col, (ps, pt, pk) in fk_map.items():
        if child_col not in treat_as_fk_cols:
            continue  # safety guard

        parent_fqn = f"{ps}.{pt}"

        # ---------------------------------------------------------------------
        # Resolve per-parent seeding spec (catalog-first) if provided
        # ---------------------------------------------------------------------
        spec = {}
        if resolve_seed_spec:
            try:
                spec = _as_dict_spec(resolve_seed_spec(parent_fqn, child_col, pk))
            except Exception as e:
                # Do not fail seeding because resolver failed; just report it and fall back
                spec = {}
                report_rows.append({
                    "child_column": child_col,
                    "parent_table": parent_fqn,
                    "parent_key": pk,
                    "status": f"RESOLVER WARNING: {e} (fell back)",
                    "view_column_used": "",
                    "extra_columns": "",
                })

        mode = (spec.get("mode") or "from_view").strip().lower()
        eff_key_strategy = (spec.get("key_strategy") or key_strategy).strip()
        pk_from = spec.get("pk_from")  # list[str] of view columns for hashing; optional
        defaults = spec.get("defaults") or {}
        static_rows = spec.get("static_rows")

        # ---------------------------------------------------------------------
        # If catalog says static, seed static rows and continue
        # ---------------------------------------------------------------------
        if mode == "static":
            try:
                ok, msg = _seed_static_rows(ps, pt, static_rows or [])
                report_rows.append({
                    "child_column": child_col,
                    "parent_table": parent_fqn,
                    "parent_key": pk,
                    "status": msg if ok else f"ERROR: {msg}",
                    "view_column_used": "",
                    "extra_columns": "",
                })
            except Exception as e:
                report_rows.append({
                    "child_column": child_col,
                    "parent_table": parent_fqn,
                    "parent_key": pk,
                    "status": f"ERROR (static): {e}",
                    "view_column_used": "",
                    "extra_columns": "",
                })
            continue

        # ---------------------------------------------------------------------
        # Prefer NAT column if present (same as your current logic)
        # ---------------------------------------------------------------------
        nat_col = f"{child_col}{use_nat_suffix}"
        norm_col = nat_col if nat_col.lower() in view_cols else child_col

        ok, why = _parent_pk_only_is_safe(conn, ps, pt, pk)
        if not ok:
            report_rows.append({
                "child_column": child_col,
                "parent_table": parent_fqn,
                "parent_key": pk,
                "status": f"SKIP: {why}",
                "view_column_used": norm_col,
                "extra_columns": "",
            })
            continue

        # Use existing PK column type/len on parent table
        _, key_len = _col_type_len_chars(conn, ps, pt, pk)

        # Base NAT expression from the chosen norm_col (your current behavior)
        nat_expr = f"LTRIM(RTRIM(CAST(v.[{norm_col}] AS nvarchar(4000))))"

        # ---------------------------------------------------------------------
        # Build key expression
        # - default behavior (no spec or no pk_from) = your current nat_expr-based logic
        # - if pk_from is provided, hash/concat based on those view columns (catalog override)
        # ---------------------------------------------------------------------
        def _expr_for_view_col(colname: str) -> str:
            return f"LTRIM(RTRIM(CAST(v.[{colname}] AS nvarchar(4000))))"

        if pk_from:
            # Build a concatenated NAT string from pk_from columns (COALESCE to stabilize)
            pk_cols = [c for c in pk_from if isinstance(c, str) and c.strip()]
            # only keep columns that actually exist in the view
            pk_cols = [c for c in pk_cols if c.lower() in view_cols]
            if not pk_cols:
                # fall back to norm_col if pk_from invalid
                pk_cols = [norm_col]

            parts = [f"COALESCE({_expr_for_view_col(c)}, N'')" for c in pk_cols]
            nat_expr_for_key = " + N'|' + ".join(parts) if parts else nat_expr
        else:
            nat_expr_for_key = nat_expr

        # Key expression respects parent PK length where known (your current behavior)
        if eff_key_strategy == "raw" or key_len is None:
            key_expr = nat_expr_for_key
        elif eff_key_strategy == "hash_always":
            key_expr = f"""
            LEFT(CONVERT(varchar(40),
                 HASHBYTES('SHA1', CONVERT(nvarchar(4000), {nat_expr_for_key})), 2),
                 {key_len})
            """
        else:
            # hash_if_too_long
            key_expr = f"""
            CASE
              WHEN LEN({nat_expr_for_key}) <= {key_len} THEN {nat_expr_for_key}
              ELSE LEFT(CONVERT(varchar(40),
                   HASHBYTES('SHA1', CONVERT(nvarchar(4000), {nat_expr_for_key})), 2),
                   {key_len})
            END
            """

        # ---------------------------------------------------------------------
        # Extra mapped parent columns (ONLY if user explicitly mapped them)
        # plus catalog defaults (constants)
        # ---------------------------------------------------------------------
        mappings = parent_mappings.get(parent_fqn, []) or parent_mappings.get(parent_fqn.lower(), [])

        extra_cols = []
        extra_vals = []

        seen = {pk.lower()}

        # Catalog defaults first (do not override PK; do not duplicate)
        for tgt, const_val in (defaults or {}).items():
            tgt = (str(tgt) or "").strip()
            if not tgt:
                continue
            if tgt.lower() in seen:
                continue
            if tgt.lower() == pk.lower():
                continue
            seen.add(tgt.lower())
            extra_cols.append(f"[{tgt}]")
            # constants: MIN(N'val') is fine for grouped query
            val_sql = "N'" + str(const_val).replace("'", "''") + "'" if const_val is not None else "NULL"
            extra_vals.append(f"MIN({val_sql}) AS [{tgt}]")

        # User-mapped extras from view
        for m in (mappings or []):
            tgt = (m.get("target_column") or "").strip()
            src = (m.get("source_column") or "").strip()

            if not tgt or not src:
                continue
            if tgt.lower() in seen:
                continue
            if src.lower() not in view_cols:
                continue
            if tgt.lower() == pk.lower():
                continue

            seen.add(tgt.lower())

            ttype, tlen = _col_type_len_chars(conn, ps, pt, tgt)
            expr = f"LTRIM(RTRIM(CAST(v.[{src}] AS nvarchar(4000))))"
            if tlen:
                expr = f"LEFT({expr}, {tlen})"

            extra_cols.append(f"[{tgt}]")
            extra_vals.append(f"MIN({expr}) AS [{tgt}]")

        insert_cols = ", ".join([f"[{pk}]"] + extra_cols)
        select_cols = ", ".join(
            [f"k AS [{pk}]"] +
            [f"[{c.strip('[]')}]" for c in extra_cols]
        )

        # IMPORTANT: keep your original "view_column_used" semantics:
        # filter uses nat_expr from norm_col (child column) so FK values come from the actual FK mapping
        sql = f"""
        ;WITH src AS (
            SELECT
                {key_expr} AS k
                {"," if extra_vals else ""} {", ".join(extra_vals)}
            FROM {view_name} v
            WHERE {nat_expr} IS NOT NULL
              AND {nat_expr} <> ''
            GROUP BY {key_expr}
        )
        INSERT INTO [{ps}].[{pt}] ({insert_cols})
        SELECT {select_cols}
        FROM src
        WHERE k IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM [{ps}].[{pt}] t
              WHERE t.[{pk}] = k
          );
        """

        exec_sql(conn, sql)

        report_rows.append({
            "child_column": child_col,
            "parent_table": parent_fqn,
            "parent_key": pk,
            "status": "SEEDED (PK only)" + (" + extras" if extra_cols else "") + (" [catalog]" if resolve_seed_spec else ""),
            "view_column_used": norm_col,
            "extra_columns": ", ".join(extra_cols) if extra_cols else "",
        })

    return pd.DataFrame(report_rows)

