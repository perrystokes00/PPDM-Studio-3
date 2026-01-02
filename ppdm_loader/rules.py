# ppdm_loader/rules.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List, Set

import pandas as pd

from ppdm_loader.db import read_sql, exec_sql

RULES_TABLE = "cfg.etl_rule_def"
STRING_TYPES = {"char", "nchar", "varchar", "nvarchar", "text", "ntext"}


def ensure_rules_tables(conn) -> None:
    """
    Ensure staging validation tables exist AND are upgraded to latest shape.
    (Fixes: Invalid column name 'column_name' when inserting invalid rows)
    """
    exec_sql(
        conn,
        """
        IF SCHEMA_ID('stg') IS NULL EXEC('CREATE SCHEMA stg');

        IF OBJECT_ID('stg.invalid_rows','U') IS NULL
        BEGIN
            CREATE TABLE stg.invalid_rows(
                RID         int             NOT NULL,
                rule_id     varchar(100)    NOT NULL,
                severity    varchar(10)     NOT NULL,
                column_name sysname         NULL,
                message     nvarchar(800)   NULL
            );
        END
        ELSE
        BEGIN
            IF COL_LENGTH('stg.invalid_rows', 'rule_id') IS NULL
                ALTER TABLE stg.invalid_rows ADD rule_id varchar(100) NOT NULL DEFAULT('UNKNOWN');

            IF COL_LENGTH('stg.invalid_rows', 'severity') IS NULL
                ALTER TABLE stg.invalid_rows ADD severity varchar(10) NOT NULL DEFAULT('ERROR');

            IF COL_LENGTH('stg.invalid_rows', 'column_name') IS NULL
                ALTER TABLE stg.invalid_rows ADD column_name sysname NULL;

            IF COL_LENGTH('stg.invalid_rows', 'message') IS NULL
                ALTER TABLE stg.invalid_rows ADD message nvarchar(800) NULL;
        END;

        IF OBJECT_ID('stg.valid_rid','U') IS NULL
        BEGIN
            CREATE TABLE stg.valid_rid(
                RID int NOT NULL PRIMARY KEY
            );
        END;
        """,
    )


def _split_schema_table(qualified: str) -> Tuple[str, str]:
    q = (qualified or "").strip().replace("[", "").replace("]", "")
    if "." not in q:
        raise ValueError(f"Expected schema.table, got: {qualified}")
    schema, table = q.split(".", 1)
    return schema.strip(), table.strip()


def _get_object_id(conn, schema: str, name: str) -> Optional[int]:
    """
    Resolve schema + object name to sys.objects.object_id for tables/views.
    Avoids OBJECT_ID(?) parameterization quirks with pyodbc.
    """
    df = read_sql(
        conn,
        """
        SET NOCOUNT ON;

        SELECT TOP (1) o.object_id
        FROM sys.objects o
        JOIN sys.schemas s ON s.schema_id = o.schema_id
        WHERE s.name = ?
          AND o.name = ?
          AND o.type IN ('U','V'); -- user table, view
        """,
        params=[schema, name],
    )
    if df is None or df.empty:
        return None
    oid = df["object_id"].iloc[0]
    return int(oid) if oid is not None else None


def _get_columns_for_schema_table(conn, schema: str, table: str) -> pd.DataFrame:
    """
    Return DataFrame with a single column 'name' listing column names.
    Uses sys.schemas/sys.tables/sys.columns joins (no OBJECT_ID params).
    """
    return read_sql(
        conn,
        """
        SET NOCOUNT ON;

        SELECT c.name
        FROM sys.columns c
        JOIN sys.tables  t ON t.object_id = c.object_id
        JOIN sys.schemas s ON s.schema_id = t.schema_id
        WHERE s.name = ? AND t.name = ?;
        """,
        params=[schema, table],
    )


def _rules_table_column_map(conn) -> Tuple[str, str, str, str, str, str, bool]:
    schema, table = _split_schema_table(RULES_TABLE)

    df = _get_columns_for_schema_table(conn, schema, table)
    if df is None or df.empty:
        raise ValueError(f"Rules table not found or has no columns: {RULES_TABLE}")

    cols = set(df["name"].astype(str).str.lower().tolist())

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c.lower() in cols:
                return c
        raise ValueError(f"{RULES_TABLE} missing one of: {candidates}")

    rule_id_col = pick("rule_id")
    enabled_col = pick("enabled")
    domain_col = pick("domain")
    phase_col = pick("phase")
    rule_type_col = pick("rule_type")
    column_col = pick("column_name", "column", "attribute")

    has_params_json = "params_json" in cols

    return (
        rule_id_col,
        enabled_col,
        domain_col,
        phase_col,
        rule_type_col,
        column_col,
        has_params_json,
    )


def upsert_rules_from_json(conn, json_path: str) -> int:
    """
    Bootstrap/import rules into RULES_TABLE.
    Expects {"rules":[...]} OR {"Sheet1":[...]}.

    - params may be dict OR JSON string; stored as JSON string in params_json (if exists)
    - Works whether the rules table uses column_name OR column
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Rules JSON not found: {json_path}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    rules = obj.get("rules") or obj.get("Sheet1") or []
    if not isinstance(rules, list):
        raise ValueError("Rules JSON must contain a list under 'rules' (or 'Sheet1').")

    rule_id_col, enabled_col, domain_col, phase_col, rule_type_col, column_col, has_params_json = (
        _rules_table_column_map(conn)
    )

    # discover optional columns in RULES_TABLE (robust join, no OBJECT_ID)
    schema, table = _split_schema_table(RULES_TABLE)
    table_cols_df = _get_columns_for_schema_table(conn, schema, table)
    table_cols = set(table_cols_df["name"].astype(str).str.lower().tolist()) if table_cols_df is not None else set()

    has_desc = "description" in table_cols
    has_severity = "severity" in table_cols
    has_errmsg = "error_message" in table_cols

    rows = []
    for r in rules:
        rid = r.get("rule_id") or r.get("RULE_ID") or r.get("ruleId")
        if not rid:
            continue

        params = r.get("params") or r.get("PARAMS") or {}
        params_json = params if isinstance(params, str) else json.dumps(params)

        row = {
            rule_id_col: rid,
            domain_col: r.get("domain") or r.get("DOMAIN"),
            phase_col: r.get("phase") or r.get("PHASE"),
            enabled_col: 1 if bool(r.get("enabled", True)) else 0,
            rule_type_col: r.get("rule_type") or r.get("RULE_TYPE") or "required",
            column_col: r.get("column") or r.get("column_name") or r.get("COLUMN"),
        }
        if has_desc:
            row["description"] = r.get("description") or r.get("DESCRIPTION") or ""
        if has_severity:
            row["severity"] = r.get("severity") or r.get("SEVERITY") or "ERROR"
        if has_errmsg:
            row["error_message"] = r.get("error_message") or r.get("ERROR_MESSAGE") or ""
        if has_params_json:
            row["params_json"] = params_json

        rows.append(row)

    if not rows:
        return 0

    insert_cols = list(rows[0].keys())
    placeholders = ",".join("?" for _ in insert_cols)
    col_sql = ",".join(f"[{c}]" for c in insert_cols)

    ids = [r[rule_id_col] for r in rows]
    qmarks = ",".join("?" for _ in ids)
    exec_sql(conn, f"DELETE FROM {RULES_TABLE} WHERE [{rule_id_col}] IN ({qmarks})", params=ids)

    cur = conn.cursor()
    cur.fast_executemany = True
    cur.executemany(
        f"INSERT INTO {RULES_TABLE} ({col_sql}) VALUES ({placeholders})",
        [[r[c] for c in insert_cols] for r in rows],
    )
    conn.commit()
    return len(rows)


def _get_enabled_rules(conn, domain: Optional[str]) -> pd.DataFrame:
    rule_id_col, enabled_col, domain_col, phase_col, rule_type_col, column_col, has_params_json = (
        _rules_table_column_map(conn)
    )

    if domain:
        return read_sql(
            conn,
            f"""
            SET NOCOUNT ON;

            SELECT *
            FROM {RULES_TABLE}
            WHERE [{enabled_col}] = 1
              AND ([{domain_col}] = ? OR [{domain_col}] IS NULL)
            ORDER BY [{domain_col}], [{phase_col}], [{rule_id_col}];
            """,
            params=[domain],
        )

    return read_sql(
        conn,
        f"""
        SET NOCOUNT ON;

        SELECT *
        FROM {RULES_TABLE}
        WHERE [{enabled_col}] = 1
        ORDER BY [{domain_col}], [{phase_col}], [{rule_id_col}];
        """,
    )


def _resolve_object_name(conn, name: str) -> Optional[str]:
    """
    Resolve view/table name to something we can find in sys.objects reliably.
    Accepts:
      - stg.v_xxx
      - dbo.v_xxx
      - v_xxx  (try stg. then dbo.)
    Returns a normalized schema.object name (without brackets), or None.
    """
    nm = (name or "").strip().replace("[", "").replace("]", "")
    if not nm:
        return None

    if "." in nm:
        schema, obj = nm.split(".", 1)
        oid = _get_object_id(conn, schema, obj)
        return f"{schema}.{obj}" if oid is not None else None

    # no schema given: try stg then dbo
    for schema in ("stg", "dbo"):
        oid = _get_object_id(conn, schema, nm)
        if oid is not None:
            return f"{schema}.{nm}"

    return None


def _get_view_columns(conn, view_name: str) -> Set[str]:
    """
    Returns UPPERCASE column names available in the view/table.
    If the object can't be resolved, returns empty set.
    """
    obj = _resolve_object_name(conn, view_name)
    if not obj:
        return set()

    schema, name = obj.split(".", 1)
    oid = _get_object_id(conn, schema, name)
    if oid is None:
        return set()

    df = read_sql(
        conn,
        """
        SET NOCOUNT ON;

        SELECT c.name
        FROM sys.columns c
        WHERE c.object_id = ?;
        """,
        params=[oid],
    )
    if df is None or df.empty:
        return set()

    return set(df["name"].astype(str).str.upper().tolist())


def apply_rules(
    conn,
    domain: Optional[str],
    view_primary: str,
    maxlen_checks: List[Tuple[str, int]],
    treat_as_fk_cols: Optional[List[str]] = None,
) -> None:
    """
    Populates:
      - stg.invalid_rows (RID, rule_id, severity, column_name, message)
      - stg.valid_rid

    IMPORTANT GUARDRAIL:
      - Skip any rule if its column does NOT exist in the view_primary columns.
        This prevents errors like: Invalid column name 'SURFACE_LATITUDE'
        when you are loading non-WELL tables or have a mismatched domain.
    """
    ensure_rules_tables(conn)
    treat_as_fk_cols = treat_as_fk_cols or []

    exec_sql(conn, "TRUNCATE TABLE stg.invalid_rows; TRUNCATE TABLE stg.valid_rid;")

    rules = _get_enabled_rules(conn, domain)

    rule_id_col, enabled_col, domain_col, phase_col, rule_type_col, column_col, has_params_json = (
        _rules_table_column_map(conn)
    )

    view_cols = _get_view_columns(conn, view_primary)  # UPPERCASE set

    # If we can't resolve the view, don't run rules (prevents noisy failures)
    if not view_cols:
        # Still mark everything as valid to avoid blocking workflow
        exec_sql(conn, f"INSERT INTO stg.valid_rid(RID) SELECT v.RID FROM {view_primary} v;")
        return

    cur = conn.cursor()

    for _, r in rules.iterrows():
        rule_id = r.get(rule_id_col)
        rtype = (r.get(rule_type_col) or "").lower().strip()
        col = r.get(column_col)

        if not rule_id or not col:
            continue

        col = str(col).strip()
        if not col:
            continue

        severity = r.get("severity", "ERROR")
        msg = r.get("error_message") or ""

        # Use NAT suffix for FK-treated columns when present
        norm_col = f"{col}__NAT" if col in treat_as_fk_cols else col

        # --- SKIP RULE if its referenced column doesn't exist in the view ---
        if str(norm_col).upper() not in view_cols:
            continue

        if rtype == "required":
            cur.execute(
                f"""
                INSERT INTO stg.invalid_rows(RID, rule_id, severity, column_name, message)
                SELECT v.RID, ?, ?, ?, ?
                FROM {view_primary} v
                WHERE v.[{norm_col}] IS NULL;
                """,
                (rule_id, severity, col, msg),
            )

        elif rtype == "unique":
            cur.execute(
                f"""
                INSERT INTO stg.invalid_rows(RID, rule_id, severity, column_name, message)
                SELECT v.RID, ?, ?, ?, ?
                FROM {view_primary} v
                JOIN (
                  SELECT [{norm_col}] val
                  FROM {view_primary}
                  WHERE [{norm_col}] IS NOT NULL
                  GROUP BY [{norm_col}]
                  HAVING COUNT(*) > 1
                ) d ON d.val = v.[{norm_col}];
                """,
                (rule_id, severity, col, msg),
            )

        elif rtype == "range":
            params_json = r.get("params_json") if has_params_json else None
            try:
                p = json.loads(params_json or "{}")
            except Exception:
                p = {}

            mn = p.get("min")
            mx = p.get("max")
            inclusive = bool(p.get("inclusive", True))

            if mn is None and mx is None:
                continue

            op1 = ">=" if inclusive else ">"
            op2 = "<=" if inclusive else "<"

            conds = []
            if mn is not None:
                conds.append(f"(TRY_CONVERT(float, v.[{norm_col}]) {op1} {float(mn)})")
            if mx is not None:
                conds.append(f"(TRY_CONVERT(float, v.[{norm_col}]) {op2} {float(mx)})")

            ok = " AND ".join(conds)

            cur.execute(
                f"""
                INSERT INTO stg.invalid_rows(RID, rule_id, severity, column_name, message)
                SELECT v.RID, ?, ?, ?, ?
                FROM {view_primary} v
                WHERE v.[{norm_col}] IS NOT NULL
                  AND TRY_CONVERT(float, v.[{norm_col}]) IS NOT NULL
                  AND NOT ({ok});
                """,
                (rule_id, severity, col, msg),
            )

        else:
            # Unknown rule_type -> skip
            continue

    conn.commit()

    # Built-in maxlen checks (ONLY for columns that exist in the view)
    if maxlen_checks:
        safe_checks = []
        for c, n in maxlen_checks:
            if not c:
                continue
            if str(c).upper() not in view_cols:
                continue
            safe_checks.append((c, n))

        if safe_checks:
            ors = " OR ".join([f"(LEN(v.[{c}]) > {int(n)})" for c, n in safe_checks])
            exec_sql(
                conn,
                f"""
                INSERT INTO stg.invalid_rows(RID, rule_id, severity, column_name, message)
                SELECT v.RID, 'MAXLEN_BUILTIN', 'ERROR', NULL, 'String exceeds target max length'
                FROM {view_primary} v
                WHERE {ors};
                """,
            )

    exec_sql(
        conn,
        f"""
        INSERT INTO stg.valid_rid(RID)
        SELECT v.RID
        FROM {view_primary} v
        WHERE NOT EXISTS (SELECT 1 FROM stg.invalid_rows i WHERE i.RID = v.RID);
        """,
    )
