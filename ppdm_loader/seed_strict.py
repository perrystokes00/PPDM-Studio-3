# ppdm_loader/seed_strict.py
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# -----------------------------
# Seed file loader
# -----------------------------

def load_seed_rows_from_json(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      { "name": "dbo.table", "rows": [ {...}, {...} ] }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Seed JSON 'rows' must be a list")
    # Ensure dict rows
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Each seed row must be an object/dict")
        out.append(dict(r))
    return out

# -----------------------------
# DB helpers
# -----------------------------

def fetch_table_columns(conn, schema: str, table: str) -> List[str]:
    sql = """
    SELECT c.name
    FROM sys.columns c
    JOIN sys.tables t ON t.object_id = c.object_id
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id;
    """
    cur = conn.cursor()
    cur.execute(sql, (schema, table))
    rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []

def fetch_existing_pk_set(conn, schema: str, table: str, keys: List[str]) -> set[Tuple[Any, ...]]:
    cols = ", ".join(f"[{k}]" for k in keys)
    sql = f"SELECT {cols} FROM [{schema}].[{table}];"
    cur = conn.cursor()
    cur.execute(sql)
    existing = set()
    for row in cur.fetchall():
        existing.add(tuple(row))
    return existing

def _col_present(cols: List[str], name: str) -> bool:
    return any(c.lower() == name.lower() for c in cols)

def _actual_col_name(cols: List[str], name: str) -> str:
    # return actual case from DB columns list
    for c in cols:
        if c.lower() == name.lower():
            return c
    return name

# -----------------------------
# Strict missing-only seeder
# -----------------------------

def seed_missing_rows_strict(
    *,
    conn,
    table_fq: str,
    keys: List[str],
    rows: List[Dict[str, Any]],
    created_by: str,
) -> int:
    """
    Strict:
      - No mapping
      - Columns in seed rows must exist in DB (non-existent columns are rejected)
      - Inserts only missing PK tuples
      - Auto-populates PPDM_GUID and audit columns if present
    """
    if "." not in table_fq:
        raise ValueError("table_fq must be like 'dbo.r_source'")
    schema, table = table_fq.split(".", 1)

    db_cols = fetch_table_columns(conn, schema, table)
    if not db_cols:
        raise ValueError(f"Target table not found or has no columns: {table_fq}")

    # Validate keys exist in DB
    missing_in_db = [k for k in keys if not _col_present(db_cols, k)]
    if missing_in_db:
        raise ValueError(f"Catalog keys not found in DB columns: {missing_in_db}")

    existing = fetch_existing_pk_set(conn, schema, table, keys)

    # Determine standard PPDM columns
    has_guid = _col_present(db_cols, "PPDM_GUID")
    has_row_created_by = _col_present(db_cols, "ROW_CREATED_BY")
    has_row_changed_by = _col_present(db_cols, "ROW_CHANGED_BY")
    has_row_created_date = _col_present(db_cols, "ROW_CREATED_DATE")
    has_row_changed_date = _col_present(db_cols, "ROW_CHANGED_DATE")

    # Build insert list:
    # - include only columns that exist in DB
    # - reject any seed columns not in DB
    seed_cols_union = set()
    for r in rows:
        seed_cols_union.update(r.keys())

    unknown_cols = [c for c in seed_cols_union if not _col_present(db_cols, c)]
    if unknown_cols:
        raise ValueError(f"Seed contains column(s) not in target table {table_fq}: {unknown_cols}")

    # Use a stable column order: keys first, then other seed cols, then auto cols
    ordered_seed_cols = []
    # keys in order
    for k in keys:
        ordered_seed_cols.append(_actual_col_name(db_cols, k))
    # other cols in file order-ish (sorted for determinism)
    for c in sorted(seed_cols_union, key=lambda x: x.lower()):
        actual = _actual_col_name(db_cols, c)
        if actual not in ordered_seed_cols:
            ordered_seed_cols.append(actual)

    auto_cols = []
    if has_guid and "PPDM_GUID" not in [c.upper() for c in ordered_seed_cols]:
        auto_cols.append(_actual_col_name(db_cols, "PPDM_GUID"))
    if has_row_created_by and "ROW_CREATED_BY" not in [c.upper() for c in ordered_seed_cols]:
        auto_cols.append(_actual_col_name(db_cols, "ROW_CREATED_BY"))
    if has_row_changed_by and "ROW_CHANGED_BY" not in [c.upper() for c in ordered_seed_cols]:
        auto_cols.append(_actual_col_name(db_cols, "ROW_CHANGED_BY"))
    if has_row_created_date and "ROW_CREATED_DATE" not in [c.upper() for c in ordered_seed_cols]:
        auto_cols.append(_actual_col_name(db_cols, "ROW_CREATED_DATE"))
    if has_row_changed_date and "ROW_CHANGED_DATE" not in [c.upper() for c in ordered_seed_cols]:
        auto_cols.append(_actual_col_name(db_cols, "ROW_CHANGED_DATE"))

    insert_cols = ordered_seed_cols + [c for c in auto_cols if c not in ordered_seed_cols]

    # Prepare parameter rows for missing PKs only
    to_insert = []
    for r in rows:
        pk = tuple(r.get(k) for k in keys)
        if pk in existing:
            continue

        row_out: Dict[str, Any] = {}

        # Copy seed values as-is
        for c in ordered_seed_cols:
            # map to seed key case-insensitively
            # find the matching key in r
            val = None
            found = False
            for rk, rv in r.items():
                if rk.lower() == c.lower():
                    val = rv
                    found = True
                    break
            if found:
                row_out[c] = val
            else:
                # column not in this row (allowed if nullable)
                row_out[c] = None

        # Auto-populate
        if has_guid:
            # If seed provided it (case-insensitive), keep it; else generate
            provided = any(k.lower() == "ppdm_guid" for k in r.keys())
            if not provided:
                row_out[_actual_col_name(db_cols, "PPDM_GUID")] = str(uuid.uuid4()).upper()

        if has_row_created_by:
            provided = any(k.lower() == "row_created_by" for k in r.keys())
            if not provided:
                row_out[_actual_col_name(db_cols, "ROW_CREATED_BY")] = created_by

        if has_row_changed_by:
            provided = any(k.lower() == "row_changed_by" for k in r.keys())
            if not provided:
                row_out[_actual_col_name(db_cols, "ROW_CHANGED_BY")] = created_by

        # Dates: let SQL default if you prefer; here we set GETDATE() via SQL expression is harder w/ params.
        # We'll set them in SQL as GETDATE() only if columns exist and not provided.
        # (Implemented by leaving them NULL and using COALESCE defaults requires table defaults.)
        # So we only set if provided; otherwise leave None.
        # If your tables lack defaults and you want this forced, tell me and I’ll switch to GETDATE() in SQL.

        to_insert.append(row_out)
        existing.add(pk)

    if not to_insert:
        return 0

    # Build INSERT
    cols_sql = ", ".join(f"[{c}]" for c in insert_cols)
    params_sql = ", ".join("?" for _ in insert_cols)
    sql = f"INSERT INTO [{schema}].[{table}] ({cols_sql}) VALUES ({params_sql});"

    values = []
    for ro in to_insert:
        values.append([ro.get(c) for c in insert_cols])

    cur = conn.cursor()
    cur.fast_executemany = True
    cur.executemany(sql, values)
    conn.commit()
    return len(to_insert)
