# ppdm_loader/seed_batch.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import json
import pandas as pd

from ppdm_loader.seed_core import seed_missing_rows
import uuid


# -----------------------------
# Structured log row
# -----------------------------
@dataclass
class SeedLogRow:
    ts: str
    table: str
    file: str
    status: str          # OK / SKIP / ERROR
    inserted: int
    elapsed_ms: int
    message: str


# -----------------------------
# Catalog loading
# -----------------------------
def load_seed_catalog(catalog_path: Path) -> Dict[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# DB helpers
# -----------------------------
def _db_table_exists(conn, schema: str, table: str) -> bool:
    sql = """
    SELECT 1
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?;
    """
    cur = conn.cursor()
    cur.execute(sql, (schema, table))
    return cur.fetchone() is not None


def _db_fetch_pk_columns(conn, schema: str, table: str) -> List[str]:
    sql = """
    SELECT c.name, ic.key_ordinal
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    JOIN sys.indexes i ON i.object_id = t.object_id AND i.is_primary_key = 1
    JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id
    JOIN sys.columns c ON c.object_id = t.object_id AND c.column_id = ic.column_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY ic.key_ordinal;
    """
    cur = conn.cursor()
    cur.execute(sql, (schema, table))
    rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []


def _db_fetch_table_columns(conn, schema: str, table: str) -> List[str]:
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
    return [r[0] for r in cur.fetchall()]


# -----------------------------
# Seed file loading
# -----------------------------
def _load_seed_rows(seed_file: Path) -> List[Dict[str, Any]]:
    with seed_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Seed JSON must contain a list under 'rows'")
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Seed JSON rows must be objects")
        out.append(dict(r))
    return out


def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# -----------------------------
# Core seeding (idempotent, missing-only)
# -----------------------------
def _seed_one(
    *,
    conn,
    target_schema: str,
    target_table: str,
    rows: List[Dict[str, Any]],
    loaded_by: str,
    strict_columns: bool = False,  # batch default False (drops unknown cols)
) -> Tuple[int, str]:
    # DB truth
    pk_cols = _db_fetch_pk_columns(conn, target_schema, target_table)
    pk_cols = [c.strip().upper() for c in pk_cols]
    if not pk_cols:
        return (0, "ERROR: Target table has no PK (or PK not found).")

    df = _rows_to_df(rows)
    if df.empty:
        return (0, "ERROR: No rows found in seed JSON.")

    # ---- Case-insensitive PK validation + rename DF to DB PK casing ----
    df.columns = [str(c) for c in df.columns]
    df_cols_upper = {str(c).upper(): str(c) for c in df.columns}
    pk_upper = [str(c).upper() for c in pk_cols]

    missing_pk_upper = [c for c in pk_upper if c not in df_cols_upper]
    if missing_pk_upper:
        missing_db = [pk_cols[pk_upper.index(u)] for u in missing_pk_upper]
        return (0, f"ERROR: Seed missing PK column(s): {missing_db}")

    rename_map = {}
    for pk in pk_cols:
        u = str(pk).upper()
        rename_map[df_cols_upper[u]] = pk
    df = df.rename(columns=rename_map)

    # ---- Column validation vs DB ----
    db_cols = _db_fetch_table_columns(conn, target_schema, target_table)
    db_cols_u = {c.strip().upper(): c.strip() for c in db_cols}

    df_u = {c.upper(): c for c in df.columns}
    extra = [df_u[u] for u in df_u.keys() if u not in db_cols_u]
    if extra:
        if strict_columns:
            return (0, f"ERROR: Seed contains column(s) not in DB table {target_schema}.{target_table}: {sorted(extra)}")
        # batch mode: drop them
        df = df[[c for c in df.columns if c.upper() in db_cols_u]]

    # ---- Auto-populate PPDM_GUID + audit columns if present in DB ----
    if "PPDM_GUID" in db_cols_u:
        guid_col = db_cols_u["PPDM_GUID"]
        if guid_col not in df.columns:
            df[guid_col] = [str(uuid.uuid4()).upper() for _ in range(len(df))]
        else:
            s = df[guid_col].astype("string")
            mask = s.isna() | (s.str.strip() == "")
            if mask.any():
                df.loc[mask, guid_col] = [str(uuid.uuid4()).upper() for _ in range(int(mask.sum()))]

    if "ROW_CREATED_BY" in db_cols_u and db_cols_u["ROW_CREATED_BY"] not in df.columns:
        df[db_cols_u["ROW_CREATED_BY"]] = loaded_by
    if "ROW_CHANGED_BY" in db_cols_u and db_cols_u["ROW_CHANGED_BY"] not in df.columns:
        df[db_cols_u["ROW_CHANGED_BY"]] = loaded_by

    inserted = seed_missing_rows(
        conn,
        target_schema=target_schema,
        target_table=target_table,
        pk_cols=pk_cols,
        insert_df=df,
        loaded_by=loaded_by,
    )
    return (int(inserted), "OK")


# -----------------------------
# Disk logging
# -----------------------------
def write_seed_logs_to_disk(
    *,
    df: pd.DataFrame,
    log_dir: Path,
    run_id: str,
) -> Dict[str, str]:
    """
    Writes:
      - seed_run_<run_id>.csv
      - seed_run_<run_id>.jsonl
    Returns paths as strings.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / f"seed_run_{run_id}.csv"
    jsonl_path = log_dir / f"seed_run_{run_id}.jsonl"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    # JSON Lines: one object per line
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
    }


# -----------------------------
# Batch runner (Seed from Catalog)
# -----------------------------
def seed_from_catalog(
    *,
    conn,
    seed_pack_root: Path,
    catalog_path: Path,
    loaded_by: str,
    log_dir: Path | None = None,
) -> Tuple[List[SeedLogRow], pd.DataFrame, Dict[str, str]]:
    """
    Runs catalog deterministically with structured logging.
    Also writes logs to disk (CSV + JSONL).

    Returns: (logs, df, file_paths)
    """
    cat = load_seed_catalog(catalog_path)

    seeds = cat.get("seeds", [])
    if not isinstance(seeds, list):
        raise ValueError("Catalog 'seeds' must be a list")

    logs: List[SeedLogRow] = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for entry in seeds:
        ts = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()

        table_fq = str(entry.get("table", "")).strip()
        file_rel = str(entry.get("file", "")).strip()

        if not table_fq or "." not in table_fq:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logs.append(SeedLogRow(ts, table_fq or "(missing)", file_rel, "ERROR", 0, elapsed_ms, "Catalog entry missing 'table'"))
            continue

        schema, table = table_fq.split(".", 1)

        if not _db_table_exists(conn, schema, table):
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logs.append(SeedLogRow(ts, table_fq, file_rel, "SKIP", 0, elapsed_ms, "Table missing in target DB"))
            continue

        seed_file = (seed_pack_root / file_rel).resolve()
        if not seed_file.exists():
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logs.append(SeedLogRow(ts, table_fq, file_rel, "ERROR", 0, elapsed_ms, "Seed file not found"))
            continue

        try:
            rows = _load_seed_rows(seed_file)
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logs.append(SeedLogRow(ts, table_fq, file_rel, "ERROR", 0, elapsed_ms, f"Failed to parse seed JSON: {e}"))
            continue

        try:
            inserted, msg = _seed_one(
                conn=conn,
                target_schema=schema,
                target_table=table,
                rows=rows,
                loaded_by=loaded_by,
                strict_columns=False,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            status = "OK" if msg == "OK" else ("ERROR" if msg.startswith("ERROR") else "OK")
            logs.append(SeedLogRow(ts, table_fq, file_rel, status, int(inserted), elapsed_ms, msg))
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logs.append(SeedLogRow(ts, table_fq, file_rel, "ERROR", 0, elapsed_ms, str(e)))

    df = pd.DataFrame([asdict(r) for r in logs]) if logs else pd.DataFrame(
        columns=["ts", "table", "file", "status", "inserted", "elapsed_ms", "message"]
    )

    # Write to disk
    if log_dir is None:
        log_dir = seed_pack_root / "logs"

    file_paths = write_seed_logs_to_disk(df=df, log_dir=log_dir, run_id=run_id)

    return logs, df, file_paths
