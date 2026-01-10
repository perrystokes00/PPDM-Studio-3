# build_catalog.py
r"""
Build a PPDM seed catalog JSON by scanning individual seed JSON files.

What it does
- Scans: <seed-pack-root>/seeds/**/*.json
- For each seed file, reads top-level "name" (expects "schema.table")
- Introspects SQL Server for:
    - table existence
    - PK columns (in ordinal order)
- Writes catalog JSON (with structured metadata useful for batch seeding)

Typical usage (Windows auth):
  python build_catalog.py ^
    --server localhost ^
    --database PPDM39_DEMO ^
    --auth windows ^
    --seed-pack-root "C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\ppdm39_extended_seed_pack" ^
    --out "C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\ppdm39_extended_seed_pack\catalog\ppdm39_seed_catalog.json" ^
    --ppdm-version "3.9"

SQL login:
  python build_catalog.py --server localhost --database PPDM_LITE_DEMO_1 --auth sql --user sa --password **** ^
    --seed-pack-root "C:\...\ppdm_lite_seed_pack" --out "C:\...\ppdm_lite_seed_pack\catalog\ppdm_lite_seed_catalog.json" --ppdm-version "lite"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyodbc


# -----------------------------
# DB helpers
# -----------------------------
def connect_sqlserver(
    *,
    server: str,
    database: str,
    auth: str = "windows",  # windows | sql
    user: str | None = None,
    password: str | None = None,
    driver: str = "ODBC Driver 18 for SQL Server",
    trust_server_certificate: bool = True,
    encrypt: bool = False,
    timeout: int = 30,
) -> pyodbc.Connection:
    auth = (auth or "windows").strip().lower()
    parts: List[str] = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Encrypt={'yes' if encrypt else 'no'}",
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'}",
        f"Connection Timeout={timeout}",
    ]
    if auth == "windows":
        parts.append("Trusted_Connection=yes")
    elif auth == "sql":
        if not user or not password:
            raise ValueError("SQL auth requires --user and --password")
        parts.append(f"UID={user}")
        parts.append(f"PWD={password}")
    else:
        raise ValueError("auth must be 'windows' or 'sql'")

    conn_str = ";".join(parts) + ";"
    return pyodbc.connect(conn_str)


def db_table_exists(conn: pyodbc.Connection, schema: str, table: str) -> bool:
    sql = """
    SELECT 1
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?;
    """
    cur = conn.cursor()
    cur.execute(sql, (schema, table))
    return cur.fetchone() is not None


def db_fetch_pk_columns(conn: pyodbc.Connection, schema: str, table: str) -> List[str]:
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


# -----------------------------
# Seed JSON parsing
# -----------------------------
def load_seed_json(seed_path: Path) -> Dict[str, Any]:
    with seed_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Seed JSON must be a JSON object")
    return data


def extract_table_name(seed_obj: Dict[str, Any]) -> str:
    # expected: {"name": "dbo.r_source", "rows":[...]}
    name = seed_obj.get("name") or seed_obj.get("table") or ""
    name = str(name).strip()
    return name


def extract_rows_count(seed_obj: Dict[str, Any]) -> int:
    rows = seed_obj.get("rows", [])
    return len(rows) if isinstance(rows, list) else 0


# -----------------------------
# Catalog model
# -----------------------------
@dataclass
class CatalogSeedEntry:
    table: str
    file: str
    format: str = "json"
    keys: List[str] | None = None
    mode: str = "missing_only"
    on_missing_table: str = "skip"  # skip | error
    on_pk_missing: str = "error"    # error | skip
    # diagnostics (helps auditing)
    table_found: bool = False
    pk_found: bool = False
    rows_count: int = 0
    message: str = ""


def build_catalog(
    *,
    conn: pyodbc.Connection,
    seed_pack_root: Path,
    ppdm_version: str,
    schema_default: str = "dbo",
    include_nonexistent_tables: bool = True,
) -> Dict[str, Any]:
    seeds_dir = seed_pack_root / "seeds"
    if not seeds_dir.exists():
        raise FileNotFoundError(f"Seeds folder not found: {seeds_dir}")

    seed_files = sorted(seeds_dir.rglob("*.json"))
    entries: List[CatalogSeedEntry] = []

    for p in seed_files:
        rel = p.relative_to(seed_pack_root).as_posix()
        try:
            seed_obj = load_seed_json(p)
            table_fq = extract_table_name(seed_obj)
            rows_count = extract_rows_count(seed_obj)

            if not table_fq or "." not in table_fq:
                entries.append(
                    CatalogSeedEntry(
                        table=table_fq or "(missing)",
                        file=rel,
                        keys=None,
                        table_found=False,
                        pk_found=False,
                        rows_count=rows_count,
                        message="Seed JSON missing valid 'name' as schema.table",
                    )
                )
                continue

            schema, table = table_fq.split(".", 1)
            schema = (schema or schema_default).strip()
            table = table.strip()
            table_fq_norm = f"{schema}.{table}"

            found = db_table_exists(conn, schema, table)
            if not found and not include_nonexistent_tables:
                # exclude entirely
                continue

            pk_cols: List[str] = []
            pk_found = False
            msg = ""
            if found:
                pk_cols = db_fetch_pk_columns(conn, schema, table)
                pk_found = bool(pk_cols)
                if not pk_found:
                    msg = "Target table has no PK (or PK not found)."
            else:
                msg = "Table missing in target DB."

            entries.append(
                CatalogSeedEntry(
                    table=table_fq_norm,
                    file=rel,
                    keys=pk_cols if pk_cols else None,
                    table_found=found,
                    pk_found=pk_found,
                    rows_count=rows_count,
                    message=msg,
                )
            )

        except Exception as e:
            entries.append(
                CatalogSeedEntry(
                    table="(error)",
                    file=rel,
                    keys=None,
                    table_found=False,
                    pk_found=False,
                    rows_count=0,
                    message=f"Failed to parse seed JSON: {e}",
                )
            )

    # Sort for stable diffs: table then file
    entries.sort(key=lambda x: (x.table.lower(), x.file.lower()))

    catalog: Dict[str, Any] = {
        "ppdm_version": ppdm_version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed_pack_root": str(seed_pack_root),
        "defaults": {
            "schema": schema_default,
            "mode": "missing_only",
            "format": "json",
            "on_missing_table": "skip",
            "on_pk_missing": "error",
            "ppdm_guid": {"strategy": "newid_if_missing"},
            "audit": {"row_created_by": "loaded_by", "row_changed_by": "loaded_by"},
        },
        "seeds": [asdict(e) for e in entries],
        "summary": {
            "seed_files_found": len(seed_files),
            "catalog_entries": len(entries),
            "tables_found": sum(1 for e in entries if e.table_found),
            "tables_missing": sum(1 for e in entries if not e.table_found),
            "pk_missing": sum(1 for e in entries if e.table_found and not e.pk_found),
            "parse_errors": sum(1 for e in entries if e.table == "(error)"),
        },
    }
    return catalog


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a seed catalog from individual JSON seed files.")
    ap.add_argument("--server", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--auth", choices=["windows", "sql"], default="windows")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--driver", default="ODBC Driver 18 for SQL Server")

    ap.add_argument("--seed-pack-root", required=True, help="Folder that contains /seeds and /catalog")
    ap.add_argument("--out", required=True, help="Output catalog JSON file path")

    ap.add_argument("--ppdm-version", required=True, help='e.g. "3.9" or "lite"')
    ap.add_argument("--schema-default", default="dbo")

    ap.add_argument(
        "--exclude-missing-tables",
        action="store_true",
        help="If set, seed files whose table is missing in the target DB are excluded from the catalog.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    seed_pack_root = Path(args.seed_pack_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_sqlserver(
        server=args.server,
        database=args.database,
        auth=args.auth,
        user=args.user,
        password=args.password,
        driver=args.driver,
    )

    try:
        catalog = build_catalog(
            conn=conn,
            seed_pack_root=seed_pack_root,
            ppdm_version=args.ppdm_version,
            schema_default=args.schema_default,
            include_nonexistent_tables=not args.exclude_missing_tables,
        )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

        print(f"Wrote catalog: {out_path}")
        print("Summary:", json.dumps(catalog.get("summary", {}), indent=2))
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
