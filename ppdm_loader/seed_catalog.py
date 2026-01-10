# ppdm_loader/seed_catalog.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Catalog dataclasses
# -----------------------------

@dataclass
class SeedEntry:
    table: str                 # e.g. "dbo.r_well_status"
    file: str                  # relative path inside seed pack, e.g. "seeds/r_well_status.json"
    keys: List[str]            # PK columns in order
    format: str = "json"       # "json" or "csv"
    mode: str = "missing_only" # "missing_only" (default) or "merge" (future)
    on_missing_table: str = "skip"
    on_pk_missing: str = "error"


@dataclass
class SeedCatalog:
    ppdm_version: str          # "3.9" or "lite"
    defaults: Dict[str, Any]
    seeds: List[SeedEntry]


# -----------------------------
# Helpers
# -----------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _seed_file_table_name(seed_json: Dict[str, Any]) -> Optional[str]:
    # Supported shape:
    # { "name": "dbo.r_well_status", "rows": [...] }
    name = seed_json.get("name") or seed_json.get("table")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None

def _iter_seed_json_files(seed_pack_root: Path) -> List[Path]:
    seeds_dir = seed_pack_root / "seeds"
    if not seeds_dir.exists():
        return []
    return sorted(seeds_dir.rglob("*.json"))

# -----------------------------
# DB introspection (PK columns)
# You can replace this with your existing introspect module if you prefer.
# -----------------------------

def fetch_pk_columns(conn, schema: str, table: str) -> List[str]:
    """
    Returns PK columns in ordinal order. Empty list if no PK or table missing.
    """
    sql = """
    SELECT
        c.name AS column_name,
        ic.key_ordinal
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

def table_exists(conn, schema: str, table: str) -> bool:
    sql = """
    SELECT 1
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    WHERE s.name = ? AND t.name = ?;
    """
    cur = conn.cursor()
    cur.execute(sql, (schema, table))
    return cur.fetchone() is not None

# -----------------------------
# Catalog build / load
# -----------------------------

def build_catalog_from_seed_pack(
    *,
    conn,
    seed_pack_root: Path,
    ppdm_version: str,
    schema_default: str = "dbo",
    created_by: str = "Perry M Stokes",
) -> SeedCatalog:
    """
    Builds a catalog with as many seeds as possible by scanning seeds/**/*.json.
    For each seed file:
      - reads "name" -> target table
      - introspects PK keys from DB
    """
    seed_files = _iter_seed_json_files(seed_pack_root)
    entries: List[SeedEntry] = []

    for f in seed_files:
        try:
            seed_json = _load_json(f)
        except Exception:
            # Skip unreadable JSON; catalog should not crash here
            continue

        fq = _seed_file_table_name(seed_json)
        if not fq or "." not in fq:
            continue

        schema, table = fq.split(".", 1)
        schema = schema.strip() or schema_default
        table = table.strip()

        # Discover PK keys from DB
        keys = fetch_pk_columns(conn, schema, table)

        rel = str(f.relative_to(seed_pack_root)).replace("\\", "/")
        entries.append(
            SeedEntry(
                table=f"{schema}.{table}",
                file=rel,
                keys=keys,
                format="json",
                mode="missing_only",
                on_missing_table="skip",
                on_pk_missing="error",
            )
        )

    catalog = SeedCatalog(
        ppdm_version=ppdm_version,
        defaults={
            "schema": schema_default,
            "mode": "missing_only",
            "ppdm_guid": {"strategy": "new_guid_if_missing"},
            "audit": {
                "row_created_by": created_by,
                "row_changed_by": created_by,
            },
        },
        seeds=entries,
    )
    return catalog

def save_catalog(catalog: SeedCatalog, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ppdm_version": catalog.ppdm_version,
        "defaults": catalog.defaults,
        "seeds": [asdict(s) for s in catalog.seeds],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_catalog(path: Path) -> SeedCatalog:
    data = _load_json(path)
    seeds = [SeedEntry(**s) for s in data.get("seeds", [])]
    return SeedCatalog(
        ppdm_version=str(data.get("ppdm_version", "")).strip(),
        defaults=dict(data.get("defaults", {})),
        seeds=seeds,
    )
