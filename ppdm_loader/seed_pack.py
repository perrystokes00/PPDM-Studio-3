# ppdm_loader/seed_pack.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ppdm_loader.seed_generic import fetch_pk_columns, seed_missing_rows


@dataclass(frozen=True)
class SeedStep:
    table_fqn: str
    seed_file: str


@dataclass(frozen=True)
class SeedPack:
    name: str
    version: str
    model: str
    steps: list[SeedStep]


def load_pack(pack_path: str | Path) -> SeedPack:
    p = Path(pack_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    steps = []
    for s in obj.get("steps", []):
        steps.append(SeedStep(table_fqn=str(s["table"]), seed_file=str(s["seed_file"])))
    return SeedPack(
        name=str(obj.get("name", p.stem)),
        version=str(obj.get("version", "1.0")),
        model=str(obj.get("model", "")),
        steps=steps,
    )


def load_seed_df(seed_path: str | Path) -> pd.DataFrame:
    p = Path(seed_path)
    obj: Any = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        rows = obj.get("rows") or obj.get("data") or obj.get("items") or []
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    return pd.DataFrame(rows)


def run_seed_pack(
    conn,
    *,
    pack_path: str | Path,
    base_dir: str | Path | None = None,
    loaded_by: str = "Perry M Stokes",
) -> dict[str, int]:
    """
    Returns dict: { "dbo.table": inserted_count }
    """
    pack = load_pack(pack_path)
    base = Path(base_dir) if base_dir else Path(pack_path).parent

    results: dict[str, int] = {}

    for step in pack.steps:
        if "." not in step.table_fqn:
            raise ValueError(f"Invalid table fqn: {step.table_fqn}")
        schema, table = step.table_fqn.split(".", 1)

        seed_fp = (base / step.seed_file).resolve()
        df = load_seed_df(seed_fp)

        pk_cols = fetch_pk_columns(conn, schema=schema, table=table)
        if not pk_cols:
            raise ValueError(f"PK not detected for {step.table_fqn} (required).")

        inserted = seed_missing_rows(
            conn,
            target_schema=schema,
            target_table=table,
            pk_cols=pk_cols,
            insert_df=df,
            loaded_by=loaded_by,
        )
        results[step.table_fqn] = inserted

    return results
