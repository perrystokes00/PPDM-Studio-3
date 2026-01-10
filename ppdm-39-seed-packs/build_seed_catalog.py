from __future__ import annotations

import json
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
ROOT = Path(r"C:\Users\perry\OneDrive\Documents\PPDM\ppdm39-seed-catalog")
SEEDS_DIR = ROOT / "seeds" 
OUT = ROOT / "catalog" / "ppdm39_extended_seed_catalog.generated.json"

# Default key hints by table (override as needed)
# If a table isn't listed here, you'll need to fill keys later.
KEY_HINTS = {
    "r_dir_srvy_type": ["DIR_SRVY_TYPE"],
    "r_well_status": ["WELL_STATUS", "WELL_STATUS_TYPE"],
    "ppdm_unit_of_measure": ["UOM_ID"],
    "business_associate": ["BUSINESS_ASSOCIATE_ID"],
    "strat_unit": ["STRAT_UNIT_ID"],
    "well_log": ["UWI", "LOG_ID", "SOURCE"],
    "seis_set": ["SEIS_SET_ID"],
    "pden_well": ["UWI", "SOURCE"],
}

GROUP_BY_FOLDER = {
    "r": "r",
    "well": "well",
    "strat": "strat",
    "logs": "logs",
    "seismic": "seismic",
    "production": "production",
    "areas": "areas",
}

def infer_group(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    for folder, group in GROUP_BY_FOLDER.items():
        if folder in parts:
            return group
    return "misc"

def main() -> None:
    seeds = []
    for p in sorted(SEEDS_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".csv"}:
            continue

        table = p.stem  # file name without extension, e.g. r_well_status
        group = infer_group(p)

        # IMPORTANT: catalog file path is relative to ROOT
        rel = p.relative_to(ROOT).as_posix().replace("/", "\\")
        fmt = "json" if p.suffix.lower() == ".json" else "csv"

        keys = KEY_HINTS.get(table, [])
        seeds.append(
            {
                "group": group,
                "schema": "dbo",
                "table": table,
                "file": rel,
                "format": fmt,
                "keys": keys,
                "upsert_mode": "merge",
                "fk_mode": "skip_and_log",
            }
        )

    catalog = {
        "catalog_version": "1.0",
        "ppdm_version": "3.9",
        "root": str(ROOT).replace("\\", "\\\\"),
        "log_dir": "logs\\seed_runs",
        "defaults": {
            "schema": "dbo",
            "upsert_mode": "merge",
            "fk_mode": "skip_and_log",
            "table_missing_mode": "skip_and_log",
            "pk_missing_mode": "skip_and_log",
            "ppdm_guid": {"strategy": "newid_if_missing", "column": "PPDM_GUID"},
            "audit_user": "Seeder",
            "audit_defaults": {
                "ROW_CREATED_BY": "{audit_user}",
                "ROW_CHANGED_BY": "{audit_user}",
                "ROW_CREATED_DATE": "sysutcdatetime",
                "ROW_CHANGED_DATE": "sysutcdatetime",
                "ROW_EFFECTIVE_DATE": "sysutcdatetime",
            },
            "string_cleanup": {"trim": True, "strip_wrapping_quotes": True, "empty_to_null": True},
            "csv": {"delimiter": ",", "encoding": "utf-8", "header": True},
            "json": {"rows_path": "rows", "accept_shapes": ["root.rows", "rows", "list"]},
        },
        "seeds": seeds,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {len(seeds)} seed entries.")
    print("NOTE: any entries with empty 'keys' must be filled in before seeding is meaningful.")

if __name__ == "__main__":
    main()
