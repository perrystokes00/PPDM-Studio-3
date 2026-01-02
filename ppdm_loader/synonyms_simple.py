# ppdm_loader/synonyms_simple.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


# Default location: project_root/ppdm_loader/synonyms_simple.json
DEFAULT_SYNONYMS_PATH = Path(__file__).with_name("synonyms_simple.json")


def _norm(s: str) -> str:
    """Normalize a column name for matching."""
    return str(s or "").strip().upper()


def load_synonyms(path: Optional[str | Path] = None) -> Dict[str, str]:
    """
    Loads synonyms from synonyms_simple.json and returns a flat mapping:
      variant_name -> canonical_name

    Supports either of these JSON shapes:

    Shape A (recommended):
      {
        "STATE_FIPS": ["STATEFIPS", "STATE_FIPS_CODE"],
        "COUNTY_FIPS": ["COUNTYFIPS", "COUNTY_FIPS_CODE"]
      }

    Shape B (nested groups):
      {
        "area": {
          "STATE_FIPS": ["STATEFIPS"],
          "COUNTY_FIPS": ["COUNTYFIPS"]
        }
      }
    """
    p = Path(path) if path else DEFAULT_SYNONYMS_PATH
    if not p.exists():
        return {}

    raw = json.loads(p.read_text(encoding="utf-8"))

    flat: Dict[str, str] = {}

    def add_group(obj: dict):
        for canonical, variants in obj.items():
            if isinstance(variants, str):
                variants = [variants]
            if not isinstance(variants, list):
                continue
            canon_u = _norm(canonical)
            for v in variants:
                v_u = _norm(v)
                if v_u and canon_u:
                    flat[v_u] = canon_u

    # detect shape
    if isinstance(raw, dict):
        # Shape A: canonical -> [variants]
        # if any value is list, treat as shape A
        if any(isinstance(v, list) or isinstance(v, str) for v in raw.values()):
            # But could be nested dict, so check:
            if any(isinstance(v, dict) for v in raw.values()):
                # Shape B: top-level sections -> dicts
                for section in raw.values():
                    if isinstance(section, dict):
                        add_group(section)
            else:
                add_group(raw)
        else:
            # unknown
            pass

    # also map canonical to itself (handy)
    for canon in set(flat.values()):
        flat[canon] = canon

    return flat


def apply_synonyms(
    pm,
    source_cols=None,
    primary_schema=None,
    primary_table=None,
    synonyms_path=None,
):
    """
    Normalize column names using synonym mappings.
    Compatible with existing app calls.
    """
    import json
    from pathlib import Path

    def norm(x):
        return str(x or "").strip().upper()

    # Load synonyms file if present
    if synonyms_path is None:
        synonyms_path = Path(__file__).with_name("synonyms_simple.json")

    if synonyms_path.exists():
        with open(synonyms_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}

    # Flatten synonyms
    synonym_map = {}
    for canon, values in raw.items():
        if isinstance(values, list):
            for v in values:
                synonym_map[v.upper()] = canon.upper()

    # Apply to DataFrame
    if "source_column" not in pm.columns:
        return pm

    def map_col(col):
        if not col:
            return col
        key = col.strip().upper()
        return synonym_map.get(key, col)

    pm["source_column"] = pm["source_column"].apply(map_col)
    return pm



def save_mappings_as_synonyms(
    mapping_df: pd.DataFrame,
    out_path: str | Path = DEFAULT_SYNONYMS_PATH,
    *,
    target_col: str = "column_name",
    source_col: str = "source_column",
    merge_with_existing: bool = True,
) -> Path:
    """
    Takes your mapping grid (from Streamlit data_editor) and writes a synonyms JSON.

    Expects mapping_df columns:
      - target_col (default 'column_name') = canonical/target name (PPDM column)
      - source_col (default 'source_column') = source header name from CSV

    Output shape:
      {
        "CANONICAL": ["Variant1", "Variant2"]
      }
    """
    out_path = Path(out_path)

    # Build canonical -> set(variants)
    canon_to_vars: Dict[str, set[str]] = {}

    for _, r in mapping_df.iterrows():
        canon = str(r.get(target_col) or "").strip()
        var = str(r.get(source_col) or "").strip()
        if not canon or not var:
            continue
        canon_u = _norm(canon)
        var_u = _norm(var)
        canon_to_vars.setdefault(canon_u, set()).add(var_u)

    # Merge existing if requested
    if merge_with_existing and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

        # normalize existing into canon_to_vars
        if isinstance(existing, dict):
            for canon, variants in existing.items():
                canon_u = _norm(canon)
                if isinstance(variants, str):
                    variants = [variants]
                if isinstance(variants, list):
                    for v in variants:
                        canon_to_vars.setdefault(canon_u, set()).add(_norm(v))

    # Convert sets to sorted lists
    payload = {canon: sorted(list(vars_set)) for canon, vars_set in sorted(canon_to_vars.items())}

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
