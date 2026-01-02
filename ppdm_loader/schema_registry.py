# ppdm_loader/schema_registry.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "schema_registry"

FILES = {
    "PPDM 3.9": REGISTRY_DIR / "ppdm_39_schema_domain.json",
    "PPDM Lite": REGISTRY_DIR / "ppdm_lite_schema_domain.json",
}

@st.cache_data(show_spinner=False)
def load_schema_catalog(model: str) -> pd.DataFrame:
    fp = FILES.get(model)
    if fp is None or not fp.exists():
        raise FileNotFoundError(f"Schema catalog not found for model={model}: {fp}")

    obj = pd.read_json(fp)

    # your sample shows the key "ppdm_39_schema_domain"
    # so find the first list-valued root key
    root_key = None
    for c in obj.columns:
        # df from read_json varies; simplest safe approach:
        pass

    # Robust: read raw JSON
    import json
    with open(fp, "r", encoding="utf-8") as f:
        raw = json.load(f)

    key = next(iter(raw.keys()))
    rows = raw[key]
    df = pd.DataFrame(rows)

    # normalize
    for col in ["category", "sub_category", "table_schema", "table_name", "column_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

def catalog_tables(df: pd.DataFrame, schema: str | None = None, domain: str | None = None) -> pd.DataFrame:
    out = df
    if schema:
        out = out[out["table_schema"].str.lower() == schema.lower()]
    if domain and domain != "(All)":
        # your JSON uses category like "ANL", "WELL", "STRAT"
        out = out[out["category"].str.upper() == domain.upper()]
    return out[["table_schema", "table_name"]].drop_duplicates().sort_values(["table_schema", "table_name"])

def catalog_columns_for_table(df: pd.DataFrame, schema: str, table: str) -> pd.DataFrame:
    out = df[
        (df["table_schema"].str.lower() == schema.lower()) &
        (df["table_name"].str.lower() == table.lower())
    ].copy()
    return out.sort_values(["is_primary_key", "is_foreign_key", "column_name"], ascending=[False, False, True])
