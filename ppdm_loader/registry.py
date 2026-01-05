# ppdm_loader/registry.py
import json
import pandas as pd
from pathlib import Path
import streamlit as st

@st.cache_data(show_spinner=False)
def load_schema_registry(json_path: str) -> pd.DataFrame:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for r in data:
        rows.append({
            "schema": r.get("table_schema", "dbo"),
            "table": r["table_name"],
            "column": r["column_name"],
            "is_pk": r.get("is_primary_key") in ("YES", "Y", True, 1),
        })

    df = pd.DataFrame(rows)
    return df
