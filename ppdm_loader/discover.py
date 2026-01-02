# ppdm_loader/discover.py
from __future__ import annotations
import pandas as pd
from typing import List, Optional

from .schema_registry import load_schema_catalog, catalog_tables
from .introspect import fetch_all_columns  # fallback

def discover_top_tables(
    conn,
    source_cols: List[str],
    schema_filter: Optional[str],
    table_prefix: Optional[str],
    top_n: int = 10,
    *,
    model: str = "PPDM 3.9",
    domain: str = "(All)",
) -> pd.DataFrame:
    """
    Candidate search:
      - Prefer JSON catalog to constrain tables/columns (fast)
      - Fallback to sys.columns if catalog missing/unavailable
    """
    src = {c.lower() for c in source_cols if c}

    try:
        cat = load_schema_catalog(model)
        tables = catalog_tables(cat, schema=schema_filter, domain=domain)

        if table_prefix:
            tables = tables[tables["table_name"].str.lower().str.startswith(table_prefix.lower())]

        # Score by column hits using catalog rows (no DB scan)
        cat2 = cat.merge(tables, on=["table_schema", "table_name"], how="inner")
        cat2["hit"] = cat2["column_name"].str.lower().isin(src).astype(int)

        score = (
            cat2.groupby(["table_schema", "table_name"], as_index=False)["hit"].sum()
                .rename(columns={"hit": "matches"})
                .sort_values("matches", ascending=False)
        )
        return score[score["matches"] > 0].head(top_n)

    except Exception:
        # fallback: scan sys.columns (slower)
        cols_df = fetch_all_columns(conn, schema_filter=schema_filter, table_prefix=table_prefix)
        cols_df["hit"] = cols_df["column_name"].str.lower().isin(src).astype(int)
        score = (
            cols_df.groupby(["schema_name", "table_name"], as_index=False)["hit"].sum()
                .rename(columns={"hit": "matches"})
                .sort_values("matches", ascending=False)
        )
        return score[score["matches"] > 0].head(top_n)
