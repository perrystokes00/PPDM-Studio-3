from __future__ import annotations

from typing import List, Tuple

from .db import read_sql


def qc_raw_vs_norm(conn, view_primary: str, pairs: List[Tuple[str, str]], limit: int = 25):
    """Render RAW + NORM columns adjacent for each mapped target column."""
    cols = []
    for tgt, src in pairs:
        cols.append(f"r.[{src}] AS [RAW__{tgt}]")
        cols.append(f"v.[{tgt}] AS [NORM__{tgt}]")

    sql = f"""
        SELECT TOP ({limit}) r.RID, {', '.join(cols)}
        FROM stg.v_raw_with_rid r
        LEFT JOIN {view_primary} v ON v.RID = r.RID
        ORDER BY r.RID;
    """
    return read_sql(conn, sql)
