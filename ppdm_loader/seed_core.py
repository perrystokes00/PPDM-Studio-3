# ppdm_loader/seed_core.py
from __future__ import annotations

from typing import List, Tuple, Any
import pandas as pd


def seed_missing_rows(
    conn,
    *,
    target_schema: str,
    target_table: str,
    pk_cols: List[str],
    insert_df: pd.DataFrame,
    loaded_by: str,
) -> int:
    """
    Idempotent insert: only insert rows whose PK tuple does not already exist.
    Never updates existing rows.

    Notes:
      - pk_cols should match DB column names (case already normalized upstream)
      - insert_df columns must exist in target table
      - insert_df may include PPDM_GUID / audit columns
    """
    if insert_df is None or insert_df.empty:
        return 0

    pk_cols = [c.strip() for c in pk_cols if str(c).strip()]
    if not pk_cols:
        raise ValueError("seed_missing_rows: pk_cols is empty")

    # Fetch existing PK tuples
    pk_sql = ", ".join(f"[{c}]" for c in pk_cols)
    sql_existing = f"SELECT {pk_sql} FROM [{target_schema}].[{target_table}];"

    cur = conn.cursor()
    cur.execute(sql_existing)
    existing = set()
    for row in cur.fetchall():
        existing.add(tuple(row))

    # Filter DF to missing PK tuples
    def _pk_tuple(sr: pd.Series) -> Tuple[Any, ...]:
        return tuple(sr[c] for c in pk_cols)

    mask_missing = []
    for _, sr in insert_df.iterrows():
        mask_missing.append(_pk_tuple(sr) not in existing)

    if not any(mask_missing):
        return 0

    df_ins = insert_df.loc[mask_missing].copy()

    # INSERT
    cols = list(df_ins.columns)
    col_sql = ", ".join(f"[{c}]" for c in cols)
    val_sql = ", ".join("?" for _ in cols)

    insert_sql = f"""
    INSERT INTO [{target_schema}].[{target_table}]
    ({col_sql})
    VALUES ({val_sql});
    """

    values = df_ins[cols].values.tolist()

    cur.fast_executemany = True
    cur.executemany(insert_sql, values)
    conn.commit()
    return len(df_ins)
