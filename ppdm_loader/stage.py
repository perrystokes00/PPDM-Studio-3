# ppdm_loader/stage.py
from __future__ import annotations

from pathlib import Path
from typing import List
import hashlib
import time

import ppdm_loader.db as db


# UI -> actual delimiter char
DELIM_MAP = {
    ",": ",",
    "TAB": "\t",
    "|": "|",
    ";": ";",
}


def save_upload(uploaded, bulk_root: str | Path) -> Path:
    """
    Save a Streamlit UploadedFile to disk and return the file path.

    bulk_root can be a str or Path.
    Saves as: <stem>__<sha1_12>__<timestamp>.<ext>
    """
    bulk_root = Path(bulk_root)  # IMPORTANT: convert str -> Path
    bulk_root.mkdir(parents=True, exist_ok=True)

    data = uploaded.getbuffer()

    orig = Path(uploaded.name)
    sha = hashlib.sha1(data).hexdigest()[:12]
    ts = int(time.time())

    out_name = f"{orig.stem}__{sha}__{ts}{orig.suffix}"
    out_path = bulk_root / out_name

    with open(out_path, "wb") as f:
        f.write(data)

    return out_path


def stage_bulk_insert(
    conn,
    *,
    file_path: str,
    delimiter: str,
    has_header: bool,
    rowterm_sql: str,
) -> List[str]:
    """
    BULK INSERT file into stg.raw_data, then create stg.v_raw_with_rid.
    Returns the detected source column names.
    """
    db.exec_sql(conn, "IF SCHEMA_ID('stg') IS NULL EXEC('CREATE SCHEMA stg')")

    # ---- derive columns deterministically (no pandas guessing) ----
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().rstrip("\r\n")

    if has_header:
        raw_cols = [c.strip() for c in first_line.split(delimiter)]
        import re

        def clean(name: str) -> str:
            s = (name or "").strip()
            s = re.sub(r"[\s\.\-\/\\]+", "_", s)
            s = re.sub(r"[^0-9A-Za-z_]", "", s)
            if not s:
                s = "COL"
            if s[0].isdigit():
                s = f"C_{s}"
            return s[:120]

        cols = [clean(c) for c in raw_cols]
    else:
        n = len(first_line.split(delimiter))
        cols = [f"COL_{i+1}" for i in range(n)]

    # de-dupe
    seen = {}
    source_cols: List[str] = []
    for c in cols:
        k = seen.get(c, 0)
        if k == 0:
            source_cols.append(c)
        else:
            source_cols.append(f"{c}_{k+1}")
        seen[c] = k + 1

    cur = conn.cursor()
    cur.execute("IF OBJECT_ID('stg.raw_data','U') IS NOT NULL DROP TABLE stg.raw_data;")

    cols_sql = ", ".join(f"[{c}] NVARCHAR(MAX) NULL" for c in source_cols)
    cur.execute(f"CREATE TABLE stg.raw_data ({cols_sql});")

    firstrow = 2 if has_header else 1

    # rowterm_sql should be literal r"\n" or r"\r\n"
    if "\n" in rowterm_sql or "\r" in rowterm_sql:
        raise ValueError("rowterm_sql must be r'\\n' or r'\\r\\n' (escaped literal)")

    delimiter_safe = delimiter.replace("'", "''")
    rowterm_safe = rowterm_sql.replace("'", "''")
    file_path_safe = file_path.replace("'", "''")

    bulk_sql = f"""
    BULK INSERT stg.raw_data
    FROM '{file_path_safe}'
    WITH (
        FIRSTROW = {firstrow},
        FIELDTERMINATOR = '{delimiter_safe}',
        ROWTERMINATOR = '{rowterm_safe}',
        TABLOCK,
        KEEPNULLS,
        CODEPAGE = '65001'
    );
    """
    cur.execute(bulk_sql)
    conn.commit()

    db.exec_sql(
        conn,
        """
        CREATE OR ALTER VIEW stg.v_raw_with_rid AS
        SELECT ROW_NUMBER() OVER (ORDER BY (SELECT 1)) AS RID, *
        FROM stg.raw_data;
        """,
    )

    return source_cols
