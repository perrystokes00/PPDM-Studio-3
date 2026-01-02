# ppdm_loader/db.py
from __future__ import annotations

from typing import Any, Optional, Sequence
import re
import warnings

import pandas as pd
import pyodbc

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

_VIEW_TOKEN_RE = re.compile(
    r"\b(CREATE\s+(OR\s+ALTER\s+)?VIEW|ALTER\s+VIEW)\b",
    re.IGNORECASE,
)


def connect(
    *,
    server: str,
    database: str,
    auth: str = "windows",
    user: str | None = None,
    password: str | None = None,
    driver: str = "ODBC Driver 18 for SQL Server",
    trust_server_certificate: bool = True,
    encrypt: bool = False,
    timeout: int = 30,
) -> pyodbc.Connection:
    auth = (auth or "windows").strip().lower()

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Connection Timeout={int(timeout)}",
        f"Encrypt={'yes' if encrypt else 'no'}",
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'}",
        "MARS_Connection=yes",  # keeps Streamlit multi-queries from tripping
    ]

    if auth in ("windows", "trusted", "integrated", "sspi"):
        parts.append("Trusted_Connection=yes")
    elif auth in ("sql", "sqlserver", "sqlauth"):
        if not user:
            raise ValueError("SQL auth selected but user is blank.")
        parts.append(f"UID={user}")
        parts.append(f"PWD={password or ''}")
    else:
        raise ValueError(f"Unknown auth mode: {auth!r}")

    cs = ";".join(parts) + ";"
    conn = pyodbc.connect(cs, autocommit=False)

    # Optional but helps performance + stability
    try:
        conn.timeout = int(timeout)
    except Exception:
        pass

    return conn


def connect_master(
    *,
    server: str,
    auth: str = "windows",
    user: str = "",
    password: str = "",
    driver: str = "ODBC Driver 18 for SQL Server",
    trust_server_certificate: bool = True,
    encrypt: bool = False,
    timeout: int = 30,
) -> pyodbc.Connection:
    return connect(
        server=server,
        database="master",
        auth=auth,
        user=user or None,
        password=password or None,
        driver=driver,
        trust_server_certificate=trust_server_certificate,
        encrypt=encrypt,
        timeout=timeout,
    )


def _drain_all_resultsets(cur: pyodbc.Cursor) -> None:
    """
    IMPORTANT: prevents HY000 "Connection is busy with results..."
    by exhausting all remaining result sets.
    """
    try:
        # If current set has rows, read them (for SELECTs)
        try:
            if cur.description is not None:
                cur.fetchall()
        except Exception:
            pass

        # Move through all next resultsets
        while True:
            try:
                has_more = cur.nextset()
            except Exception:
                break
            if not has_more:
                break
            try:
                if cur.description is not None:
                    cur.fetchall()
            except Exception:
                pass
    except Exception:
        # Drain is best-effort
        pass


def read_sql(conn: pyodbc.Connection, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    """
    Robust SELECT reader:
      - advances through non-result statements
      - fetches rows
      - drains remaining resultsets
      - ALWAYS closes cursor
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, list(params) if params else [])

        # Advance until we find a resultset (or we run out)
        while cur.description is None:
            if not cur.nextset():
                _drain_all_resultsets(cur)
                return pd.DataFrame()

        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        _drain_all_resultsets(cur)
        return pd.DataFrame.from_records(rows, columns=cols)
    finally:
        try:
            cur.close()
        except Exception:
            pass


def exec_view_ddl(conn: pyodbc.Connection, view_sql: str) -> None:
    s = (view_sql or "")
    if not s.strip():
        raise ValueError("view_sql was empty")

    m = _VIEW_TOKEN_RE.search(s)
    if not m:
        raise ValueError("view_sql does not contain CREATE/ALTER VIEW")

    ddl = s[m.start():].lstrip()

    while ddl.startswith(";"):
        ddl = ddl[1:].lstrip()

    ddl = re.sub(r"^\s*GO\s*$", "", ddl, flags=re.IGNORECASE | re.MULTILINE).strip()

    cur = conn.cursor()
    try:
        cur.execute("EXEC sp_executesql ?", (ddl,))
        _drain_all_resultsets(cur)
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass


def exec_sql(conn: pyodbc.Connection, sql: str, params: Optional[Sequence[Any]] = None) -> None:
    """
    Execute general SQL and drain all result sets.
    """
    s = (sql or "")
    if _VIEW_TOKEN_RE.search(s):
        exec_view_ddl(conn, s)
        return

    cur = conn.cursor()
    try:
        cur.execute(s, list(params) if params else [])
        _drain_all_resultsets(cur)
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
