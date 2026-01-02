# ppdm_loader/db.py
from __future__ import annotations

from typing import Any, Optional, Sequence
import re
import warnings

import pandas as pd
import pyodbc

_VIEW_TOKEN_RE = re.compile(
    r"\bCREATE\s+(OR\s+ALTER\s+)?VIEW\b",
    re.IGNORECASE,
)

ODBC_DRIVER = "{ODBC Driver 18 for SQL Server}"

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
)

# Detect view DDL anywhere in the batch (not just at start)
_VIEW_TOKEN_RE = re.compile(
    r"\b(CREATE\s+(OR\s+ALTER\s+)?VIEW|ALTER\s+VIEW)\b",
    re.IGNORECASE,
)


def _build_conn_str(
    server: str,
    database: str,
    auth: str = "windows",
    user: str = "",
    password: str = "",
    *,
    encrypt: bool = False,
    trust_server_certificate: bool = True,
    mars: bool = True,
    timeout: int = 30,
) -> str:
    auth = (auth or "windows").strip().lower()

    parts = [
        f"DRIVER={ODBC_DRIVER}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Connection Timeout={int(timeout)}",
        f"Encrypt={'yes' if encrypt else 'no'}",
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'}",
    ]

    if mars:
        parts.append("MARS_Connection=yes")

    if auth in ("windows", "trusted", "integrated", "sspi"):
        parts.append("Trusted_Connection=yes")
    elif auth in ("sql", "sqlserver", "sqlauth"):
        if not user:
            raise ValueError("SQL auth selected but 'user' is blank.")
        parts.append(f"UID={user}")
        parts.append(f"PWD={password}")
    else:
        raise ValueError(f"Unknown auth mode: {auth!r} (use 'windows' or 'sql')")

    return ";".join(parts) + ";"


def connect(
    server: str,
    database: str,
    auth: str = "windows",
    user: str = "",
    password: str = "",
    *,
    encrypt: bool = False,
    trust_server_certificate: bool = True,
    mars: bool = True,
    timeout: int = 30,
) -> pyodbc.Connection:
    conn_str = _build_conn_str(
        server=server,
        database=database,
        auth=auth,
        user=user,
        password=password,
        encrypt=encrypt,
        trust_server_certificate=trust_server_certificate,
        mars=mars,
        timeout=timeout,
    )
    return pyodbc.connect(conn_str)


def connect_master(
    server: str,
    auth: str = "windows",
    user: str = "",
    password: str = "",
    *,
    encrypt: bool = False,
    trust_server_certificate: bool = True,
    mars: bool = True,
    timeout: int = 30,
) -> pyodbc.Connection:
    return connect(
        server=server,
        database="master",
        auth=auth,
        user=user,
        password=password,
        encrypt=encrypt,
        trust_server_certificate=trust_server_certificate,
        mars=mars,
        timeout=timeout,
    )


def read_sql(conn, sql, params=None):
    # Clear any pending results on this connection (pyodbc + SQL Server safety)
    cur = conn.cursor()
    try:
        while True:
            try:
                more = cur.nextset()
                if not more:
                    break
            except Exception:
                break
    finally:
        cur.close()

    return pd.read_sql(sql, conn, params=list(params) if params else None)



def exec_view_ddl(conn: pyodbc.Connection, view_sql: str) -> None:
    """
    Execute CREATE VIEW / CREATE OR ALTER VIEW safely in SQL Server.

    SQL Server requires CREATE VIEW to be the FIRST statement in the batch.
    We fix that by stripping anything before the first CREATE/ALTER VIEW token,
    then executing via sp_executesql with a parameter (no quote-escaping bugs).
    """
    s = (view_sql or "")
    if not s.strip():
        raise ValueError("view_sql was empty")

    # Remove any prelude before the first view token
    m = _VIEW_TOKEN_RE.search(s)
    if not m:
        raise ValueError("view_sql does not contain CREATE/ALTER VIEW")

    ddl = s[m.start():].lstrip()

    # Remove leading semicolons if present
    while ddl.startswith(";"):
        ddl = ddl[1:].lstrip()

    # IMPORTANT: strip GO batch separators (not valid inside sp_executesql)
    ddl = re.sub(r"^\s*GO\s*$", "", ddl, flags=re.IGNORECASE | re.MULTILINE).strip()

    cur = conn.cursor()
    # Parameterized execution avoids any quoting/escaping issues
    cur.execute("EXEC sp_executesql ?", (ddl,))
    conn.commit()


def exec_sql(conn: pyodbc.Connection, sql: str, params=None) -> None:
    """
    Execute general SQL. If the batch contains view DDL anywhere, route to exec_view_ddl().
    """
    s = (sql or "")
    if not s.strip():
        return

    if _VIEW_TOKEN_RE.search(s):
        exec_view_ddl(conn, s)
        return

    cur = conn.cursor()
    cur.execute(s, list(params) if params else [])
    conn.commit()
