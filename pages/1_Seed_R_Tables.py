# pages/1_Seed_R_Tables.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

import ppdm_loader.db as db
from common.ui import sidebar_connect, require_connection
from ppdm_loader.seed_generic import seed_missing_rows
from ppdm_loader.seed_catalog import seed_from_catalog
from ppdm_loader.seed_ui import render_seed_sidebar
from ppdm_loader.seed_core import seed_missing_rows

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="Seed R Tables", layout="wide")
sidebar_connect(page_prefix="seedr")
conn = require_connection()

DEFAULT_LOADED_BY = "Perry M Stokes"


# ============================================================
# Helpers
# ============================================================
def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _qident(name: str) -> str:
    return "[" + (name or "").replace("]", "]]") + "]"


def _qfqn(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def _parse_target_from_name(name: str) -> Tuple[str, str]:
    name = (name or "").strip()
    if not name:
        return ("dbo", "")
    if "." in name:
        s, t = name.split(".", 1)
        return (s.strip() or "dbo", t.strip())
    return ("dbo", name.strip())


def _normalize_fqn(schema: str, table: str) -> str:
    return f"{(schema or 'dbo').strip()}.{(table or '').strip()}"

def load_seed_rows(path: Path) -> list[dict]:
    """
    Load seed rows from JSON or CSV.
    Returns list[dict].
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        import json
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "rows" in obj:
            return obj["rows"]
        if isinstance(obj, list):
            return obj
        raise ValueError("Unsupported JSON seed shape")

    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported seed file type: {path.suffix}")

def _db_fetch_table_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
SELECT c.name AS column_name
FROM sys.columns c
JOIN sys.objects o ON o.object_id = c.object_id
JOIN sys.schemas s ON s.schema_id = o.schema_id
WHERE s.name = ? AND o.name = ? AND o.type = 'U'
ORDER BY c.column_id;
"""
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return [str(x) for x in df["column_name"].tolist()]


def _db_fetch_pk_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
SELECT c.name AS pk_col
FROM sys.indexes i
JOIN sys.index_columns ic
  ON i.object_id = ic.object_id AND i.index_id = ic.index_id
JOIN sys.columns c
  ON c.object_id = ic.object_id AND c.column_id = ic.column_id
JOIN sys.objects o
  ON o.object_id = i.object_id
JOIN sys.schemas s
  ON s.schema_id = o.schema_id
WHERE i.is_primary_key = 1
  AND o.type = 'U'
  AND s.name = ?
  AND o.name = ?
ORDER BY ic.key_ordinal;
"""
    df = db.read_sql(conn, sql, params=[schema, table])
    if df is None or df.empty:
        return []
    return [str(x) for x in df["pk_col"].tolist()]


def _is_r_or_ra_table(schema: str, table: str) -> bool:
    t = (table or "").lower()
    return t.startswith("r_") or t.startswith("ra_")


def _load_json_from_upload(uploaded) -> Any:
    if uploaded is None:
        return None
    raw = uploaded.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return json.loads(raw)


def _load_json_from_path(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(raw)


def _extract_target_and_rows(seed_obj: Any, *, fallback_target: str) -> Tuple[str, list[dict[str, Any]]]:
    """
    Supports:
      - { "name": "dbo.r_x", "rows": [ {...}, ... ] }
      - { "tables": { "dbo.r_x": [ {...} ] } }
      - { "dbo.r_x": [ {...} ] }
      - [ {...}, {...} ]
    """
    if seed_obj is None:
        return ("", [])

    # Shape: list => rows
    if isinstance(seed_obj, list):
        rows = [r for r in seed_obj if isinstance(r, dict)]
        return (fallback_target, rows)

    if not isinstance(seed_obj, dict):
        return ("", [])

    # Shape: {name, rows}
    if "rows" in seed_obj and isinstance(seed_obj.get("rows"), list):
        name = str(seed_obj.get("name") or fallback_target or "").strip()
        rows = [r for r in seed_obj["rows"] if isinstance(r, dict)]
        return (name, rows)

    # Shape: {tables: {...}}
    if "tables" in seed_obj and isinstance(seed_obj["tables"], dict):
        tables = seed_obj["tables"]
        # try exact key first
        if fallback_target in tables and isinstance(tables[fallback_target], list):
            rows = [r for r in tables[fallback_target] if isinstance(r, dict)]
            return (fallback_target, rows)
        # try schema-less
        want_no_schema = fallback_target.split(".", 1)[-1]
        for k, v in tables.items():
            if str(k).lower().strip() in {fallback_target.lower(), want_no_schema.lower()} and isinstance(v, list):
                rows = [r for r in v if isinstance(r, dict)]
                return (str(k), rows)
        return ("", [])

    # Shape: {"dbo.r_x": [...]}
    want_no_schema = fallback_target.split(".", 1)[-1].lower()
    for k, v in seed_obj.items():
        if not isinstance(k, str):
            continue
        kl = k.lower().strip()
        if kl in {fallback_target.lower(), want_no_schema} and isinstance(v, list):
            rows = [r for r in v if isinstance(r, dict)]
            return (k, rows)

    return ("", [])

def _rows_to_df(rows: Any) -> pd.DataFrame:
    """
    Accept:
      - list[dict]  -> rows
      - dict with { "rows": [...] } -> rows
      - dict mapping table->rows -> caller should pass the value (list)
    """
    if rows is None:
        return pd.DataFrame()

    # If someone accidentally passed the wrapper dict, unwrap it
    if isinstance(rows, dict) and "rows" in rows and isinstance(rows["rows"], list):
        rows = rows["rows"]

    if isinstance(rows, list):
        data = [r for r in rows if isinstance(r, dict)]
        df = pd.DataFrame(data)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # normalize column names: strip + upper
    df.columns = [str(c).strip().upper() for c in df.columns]

    # normalize values to strings (consistent with your seed approach)
    for c in df.columns:
        df[c] = df[c].astype(str)

    return df

def _validate_pk_present(df: pd.DataFrame, pk_cols: list[str]) -> list[str]:
    if df is None or df.empty:
        return pk_cols[:]  # all missing effectively

    df_cols_u = {str(c).strip().upper() for c in df.columns}
    missing = [c for c in pk_cols if str(c).strip().upper() not in df_cols_u]
    return missing

def _seed_one(
    *,
    conn,
    target_schema: str,
    target_table: str,
    rows: list[dict[str, Any]],
    loaded_by: str,
    strict_columns: bool = True,   # Option A: True (error on extra columns)
) -> Tuple[int, str]:
    # -------------------------
    # DB truth
    # -------------------------
    pk_cols = _db_fetch_pk_columns(conn, target_schema, target_table)
    pk_cols = [c.strip().upper() for c in pk_cols]
    if not pk_cols:
        return (0, "ERROR: Target table has no PK (or PK not found).")

    db_cols = _db_fetch_table_columns(conn, target_schema, target_table)  # <-- add helper below
    db_cols_u = {c.strip().upper(): c.strip() for c in db_cols}           # upper -> actual

    df = _rows_to_df(rows)
    if df.empty:
        return (0, "ERROR: No rows found in seed JSON.")

    # Normalize df column name handling
    df.columns = [str(c) for c in df.columns]
    df_cols_upper = {str(c).upper(): str(c) for c in df.columns}
    pk_upper = [str(c).upper() for c in pk_cols]

    # -------------------------
    # PK validation (case-insensitive)
    # -------------------------
    missing_pk_upper = [c for c in pk_upper if c not in df_cols_upper]
    if missing_pk_upper:
        missing_db = [pk_cols[pk_upper.index(u)] for u in missing_pk_upper]
        return (0, f"ERROR: Seed missing PK column(s): {missing_db}")

    # Rename PK columns in DF to match DB casing (so downstream is consistent)
    rename_map = {}
    for pk in pk_cols:
        u = pk.upper()
        rename_map[df_cols_upper[u]] = db_cols_u.get(u, pk)  # use actual DB column case if known
    df = df.rename(columns=rename_map)

    # -------------------------
    # Strict columns rule (Option A)
    # -------------------------
    # Ensure all df columns exist in DB (case-insensitive)
    df_u = {c.upper(): c for c in df.columns}
    extra = [df_u[u] for u in df_u.keys() if u not in db_cols_u]
    if extra:
        if strict_columns:
            return (0, f"ERROR: Seed contains column(s) not in DB table {target_schema}.{target_table}: {sorted(extra)}")
        else:
            # Drop unknown columns
            df = df[[c for c in df.columns if c.upper() in db_cols_u]]

    # -------------------------
    # Auto-populate PPDM_GUID + audit cols if present in DB and missing in DF
    # -------------------------
    def _ensure_col(col_u: str, value_factory):
        if col_u in db_cols_u and col_u not in {c.upper() for c in df.columns}:
            actual = db_cols_u[col_u]
            df[actual] = [value_factory() for _ in range(len(df))]

    # PPDM_GUID: create GUIDs for rows that are missing it
    # If column exists AND df already has it, fill null/blank values too.
    if "PPDM_GUID" in db_cols_u:
        guid_col = db_cols_u["PPDM_GUID"]
        if guid_col not in df.columns:
            df[guid_col] = [str(uuid.uuid4()).upper() for _ in range(len(df))]
        else:
            # Fill missing/blank
            s = df[guid_col].astype("string")
            mask = s.isna() | (s.str.strip() == "")
            if mask.any():
                df.loc[mask, guid_col] = [str(uuid.uuid4()).upper() for _ in range(int(mask.sum()))]

    # Audit columns: only if present and not provided
    _ensure_col("ROW_CREATED_BY", lambda: loaded_by)
    _ensure_col("ROW_CHANGED_BY", lambda: loaded_by)

    # If you want dates forced but tables lack defaults, we can add datetime.now()
    # _ensure_col("ROW_CREATED_DATE", lambda: pd.Timestamp.utcnow())
    # _ensure_col("ROW_CHANGED_DATE", lambda: pd.Timestamp.utcnow())

    # -------------------------
    # Seed (missing-only idempotent insert)
    # -------------------------
    inserted = seed_missing_rows(
        conn,
        target_schema=target_schema,
        target_table=target_table,
        pk_cols=pk_cols,
        insert_df=df,
        loaded_by=loaded_by,
    )
    return (inserted, "OK")


# ============================================================
# UI
# ============================================================
st.title("Seed R Tables (DB-driven PKs + JSON seeding)")

with st.expander("🔎 Debug: connection sanity", expanded=False):
    who = db.read_sql(conn, "SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
    st.dataframe(_safe_df(who), hide_index=True, width="stretch")

loaded_by = st.text_input("Loaded by", value=DEFAULT_LOADED_BY, key="seedr_loaded_by")

st.subheader("Option A — Seed a single table from one JSON file")
single_up = st.file_uploader("Seed JSON (single)", type=["json"], key="seedr_single_json")

include_non_r = st.checkbox(
    "Include non r_/ra_ tables (e.g., dbo.ppdm_unit_of_measure)",
    value=False,
    help="If unchecked, only r_/ra_ tables are allowed.",
    key="seedr_include_non_r",
)

if st.button("Seed single JSON now", type="primary", key="seedr_seed_single_btn"):
    try:
        seed_obj = _load_json_from_upload(single_up)
        fallback_target = ""
        if single_up is not None:
            # e.g. dbo.r_dir_srvy_type.json -> dbo.r_dir_srvy_type
            stem = Path(single_up.name).stem
            fallback_target = stem.replace("__", ".").replace("..", ".")
        target_name, rows = _extract_target_and_rows(seed_obj, fallback_target=fallback_target)

        if not target_name:
            st.error("Could not determine target table from JSON. Provide {name: 'dbo.r_x', rows:[...]} format.")
        else:
            s, t = _parse_target_from_name(target_name)
            if not include_non_r and not _is_r_or_ra_table(s, t):
                st.error(f"Target {s}.{t} is not an r_/ra_ table (toggle 'include non r_/ra_' if intended).")
            else:
                inserted, status = _seed_one(conn=conn, target_schema=s, target_table=t, rows=rows, loaded_by=loaded_by)
                if status == "OK":
                    st.success(f"Seeded {inserted} row(s) into {s}.{t}.")
                else:
                    st.error(status)
    except Exception as e:
        st.error(f"Single seed failed: {e}")

st.divider()

st.subheader("Option B — Batch seed from a folder of JSON files (recommended for fresh projects)")
seed_folder_txt = st.text_input(
    "Seed-pack folder (contains many .json files)",
    value="",
    placeholder=r"C:\Users\perry\OneDrive\Documents\PPDM\ppdm39-seed-catalog\seeds",
    key="seedr_seed_folder",
)

c1, c2 = st.columns([1, 2])
with c1:
    st.caption("Batch rules")
    only_r = st.checkbox("Only seed r_/ra_ targets", value=not include_non_r, key="seedr_only_r")
with c2:
    st.caption("JSON target resolution: prefers JSON.name, else filename stem (dbo.r_x.json -> dbo.r_x)")

if st.button("Seed ALL now", type="primary", key="seedr_seed_all_btn"):
    folder = Path(seed_folder_txt.strip().strip('"')) if seed_folder_txt else None
    if folder is None or not folder.exists() or not folder.is_dir():
        st.error("Seed-pack folder not found. Provide a valid directory path that contains the JSON seed files.")
    else:
        files = sorted(folder.glob("*.json"))
        if not files:
            st.error("No .json files found in that folder.")
        else:
            results = []
            for p in files:
                try:
                    seed_obj = _load_json_from_path(p)
                    fallback_target = p.stem.replace("__", ".").replace("..", ".")
                    target_name, rows = _extract_target_and_rows(seed_obj, fallback_target=fallback_target)
                    if not target_name:
                        results.append(
                            {"file": p.name, "target": "", "inserted": 0, "status": "ERROR", "error": "Unrecognized seed JSON shape. Expected {name,rows} or {tables:{...}} or list-of-rows."}
                        )
                        continue

                    s, t = _parse_target_from_name(target_name)
                    target_fqn = _normalize_fqn(s, t)

                    if only_r and not _is_r_or_ra_table(s, t):
                        results.append(
                            {"file": p.name, "target": target_fqn, "inserted": 0, "status": "SKIP", "error": "Not an r_/ra_ table (skipped by filter)"}
                        )
                        continue

                    inserted, status = _seed_one(conn=conn, target_schema=s, target_table=t, rows=rows, loaded_by=loaded_by)
                    if status == "OK":
                        results.append({"file": p.name, "target": target_fqn, "inserted": inserted, "status": "OK", "error": ""})
                    else:
                        results.append({"file": p.name, "target": target_fqn, "inserted": 0, "status": "ERROR", "error": status.replace("ERROR: ", "")})
                except Exception as e:
                    results.append({"file": p.name, "target": "", "inserted": 0, "status": "ERROR", "error": str(e)})

            out = pd.DataFrame(results)
            st.subheader("Batch results")
            st.dataframe(out, hide_index=True, width="stretch")

            ok = out[out["status"] == "OK"]
            err = out[out["status"] == "ERROR"]
            st.caption(f"OK: {len(ok)} | ERROR: {len(err)} | TOTAL: {len(out)}")

st.divider()

st.subheader("Option C — Seed from seed catalog (recommended for batch)")

catalog_path = st.text_input(
    "Seed catalog JSON path",
    value=r"C:\Users\perry\OneDrive\Documents\PPDM\ppdm39-seed-catalog\catalog\ppdm39_seed_catalog.json"
)

only_r = st.checkbox("Only seed r_/ra_ tables", value=True)

if st.button("Seed from catalog", type="primary"):
    try:
        report = seed_from_catalog(
            conn,
            catalog_json_path=catalog_path,
            only_r_tables=only_r,
            loaded_by=loaded_by,
        )
        st.success("Catalog seeding completed")
        st.dataframe(report, hide_index=True, width="stretch")
    except Exception as e:
        st.error(f"Catalog seeding failed: {e}")

with st.expander("What this page expects (so it works every time)", expanded=False):
    st.markdown(
        "**Seed JSON formats supported**\n\n"
        "**Best (recommended):**\n"
        "```json\n"
        "{\n"
        '  "name": "dbo.r_dir_srvy_type",\n'
        '  "rows": [\n'
        '    {\n'
        '      "DIR_SRVY_TYPE": "MWD",\n'
        '      "LONG_NAME": "Measurement While Drilling",\n'
        '      "SOURCE": "SYNTH",\n'
        '      "ACTIVE_IND": "Y"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )