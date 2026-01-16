# ppdm_loader/station_promote.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class StationSignature:
    parent_schema: str
    parent_table: str
    parent_keys: List[str]     # child columns participating in FK (subset of PK)
    seq_col: str               # the PK "extra" column (sequence/ordinal)


@dataclass
class StationPlan:
    enabled: bool
    reason: str
    signature: Optional[StationSignature] = None
    order_cols: Optional[List[str]] = None


def _u(x: str) -> str:
    return (x or "").strip().upper()


def choose_station_order_cols(view_cols: Sequence[str]) -> List[str]:
    """
    Pick best ordering columns for station sequencing.
    IMPORTANT: DO NOT assume STATION_ID exists (it often doesn't in PPDM).
    """
    prefer = [
        "STATION_MD",
        "MD",
        "MEASURED_DEPTH",
        "DEPTH",
        "TVD",
        "STATION_TVD",
        "STATION_TVDSS",
        "RID",  # your NORM views often include RID
    ]
    view_u = {_u(c): c for c in view_cols}
    chosen = [view_u[p] for p in prefer if p in view_u]
    return chosen


def fetch_station_signature(
    conn,
    *,
    child_schema: str,
    child_table: str,
    fetch_pk_columns,
    introspect_all_fks_for_child_table,
) -> Optional[StationSignature]:
    """
    Detect a "station table" shape:
      PK = (FK cols to ONE parent) + (one extra col)
    Returns StationSignature or None.

    Notes:
    - Works off PK + FK metadata only (no table-specific hardcoding).
    - Handles FK introspection returning either list[dict] or pandas DataFrame.
    """
    pk_cols = fetch_pk_columns(conn, schema=child_schema, table=child_table) or []
    if not pk_cols:
        return None

    fk_rows = introspect_all_fks_for_child_table(conn, child_schema=child_schema, child_table=child_table)
    if fk_rows is None:
        return None

    # DataFrame -> list[dict]
    if hasattr(fk_rows, "to_dict"):
        if getattr(fk_rows, "empty", False):
            return None
        fk_rows = fk_rows.to_dict(orient="records")

    if not fk_rows:
        return None

    # group by FK name
    fks: Dict[str, Dict[str, Any]] = {}
    for r in fk_rows:
        fk_name = r.get("fk_name") or r.get("FK_NAME") or r.get("constraint_name") or r.get("CONSTRAINT_NAME")
        if not fk_name:
            continue

        parent_schema = (
            r.get("parent_schema")
            or r.get("PARENT_SCHEMA")
            or r.get("fk_table_schema")
            or r.get("PARENT_TABLE_SCHEMA")
            or "dbo"
        )
        parent_table = (
            r.get("parent_table")
            or r.get("PARENT_TABLE")
            or r.get("fk_table_name")
            or r.get("PARENT_TABLE_NAME")
        )
        child_col = (
            r.get("child_col")
            or r.get("CHILD_COL")
            or r.get("fk_column_name")
            or r.get("COLUMN_NAME")
        )

        if not parent_table or not child_col:
            continue

        d = fks.setdefault(str(fk_name), {
            "parent_schema": str(parent_schema),
            "parent_table": str(parent_table),
            "child_cols": [],
        })
        d["child_cols"].append(str(child_col))

    pk_u = {_u(c) for c in pk_cols}

    # find an FK whose child cols are subset of PK and PK has exactly one extra col
    for info in fks.values():
        child_cols: List[str] = info["child_cols"]
        if not child_cols:
            continue

        fk_u = {_u(c) for c in child_cols}
        if not fk_u.issubset(pk_u):
            continue

        extras = [c for c in pk_cols if _u(c) not in fk_u]
        if len(extras) != 1:
            continue

        return StationSignature(
            parent_schema=info["parent_schema"],
            parent_table=info["parent_table"],
            parent_keys=child_cols,
            seq_col=extras[0],
        )

    return None


def plan_station_mode(
    *,
    signature: Optional[StationSignature],
    view_cols: Sequence[str],
    pk_cols: Sequence[str],
) -> StationPlan:
    """
    Decide whether station sequencing should be enabled.

    Core rule (your recent win):
    - If the station sequence column already exists in the NORM view, DO NOT sequence.
      (Use v.<seq_col> directly.)

    Enable sequencing ONLY when:
    - signature exists AND
    - all parent keys are in view AND
    - seq_col is part of PK AND
    - seq_col is NOT already in view (needs generation)
    """
    if not signature:
        return StationPlan(enabled=False, reason="No station signature detected.")

    view_u = {_u(c) for c in view_cols}
    pk_u = {_u(c) for c in pk_cols}

    # must be a PK column
    if _u(signature.seq_col) not in pk_u:
        return StationPlan(enabled=False, reason=f"Signature seq_col {signature.seq_col} is not in PK.")

    # must be able to partition by parent keys
    missing_parent = [c for c in signature.parent_keys if _u(c) not in view_u]
    if missing_parent:
        return StationPlan(
            enabled=False,
            reason=f"Missing parent key(s) in view: {missing_parent}.",
        )

    # IMPORTANT: if view already has seq col, do NOT generate it
    if _u(signature.seq_col) in view_u:
        return StationPlan(
            enabled=False,
            reason=f"Sequence column {signature.seq_col} already present in NORM view.",
            signature=signature,
        )

    order_cols = choose_station_order_cols(view_cols)
    if not order_cols:
        return StationPlan(
            enabled=False,
            reason="No suitable station order columns found in NORM view (need MD/TVD/RID, etc.).",
            signature=signature,
        )

    return StationPlan(
        enabled=True,
        reason="Station sequencing enabled (seq_col missing in view).",
        signature=signature,
        order_cols=order_cols,
    )


def build_station_seq_sql(
    *,
    plan: StationPlan,
    view_alias: str = "v",
) -> Tuple[str, str]:
    """
    Returns:
      (using_projection_sql, seq_expr_sql)

    - using_projection_sql: SQL fragment to include in a USING subquery (MERGE) or SELECT list (INSERT)
    - seq_expr_sql: expression that yields the sequence column value (e.g., CAST(ROW_NUMBER()...) AS numeric(8,0))

    Caller decides where/how to embed.

    Only valid when plan.enabled is True.
    """
    if not plan.enabled or not plan.signature or not plan.order_cols:
        return ("", "")

    sig = plan.signature
    part = ", ".join([f"{view_alias}.[{c}]" for c in sig.parent_keys])
    order = ", ".join([f"TRY_CONVERT(float, {view_alias}.[{c}])" if _u(c) in {"STATION_MD", "MD", "DEPTH", "TVD", "STATION_TVD", "STATION_TVDSS"} else f"{view_alias}.[{c}]"
                       for c in plan.order_cols])

    # numeric(8,0) matches DEPTH_OBS_NO style columns
    seq_expr = (
        "CAST(ROW_NUMBER() OVER ("
        f"PARTITION BY {part} "
        f"ORDER BY {order}"
        ") AS numeric(8,0))"
    )

    using_proj = f"{seq_expr} AS [{sig.seq_col}]"
    return (using_proj, seq_expr)
    
def station_seq_expr_sql(*, plan, view_alias: str = "v") -> str:
    """
    ROW_NUMBER sequence expression matching numeric(8,0).
    Use ONLY when plan.enabled is True.
    """
    if not getattr(plan, "enabled", False) or not getattr(plan, "signature", None) or not getattr(plan, "order_cols", None):
        return ""

    sig = plan.signature
    parent_keys = getattr(sig, "parent_keys", []) or []
    if not parent_keys:
        return ""

    part = ", ".join([f"{view_alias}.[{c}]" for c in parent_keys])

    order_bits = []
    for c in (plan.order_cols or []):
        cu = (c or "").strip().upper()
        if cu in {"STATION_MD", "MD", "MEASURED_DEPTH", "DEPTH", "TVD", "STATION_TVD", "STATION_TVDSS",
                  "TOP_MD", "BASE_MD", "TOP_TVD", "BASE_TVD", "TOP_DEPTH", "BASE_DEPTH"}:
            order_bits.append(f"TRY_CONVERT(float, {view_alias}.[{c}])")
        else:
            order_bits.append(f"{view_alias}.[{c}]")

    order = ", ".join(order_bits) if order_bits else f"{view_alias}.[{parent_keys[0]}]"

    return (
        "CAST(ROW_NUMBER() OVER ("
        f"PARTITION BY {part} "
        f"ORDER BY {order}"
        ") AS numeric(8,0))"
    )
