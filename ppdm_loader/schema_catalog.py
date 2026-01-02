# ppdm_loader/schema_catalog.py
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set


@dataclass(frozen=True)
class ColMeta:
    column_name: str
    data_type: str
    not_null: bool
    is_pk: bool
    is_fk: bool
    fk_table_schema: Optional[str]
    fk_table_name: Optional[str]
    fk_column_name: Optional[str]
    category: Optional[str]
    sub_category: Optional[str]


class SchemaCatalog:
    """
    JSON-first schema catalog for PPDM table discovery + FK resolution.

    Supports an optional FK overlay (e.g., PDF-extracted FK list for PPDM Lite)
    that can populate fk_map even if the DB has no FK constraints.
    """

    def __init__(self, rows: List[dict]):
        self.rows = rows

        # (schema, table) -> list[ColMeta]
        self.table_cols: Dict[Tuple[str, str], List[ColMeta]] = {}

        # (schema, table) -> set(lowercase column names)
        self.table_colset: Dict[Tuple[str, str], Set[str]] = {}

        # (schema, table, child_col_upper) -> (parent_schema, parent_table, parent_col_or_None)
        # NOTE: parent_col may be None when the overlay doesn't provide it; we’ll infer it.
        self.fk_map: Dict[Tuple[str, str, str], Tuple[str, str, Optional[str]]] = {}

        # category -> set((schema, table))
        self.by_category: Dict[str, Set[Tuple[str, str]]] = {}

        self._build()

    # ---------------------------------------------------------
    # Load / overlay
    # ---------------------------------------------------------
    @staticmethod
    def load(
        json_path: str,
        root_key: Optional[str] = None,
        fk_overlay_path: Optional[str] = None,
        fk_overlay_default_schema: str = "dbo",
    ) -> "SchemaCatalog":
        """
        Load the schema catalog JSON and optionally apply an FK overlay.

        fk_overlay_path: path to ppdm_lite11_fk_from_pdfplumber.json (or similar)
        """
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Schema catalog JSON not found: {json_path}")

        obj = json.loads(p.read_text(encoding="utf-8"))

        # Your examples show top-level key "ppdm_39_schema_domain"
        if root_key:
            data = obj.get(root_key, [])
        else:
            # auto-detect list key
            if isinstance(obj, dict):
                list_keys = [k for k, v in obj.items() if isinstance(v, list)]
                if not list_keys:
                    raise ValueError("Catalog JSON must contain a list at top level or under a key.")
                data = obj[list_keys[0]]
            elif isinstance(obj, list):
                data = obj
            else:
                raise ValueError("Unexpected JSON format for catalog.")

        if not isinstance(data, list):
            raise ValueError("Catalog list was not a list.")

        cat = SchemaCatalog(data)

        if fk_overlay_path:
            cat.load_fk_catalog(
                fk_overlay_path,
                default_schema=fk_overlay_default_schema,
            )

        return cat

    def load_fk_catalog(self, fk_overlay_path: str, default_schema: str = "dbo") -> int:
        """
        Apply an FK overlay (PDF-extracted for PPDM Lite).

        Expected format (matches your generated file):
        {
          "model": "...",
          "fks": [
            {"child_table": "L_APPLICATION", "child_column": "DATA_SOURCE", "parent_tables": ["L_PPDM_DATA_SOURCE"]},
            ...
          ]
        }

        Returns number of FK relationships applied.
        """
        p = Path(fk_overlay_path)
        if not p.exists():
            raise FileNotFoundError(f"FK overlay JSON not found: {fk_overlay_path}")

        obj = json.loads(p.read_text(encoding="utf-8"))

        # find list of fks
        fks = None
        if isinstance(obj, dict):
            if isinstance(obj.get("fks"), list):
                fks = obj["fks"]
            else:
                # fallback: first list-valued key
                for _, v in obj.items():
                    if isinstance(v, list):
                        fks = v
                        break
        elif isinstance(obj, list):
            fks = obj

        if not isinstance(fks, list):
            raise ValueError("FK overlay JSON must contain a list of FK rows (e.g., under 'fks').")

        applied = 0
        for r in fks:
            child_table = (r.get("child_table") or "").strip()
            child_col = (r.get("child_column") or "").strip()
            parents = r.get("parent_tables") or []
            if not child_table or not child_col or not parents:
                continue

            # If multiple parent tables are listed, we store them all as separate candidates.
            # resolve_fk_parent() will choose the best parent column, but parent table ambiguity
            # may still require human review later.
            for pt in parents:
                parent_table = (pt or "").strip()
                if not parent_table:
                    continue

                self.fk_map[(default_schema, child_table, child_col.upper())] = (
                    default_schema,
                    parent_table,
                    None,  # parent column unknown in overlay; infer later
                )
                applied += 1

        return applied

    # ---------------------------------------------------------
    # Build indices from schema rows
    # ---------------------------------------------------------
    def _build(self) -> None:
        for r in self.rows:
            schema = (r.get("table_schema") or "").strip()
            table = (r.get("table_name") or "").strip()
            col = (r.get("column_name") or "").strip()
            if not schema or not table or not col:
                continue

            key = (schema, table)
            cat = (r.get("category") or r.get("Category") or "").strip() or None
            sub = (r.get("sub_category") or r.get("Sub-Category") or "").strip() or None

            is_pk = str(r.get("is_primary_key") or r.get("is_pk") or "NO").upper() in ("YES", "Y", "1", "TRUE")
            is_fk = str(r.get("is_foreign_key") or r.get("is_fk") or "NO").upper() in ("YES", "Y", "1", "TRUE")
            not_null = str(r.get("not_null") or "NO").upper() in ("YES", "Y", "1", "TRUE")

            cm = ColMeta(
                column_name=col,
                data_type=str(r.get("data_type") or ""),
                not_null=not_null,
                is_pk=is_pk,
                is_fk=is_fk,
                fk_table_schema=r.get("fk_table_schema"),
                fk_table_name=r.get("fk_table_name"),
                fk_column_name=r.get("fk_column_name"),
                category=cat,
                sub_category=sub,
            )

            self.table_cols.setdefault(key, []).append(cm)

            if cat:
                self.by_category.setdefault(cat.upper(), set()).add(key)

            # If schema JSON already has explicit FK target, keep it (more reliable than overlay).
            if is_fk and cm.fk_table_schema and cm.fk_table_name:
                self.fk_map[(schema, table, col.upper())] = (
                    cm.fk_table_schema,
                    cm.fk_table_name,
                    cm.fk_column_name,  # may be None in some extracts; we infer if missing
                )

        # build colsets
        for key, cols in self.table_cols.items():
            self.table_colset[key] = {c.column_name.lower() for c in cols}

    # ---------------------------------------------------------
    # Discovery + FK resolution
    # ---------------------------------------------------------
    def tables_in_category(self, category: Optional[str]) -> Iterable[Tuple[str, str]]:
        if not category:
            return self.table_cols.keys()
        return self.by_category.get(category.upper(), set())

    def discover_tables(
        self,
        source_cols: List[str],
        category: Optional[str] = None,
        schema_filter: Optional[str] = None,
        table_prefix: Optional[str] = None,
        top_n: int = 10,
    ) -> List[dict]:
        """
        Returns a list of dicts: {"schema":..., "table":..., "matches":...}
        """
        src_set = {c.lower() for c in source_cols if c}

        candidates: List[dict] = []
        for (schema, table) in self.tables_in_category(category):
            if schema_filter and schema.lower() != schema_filter.lower():
                continue
            if table_prefix and not table.lower().startswith(table_prefix.lower()):
                continue

            colset = self.table_colset.get((schema, table), set())
            matches = len(colset & src_set)
            if matches > 0:
                candidates.append({"schema": schema, "table": table, "matches": matches})

        candidates.sort(key=lambda x: x["matches"], reverse=True)
        return candidates[:top_n]

    def _infer_parent_pk_column(self, parent_schema: str, parent_table: str, child_column: str) -> Optional[str]:
        """
        Heuristics for overlays that don't carry parent_column:
          1) If parent table has exactly one PK column -> use it
          2) Else if child_column matches a PK column name -> use that
          3) Else None
        """
        cols = self.table_cols.get((parent_schema, parent_table), [])
        pk_cols = [c.column_name for c in cols if c.is_pk]
        if len(pk_cols) == 1:
            return pk_cols[0]
        for pk in pk_cols:
            if pk.upper() == child_column.upper():
                return pk
        return None

    def resolve_fk_parent(
        self,
        child_schema: str,
        child_table: str,
        child_column: str,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Returns (parent_schema, parent_table, parent_column).

        If the parent_column is missing (overlay), infer it from parent PK metadata.
        """
        hit = self.fk_map.get((child_schema, child_table, child_column.upper()))
        if not hit:
            return None

        ps, pt, pc = hit
        if pc:
            return (ps, pt, pc)

        inferred = self._infer_parent_pk_column(ps, pt, child_column)
        if inferred:
            # cache the inference
            self.fk_map[(child_schema, child_table, child_column.upper())] = (ps, pt, inferred)
            return (ps, pt, inferred)

        return None

    # ---------------------------------------------------------
    # UI helpers: categories, sub-categories, table lists, mapping DF
    # ---------------------------------------------------------
    def list_categories(self) -> List[str]:
        """Distinct category names (uppercased) present in the catalog."""
        return sorted(self.by_category.keys())

    def list_subcategories(self, category: str) -> List[str]:
        """Distinct sub-categories under a category."""
        cat = (category or "").strip().upper()
        if not cat or cat == "(ALL)":
            return []
        subs: Set[str] = set()
        for (schema, table) in self.by_category.get(cat, set()):
            for cm in self.table_cols.get((schema, table), []):
                if cm.sub_category:
                    subs.add(str(cm.sub_category))
        return sorted(subs)

    def list_tables(
        self,
        category: Optional[str] = None,
        sub_category: Optional[str] = None,
        schema_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Returns distinct tables as ["schema.table", ...] filtered by category/sub_category/schema.
        """
        cat = (category or "").strip().upper() or None
        sub = (sub_category or "").strip() or None
        sch = (schema_filter or "").strip().lower() or None

        tables: Set[Tuple[str, str]] = set()

        base = self.tables_in_category(cat) if cat else self.table_cols.keys()

        for (schema, table) in base:
            if sch and schema.lower() != sch:
                continue
            if sub:
                cols = self.table_cols.get((schema, table), [])
                if not any((c.sub_category or "") == sub for c in cols):
                    continue
            tables.add((schema, table))

        return sorted([f"{s}.{t}" for (s, t) in tables])

    def table_columns_df(self, schema: str, table: str):
        """
        Returns a pandas DataFrame suitable for the mapping grid:
        column_name, data_type, not_null, is_primary_key, is_foreign_key, fk_*
        """
        import pandas as pd  # local import to avoid forcing pandas in non-UI contexts

        key = (schema, table)
        cols = self.table_cols.get(key, [])
        if not cols:
            return pd.DataFrame(
                columns=[
                    "column_name",
                    "data_type",
                    "not_null",
                    "is_primary_key",
                    "is_foreign_key",
                    "fk_table_schema",
                    "fk_table_name",
                    "fk_column_name",
                    "category",
                    "sub_category",
                ]
            )

        rows = []
        for c in cols:
            rows.append(
                {
                    "column_name": c.column_name,
                    "data_type": c.data_type,
                    "not_null": "YES" if c.not_null else "NO",
                    "is_primary_key": "YES" if c.is_pk else "NO",
                    "is_foreign_key": "YES" if c.is_fk else "NO",
                    "fk_table_schema": c.fk_table_schema,
                    "fk_table_name": c.fk_table_name,
                    "fk_column_name": c.fk_column_name,
                    "category": c.category,
                    "sub_category": c.sub_category,
                }
            )

        df = pd.DataFrame(rows)
        # Show required+PK first
        return df.sort_values(
            ["is_primary_key", "not_null", "column_name"],
            ascending=[False, False, True],
        )
