# ppdm_loader/schema_catalog.py
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any


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

    Supports:
      - Base schema catalog (your domain JSON with columns)
      - Optional FK overlay (PDF-derived Lite FK mapping):
          { "fks": [ { "child_table": "...", "child_column": "...", "parent_tables": ["..."] }, ... ] }
    """

    def __init__(self, rows: List[dict]):
        self.rows = rows

        # (schema, table) -> list[ColMeta]
        self.table_cols: Dict[Tuple[str, str], List[ColMeta]] = {}

        # (schema, table) -> set(lowercase column names)
        self.table_colset: Dict[Tuple[str, str], Set[str]] = {}

        # (schema, table, child_col_upper) -> (parent_schema, parent_table, parent_col)
        self.fk_map: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}

        # category -> set((schema, table))
        self.by_category: Dict[str, Set[Tuple[str, str]]] = {}

        self._build_base()

    # -----------------------------
    # Load
    # -----------------------------
    @staticmethod
    def load(
        json_path: str,
        root_key: Optional[str] = None,
        fk_overlay_path: Optional[str] = None,
        fk_overlay_default_schema: str = "dbo",
    ) -> "SchemaCatalog":
        """
        Load base catalog JSON and optionally overlay FK mapping.

        fk_overlay_path:
          - path to your PDF-extracted FK json (ppdm_lite11_fk_from_pdfplumber.json)
        """
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Schema catalog JSON not found: {json_path}")

        obj = json.loads(p.read_text(encoding="utf-8"))

        if root_key:
            data = obj.get(root_key, [])
        else:
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

        # overlay FKs (optional)
        if fk_overlay_path:
            cat.load_fk_catalog(fk_overlay_path, default_schema=fk_overlay_default_schema)

        return cat

    # -----------------------------
    # Build base (your domain JSON)
    # -----------------------------
    def _build_base(self) -> None:
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

            # if base catalog already has FK detail, prefer it
            if is_fk and cm.fk_table_schema and cm.fk_table_name:
                parent_col = cm.fk_column_name or self._guess_parent_key(cm.fk_table_schema, cm.fk_table_name, child_column=col)
                if parent_col:
                    self.fk_map[(schema, table, col.upper())] = (cm.fk_table_schema, cm.fk_table_name, parent_col)

        for key, cols in self.table_cols.items():
            self.table_colset[key] = {c.column_name.lower() for c in cols}

    # -----------------------------
    # FK overlay (PDF-derived)
    # -----------------------------
    def load_fk_catalog(self, fk_overlay_path: str, default_schema: str = "dbo") -> None:
        """
        Overlay FK relationships onto fk_map.

        Expected format (what you generated):
          {
            "model": "...",
            "fks": [
              {"child_table":"L_APPLICATION","child_column":"DATA_SOURCE","parent_tables":["L_PPDM_DATA_SOURCE"]},
              ...
            ]
          }

        Since overlay often has no parent column, we guess it.
        """
        p = Path(fk_overlay_path)
        if not p.exists():
            raise FileNotFoundError(f"FK overlay JSON not found: {fk_overlay_path}")

        obj = json.loads(p.read_text(encoding="utf-8"))
        fks = obj.get("fks", obj if isinstance(obj, list) else [])
        if not isinstance(fks, list):
            raise ValueError("FK overlay must contain a list at key 'fks' (or be a list).")

        # Build a case-insensitive table-name index from base catalog
        # so overlay 'L_APPLICATION' can match base 'l_application'
        base_tables = {(s.lower(), t.lower()): (s, t) for (s, t) in self.table_cols.keys()}

        for row in fks:
            ct = str(row.get("child_table") or "").strip()
            cc = str(row.get("child_column") or "").strip()
            parents = row.get("parent_tables") or []

            if not ct or not cc or not parents:
                continue

            # normalize child table to what base catalog uses (if present)
            child_key = base_tables.get((default_schema.lower(), ct.lower()), (default_schema, ct))
            child_schema, child_table = child_key

            for pt in parents:
                pt = str(pt).strip()
                if not pt:
                    continue

                parent_key = base_tables.get((default_schema.lower(), pt.lower()), (default_schema, pt))
                parent_schema, parent_table = parent_key

                parent_col = self._guess_parent_key(parent_schema, parent_table, child_column=cc)
                if not parent_col:
                    # last resort: just use child column name
                    parent_col = cc

                self.fk_map[(child_schema, child_table, cc.upper())] = (parent_schema, parent_table, parent_col)

    def _guess_parent_key(self, parent_schema: str, parent_table: str, child_column: str) -> Optional[str]:
        """
        Guess parent key column:
          1) If parent has exactly one PK -> use it
          2) If parent has a column with same name as child_column -> use it
          3) If parent has a column ending with _ID and exactly one such -> use it
          4) Else None
        """
        cols = self.table_cols.get((parent_schema, parent_table))
        if not cols:
            return None

        pks = [c.column_name for c in cols if c.is_pk]
        if len(pks) == 1:
            return pks[0]

        # exact match
        for c in cols:
            if c.column_name.strip().upper() == child_column.strip().upper():
                return c.column_name

        # single *_ID fallback
        id_cols = [c.column_name for c in cols if c.column_name.upper().endswith("_ID")]
        if len(id_cols) == 1:
            return id_cols[0]

        return None

    # -----------------------------
    # Discovery + FK resolve
    # -----------------------------
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
        src_set = {c.lower() for c in source_cols if c}

        candidates = []
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

    def resolve_fk_parent(
        self,
        child_schema: str,
        child_table: str,
        child_column: str,
    ) -> Optional[Tuple[str, str, str]]:
        # exact
        hit = self.fk_map.get((child_schema, child_table, child_column.upper()))
        if hit:
            return hit

        # case-insensitive fallback
        cs = child_schema.lower()
        ct = child_table.lower()
        cc = child_column.upper()
        for (s, t, c), v in self.fk_map.items():
            if s.lower() == cs and t.lower() == ct and c == cc:
                return v
        return None
