# common/ui.py
from __future__ import annotations

from pathlib import Path
import json
import streamlit as st

import ppdm_loader.db as db
from ppdm_loader.seed_ui import render_seed_sidebar

CATALOG_DIR = Path(r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\schema_registry")


def _list_catalog_files() -> list[str]:
    if not CATALOG_DIR.exists():
        return []
    return sorted([p.name for p in CATALOG_DIR.glob("*.json")])


def _load_catalog_json(filename: str) -> dict:
    p = CATALOG_DIR / filename
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_database_names(conn) -> list[str]:
    sql = """
    SET NOCOUNT ON;
    SELECT name
    FROM sys.databases
    WHERE database_id > 4
      AND state_desc = 'ONLINE'
    ORDER BY name;
    """
    df = db.read_sql(conn, sql)
    if df is None or df.empty:
        return []
    return df["name"].astype(str).tolist()


def sidebar_connect(*, page_prefix: str = "") -> None:
    """
    Call ONCE per page script (near the top).
    Uses page_prefix to keep widget keys unique across pages.
    """
    ss = st.session_state

    # ---------- defaults ----------
    ss.setdefault("conn_server", "localhost")
    ss.setdefault("conn_driver", "ODBC Driver 18 for SQL Server")
    ss.setdefault("conn_use_sql_login", False)
    ss.setdefault("conn_user", "")
    ss.setdefault("conn_password", "")
    ss.setdefault("conn_params_confirmed", False)

    ss.setdefault("available_databases", [])
    ss.setdefault("conn_database", "")

    ss.setdefault("conn", None)

    ss.setdefault("ppdm_version", "PPDM 3.9")   # "PPDM Lite" also allowed
    ss.setdefault("ppdm_domain", "WELL")
    ss.setdefault("catalog_file_selectbox", "(none)")
    ss.setdefault("catalog_json", None)

    # ========== SEED FROM CATALOG (GLOBAL TOOL) ==========
    st.divider()
    st.subheader("Reference seeding")

    conn = ss.get("conn")
    if conn is None:
        st.caption("Connect to enable seeding.")
    else:
        # Map your UI PPDM version label -> seed flavor key
        ppdm_flavor = "ppdm39"
        if ss.get("ppdm_version", "").lower().startswith("ppdm lite"):
            ppdm_flavor = "ppdm_lite"

        render_seed_sidebar(conn, ppdm_flavor=ppdm_flavor)

    # ---------- UI ----------
    with st.sidebar:
        st.title("PPDM Studio")

        # ========== LAUNCHER (TOP) ==========
        st.subheader("Launcher")
        try:
            st.page_link("launchpad.py", label="🏠 Launchpad")
            st.page_link("pages/1_Seed_Reference_Tables.py", label="🌱 Seed r_ tables")
            st.page_link("pages/2_Load_Entity_Data.py", label="📥 Load entity data")
            st.page_link("pages/3_Load_Relationships.py", label="🔗 Load ra_ relationships")
            st.page_link("pages/4_QC_Reports.py", label="✅ QC reports")
        except Exception:
            st.caption("Use Streamlit’s Pages list to navigate.")

        st.divider()

        # ========== CONNECTION ==========
        st.subheader("SQL Server connection")

        st.text_input("Server", key=f"{page_prefix}_conn_server", value=ss["conn_server"])
        st.text_input("ODBC Driver", key=f"{page_prefix}_conn_driver", value=ss["conn_driver"])

        # sync back into stable session keys
        ss["conn_server"] = ss.get(f"{page_prefix}_conn_server", ss["conn_server"])
        ss["conn_driver"] = ss.get(f"{page_prefix}_conn_driver", ss["conn_driver"])

        ss["conn_use_sql_login"] = st.toggle(
            "Use SQL Login (otherwise Windows auth)",
            value=bool(ss.get("conn_use_sql_login", False)),
            key=f"{page_prefix}_use_sql_login",
        )

        auth_mode = "windows"
        if ss["conn_use_sql_login"]:
            auth_mode = "sql"
            st.text_input("User", key=f"{page_prefix}_conn_user", value=ss.get("conn_user", ""))
            st.text_input("Password", type="password", key=f"{page_prefix}_conn_password", value=ss.get("conn_password", ""))

            ss["conn_user"] = ss.get(f"{page_prefix}_conn_user", "")
            ss["conn_password"] = ss.get(f"{page_prefix}_conn_password", "")
        else:
            # keep them blank (but stable)
            ss["conn_user"] = ss.get("conn_user", "")
            ss["conn_password"] = ss.get("conn_password", "")

        cA, cB = st.columns(2)
        if cA.button("Confirm server/auth", type="primary", key=f"{page_prefix}_confirm_conn"):
            ss["conn_params_confirmed"] = True
            ss["available_databases"] = []
            ss["conn_database"] = ""
            st.success("Confirmed. Refresh DB list below.")

        if cB.button("Disconnect", key=f"{page_prefix}_disconnect"):
            try:
                if ss.get("conn") is not None and hasattr(ss["conn"], "close"):
                    ss["conn"].close()
            except Exception:
                pass
            ss["conn"] = None
            st.success("Disconnected.")

        # ========== DATABASE SELECT ==========
        st.subheader("Database")

        if not ss.get("conn_params_confirmed", False):
            st.caption("Confirm server/auth to enable DB listing.")
        else:
            r1, r2 = st.columns(2)
            if r1.button("Refresh databases", key=f"{page_prefix}_refresh_dbs"):
                try:
                    cm = db.connect_master(
                        server=ss["conn_server"],
                        auth=auth_mode,
                        user=ss.get("conn_user") or None,
                        password=ss.get("conn_password") or None,
                        driver=ss.get("conn_driver") or "ODBC Driver 18 for SQL Server",
                    )
                    dbs = _fetch_database_names(cm)
                    try:
                        cm.close()
                    except Exception:
                        pass

                    ss["available_databases"] = dbs
                    if dbs:
                        ss["conn_database"] = dbs[0] if ss.get("conn_database") not in dbs else ss["conn_database"]
                    st.success(f"Found {len(dbs)} database(s).")
                except Exception as e:
                    ss["available_databases"] = []
                    st.error(f"Refresh failed: {e}")

            dbs = ss.get("available_databases") or []
            if dbs:
                st.selectbox("Choose database", options=dbs, key=f"{page_prefix}_db_pick", index=dbs.index(ss["conn_database"]) if ss.get("conn_database") in dbs else 0)
                ss["conn_database"] = ss.get(f"{page_prefix}_db_pick", ss.get("conn_database", ""))
            else:
                st.selectbox("Choose database", options=[ss.get("conn_database", "") or ""], key=f"{page_prefix}_db_pick_empty")
                st.caption("No list yet — click Refresh databases.")

            if r2.button("Connect", type="primary", key=f"{page_prefix}_connect"):
                try:
                    if not ss.get("conn_database"):
                        raise ValueError("Pick a database first (refresh list).")

                    ss["conn"] = db.connect(
                        server=ss["conn_server"],
                        database=ss["conn_database"],
                        auth=auth_mode,
                        user=ss.get("conn_user") or None,
                        password=ss.get("conn_password") or None,
                        driver=ss.get("conn_driver") or "ODBC Driver 18 for SQL Server",
                    )
                    st.success(f"Connected to {ss['conn_database']}.")
                except Exception as e:
                    ss["conn"] = None
                    st.error(f"Connect failed: {e}")

        st.divider()

        # ========== PPDM VERSION ==========
        st.subheader("PPDM version")
        v1, v2 = st.columns(2)
        if v1.button("PPDM 3.9", use_container_width=True, key=f"{page_prefix}_ppdm39"):
            ss["ppdm_version"] = "PPDM 3.9"
        if v2.button("PPDM Lite (3.0)", use_container_width=True, key=f"{page_prefix}_ppdmlite"):
            ss["ppdm_version"] = "PPDM Lite (3.0)"
        st.caption(f"Active: **{ss.get('ppdm_version')}**")

        st.divider()

        # ========== JSON CATALOG CACHE ==========
        st.subheader("Cached schema (JSON)")
        files = _list_catalog_files()
        if not files:
            st.caption(f"No JSON files found in {CATALOG_DIR}.")
            ss["catalog_file_selectbox"] = "(none)"
        else:
            st.selectbox(
                "Schema catalog JSON",
                options=["(none)"] + files,
                key=f"{page_prefix}_catalog_pick",
            )
            ss["catalog_file_selectbox"] = ss.get(f"{page_prefix}_catalog_pick", "(none)")

        if st.button("Load selected JSON", key=f"{page_prefix}_load_json"):
            sel = ss.get("catalog_file_selectbox")
            if not sel or sel == "(none)":
                ss["catalog_json"] = None
                st.info("No catalog selected.")
            else:
                try:
                    ss["catalog_json"] = _load_catalog_json(sel)
                    st.success(f"Loaded: {sel}")
                except Exception as e:
                    ss["catalog_json"] = None
                    st.error(f"Catalog load failed: {e}")

        st.divider()

        # ========== DOMAIN FILTER ==========
        st.subheader("Domain filter")
        domains = [
            "WELL", "STRAT", "SEIS", "FIELD", "PROD", "LAND", "BUSINESS", "AREA", "PDEN"
        ]
        st.selectbox("Domain", options=domains, key=f"{page_prefix}_domain_pick", index=domains.index(ss.get("ppdm_domain", "WELL")) if ss.get("ppdm_domain", "WELL") in domains else 0)
        ss["ppdm_domain"] = ss.get(f"{page_prefix}_domain_pick", "WELL")


def require_connection():
    conn = st.session_state.get("conn")
    if conn is None:
        st.error("Not connected. Use the sidebar to connect.")
        st.stop()
    return conn
