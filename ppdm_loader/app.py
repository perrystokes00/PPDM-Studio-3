# --- app.py sidebar block (add/imports) ---
from pathlib import Path
import pandas as pd
import streamlit as st

from ppdm_loader.seed_catalog import build_catalog_from_seed_pack, save_catalog, load_catalog
from ppdm_loader.seed_batch import seed_from_catalog
from ppdm_loader.seed_strict import load_seed_rows_from_json, seed_missing_rows_strict


def render_seed_sidebar(conn, *, ppdm_flavor: str):
    """
    ppdm_flavor: "ppdm39" or "ppdm_lite"
    """
    st.sidebar.subheader("Seeding")

    seed_pack_root = Path(st.sidebar.text_input(
        "Extended seed pack root",
        value=r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\ppdm39_extended_seed_pack",  # change as needed
        help="Folder containing /seeds/*.json and /catalog (will be created)."
    ))

    created_by = st.sidebar.text_input("Audit user (ROW_CREATED_BY)", value="Perry M Stokes")

    # Catalog path based on flavor
    catalog_dir = seed_pack_root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    if ppdm_flavor == "ppdm39":
        catalog_path = catalog_dir / "ppdm39_seed_catalog.json"
        ppdm_version = "3.9"
    else:
        catalog_path = catalog_dir / "ppdm_lite_seed_catalog.json"
        ppdm_version = "lite"

    # -----------------------------
    # Build/refresh catalog
    # -----------------------------
    with st.sidebar.expander("Catalog", expanded=False):
        st.write("Build a single catalog by scanning seeds/**/*.json and introspecting PKs from the target DB.")
        if st.button("Build/Refresh Catalog", key=f"btn_build_catalog_{ppdm_flavor}"):
            if not seed_pack_root.exists():
                st.error("Seed pack root does not exist.")
            else:
                cat = build_catalog_from_seed_pack(
                    conn=conn,
                    seed_pack_root=seed_pack_root,
                    ppdm_version=ppdm_version,
                    schema_default="dbo",
                    created_by=created_by,
                )
                save_catalog(cat, catalog_path)
                st.success(f"Catalog saved: {catalog_path}")
                st.caption(f"Seeds discovered: {len(cat.seeds)}")

        if catalog_path.exists():
            cat = load_catalog(catalog_path)
            st.caption(f"Current catalog: {catalog_path.name}  •  entries: {len(cat.seeds)}")
        else:
            st.warning("Catalog not found yet. Build it first.")

    # -----------------------------
    # Seed from catalog (batch)
    # -----------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Seed from Catalog", type="primary", key=f"btn_seed_from_catalog_{ppdm_flavor}"):
        if not catalog_path.exists():
            st.sidebar.error("Catalog not found. Click Build/Refresh Catalog first.")
        else:
            logs, df = seed_from_catalog(
                conn=conn,
                seed_pack_root=seed_pack_root,
                catalog_path=catalog_path,
                created_by=created_by,
            )
            st.session_state[f"seed_report_{ppdm_flavor}"] = df

    # Show report
    df = st.session_state.get(f"seed_report_{ppdm_flavor}")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.sidebar.success("Seed run complete. Report shown on main page.")


def render_seed_report_main(ppdm_flavor: str):
    df = st.session_state.get(f"seed_report_{ppdm_flavor}")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Seed Run Report")
        st.dataframe(df, use_container_width=True)
        # quick summary
        ok = int((df["status"] == "OK").sum())
        skip = int((df["status"] == "SKIP").sum())
        err = int((df["status"] == "ERROR").sum())
        st.caption(f"OK: {ok} • SKIP: {skip} • ERROR: {err} • Total: {len(df)}")


# -----------------------------
# Option A: Single-table seeding (STRICT, no mapping grid)
# -----------------------------

def render_single_table_seed_strict(conn):
    st.subheader("Option A — Seed a Single Table (Strict, No Mapping Grid)")

    col1, col2 = st.columns([2, 3])
    with col1:
        table_fq = st.text_input("Target table (schema.table)", value="dbo.r_well_status")
        created_by = st.text_input("Audit user", value="Perry M Stokes", key="single_created_by")
    with col2:
        seed_file = st.text_input(
            "Seed JSON file path",
            value=r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_2\ppdm39_extended_seed_pack\seeds\r_well_status.json",
        )

    keys_csv = st.text_input("PK keys (comma-separated, in order)", value="STATUS_TYPE, STATUS")
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]

    if st.button("Seed Single Table (Strict)", key="btn_seed_single_strict"):
        p = Path(seed_file)
        if not p.exists():
            st.error("Seed file not found.")
            return
        try:
            rows = load_seed_rows_from_json(p)
            inserted = seed_missing_rows_strict(
                conn=conn,
                table_fq=table_fq,
                keys=keys,
                rows=rows,
                created_by=created_by,
            )
            st.success(f"Done. Inserted {inserted} missing row(s).")
        except Exception as e:
            st.error(str(e))
