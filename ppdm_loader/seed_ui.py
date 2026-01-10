# ppdm_loader/seed_ui.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from ppdm_loader.seed_batch import seed_from_catalog


def render_seed_sidebar(conn, *, ppdm_flavor: str):
    ss = st.session_state

    default_root = r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\ppdm-39-seed-packs"
    if ppdm_flavor == "ppdm_lite":
        default_root = r"C:\Users\perry\OneDrive\Documents\PPDM_Studio_3\ppdm-lite-seed-packs"

    seed_pack_root = Path(st.text_input("Seed pack root", value=ss.get(f"{ppdm_flavor}_seed_root", default_root)))
    ss[f"{ppdm_flavor}_seed_root"] = str(seed_pack_root)

    loaded_by = st.text_input("Loaded by", value=ss.get("loaded_by", "Perry M Stokes"))
    ss["loaded_by"] = loaded_by

    catalog_dir = seed_pack_root / "catalog"
    catalog_name = "ppdm39_seed_catalog.json" if ppdm_flavor == "ppdm39" else "ppdm_lite_seed_catalog.json"
    catalog_path = catalog_dir / catalog_name

    log_dir = seed_pack_root / "logs"
    st.caption(f"Catalog: {catalog_path}")
    st.caption(f"Logs: {log_dir}")

    if st.button("Seed from Catalog", type="primary", key=f"btn_seed_from_catalog_{ppdm_flavor}"):
        if not seed_pack_root.exists():
            st.error("Seed pack root not found.")
        elif not catalog_path.exists():
            st.error("Catalog JSON not found. Build/create it under /catalog.")
        else:
            _, df, file_paths = seed_from_catalog(
                conn=conn,
                seed_pack_root=seed_pack_root,
                catalog_path=catalog_path,
                loaded_by=loaded_by,
                log_dir=log_dir,
            )
            ss[f"seed_report_{ppdm_flavor}"] = df
            ss[f"seed_log_files_{ppdm_flavor}"] = file_paths
            st.success("Seeding complete. Logs written to disk.")

    # Tiny status + file paths
    df = ss.get(f"seed_report_{ppdm_flavor}")
    if isinstance(df, pd.DataFrame) and not df.empty:
        ok = int((df["status"] == "OK").sum())
        skip = int((df["status"] == "SKIP").sum())
        err = int((df["status"] == "ERROR").sum())
        st.caption(f"Last run: OK {ok} • SKIP {skip} • ERROR {err}")

    files = ss.get(f"seed_log_files_{ppdm_flavor}")
    if isinstance(files, dict) and files:
        st.caption(f"Wrote CSV: {files.get('csv','')}")
        st.caption(f"Wrote JSONL: {files.get('jsonl','')}")


def render_seed_report_main(*, ppdm_flavor: str):
    df = st.session_state.get(f"seed_report_{ppdm_flavor}")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Seed Run Report")
        st.dataframe(df, use_container_width=True)
