# launchpad.py
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

from common.ui import sidebar_connect

st.set_page_config(
    page_title="PPDM Studio — Launchpad",
    layout="wide",
)

sidebar_connect(page_prefix="lp")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------
# Header / Branding
# --------------------------------------------------
BANNER = ROOT / "assets" / "ppdm_loader_banner.png"

if BANNER.exists():
    st.image(BANNER, use_container_width=True)
else:
    st.title("PPDM Studio — Launchpad")

st.markdown(
    """
    **PPDM Loader Studio**  
    PPDM 3.9 & Lite • ETL Pipelines • Data Explorer • Schema ERD Viewer
    """
)

st.divider()

st.markdown("Use the sidebar launcher to open a tool.")
