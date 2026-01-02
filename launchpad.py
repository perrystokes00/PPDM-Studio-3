# launchpad.py
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

from common.ui import sidebar_connect

st.set_page_config(page_title="PPDM Studio — Launchpad", layout="wide")
sidebar_connect(page_prefix="lp")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("PPDM Studio — Launchpad")
st.markdown("Use the sidebar Launcher to open a tool.")
