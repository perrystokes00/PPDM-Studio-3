# common/state.py
from __future__ import annotations
import streamlit as st

def ss_get(key: str, default=None):
    return st.session_state.get(key, default)

def ss_set(key: str, value):
    st.session_state[key] = value
