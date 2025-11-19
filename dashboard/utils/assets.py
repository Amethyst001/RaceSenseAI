"""Asset loading utilities for dashboard."""
import streamlit as st
from pathlib import Path


def load_css():
    """Load and inject custom CSS."""
    css_file = Path(__file__).parent.parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def load_js():
    """Load and inject custom JavaScript."""
    js_file = Path(__file__).parent.parent / "assets" / "custom.js"
    if js_file.exists():
        with open(js_file, 'r', encoding='utf-8') as f:
            js_content = f.read()
        st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)


def load_all_assets():
    """Load all custom assets (CSS and JS)."""
    load_css()
    load_js()
