"""Streamlit Cloud entry point — loads the main UI module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

importlib.import_module("app.ui.streamlit_app")
