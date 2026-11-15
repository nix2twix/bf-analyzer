# app.py
import streamlit as st
from styles.styles import loadStyles, loadHeader, loadFooter

st.set_page_config(page_title="Biofilm Analyzer", layout="wide", page_icon="🧪")
loadStyles()
loadHeader()
analyzer_page = st.Page("pages/analyzer.py", title="Analyzer", icon=":material/biotech:", default=True)
docs_page = st.Page("pages/docs.py", title="Documentation", icon=":material/menu_book:")

pg = st.navigation([analyzer_page, docs_page], position="hidden")
pg.run()

loadFooter()