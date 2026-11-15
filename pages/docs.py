# pages/docs.py
import streamlit as st

from core.guideContent import GuideContent
from core.messages import Message

_container = st.container(horizontal=True, vertical_alignment="bottom")
with _container: 
    st.markdown("# 📖 Documentation")

    language = st.segmented_control(
        "Language",
        ["RU", "EN"],
        default="RU",
        label_visibility="collapsed",
        key="docs_language",
    )

    if st.button("⬅ Back to segmentation", key="docs_back_top", type="primary"):
        st.switch_page("pages/analyzer.py")
tab_quick, tab_batch, tab_overlay, tab_examples, tab_faq = st.tabs([
    "🚀 Quick start",
    "📦 Batch",
    "🖱️ Overlay",
    "🖼️ Examples",
    "❓ FAQ",
])

with tab_quick:
    GuideContent.quickStart(language)
with tab_batch:
    GuideContent.batch(language)
with tab_overlay:
    GuideContent.overlay(language)
with tab_examples:
    GuideContent.examples(language)
with tab_faq:
    GuideContent.faq(language)
    st.markdown("")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("♻ Clear cache", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            if "image_cache" in st.session_state:
                st.session_state.image_cache = {}
            st.rerun()
    with col2:
        if st.button("⬅ Back to analyzer", width="stretch"):
            st.switch_page("pages/analyzer.py")

st.markdown("---")
st.subheader("📚 Citation & contacts")
Message.citationAndContacts()