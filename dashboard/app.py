"""
Main Streamlit dashboard for IndustrialSentinel.
"""
import streamlit as st

st.set_page_config(
    page_title="IndustrialSentinel",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 IndustrialSentinel")
st.markdown("**Predictive maintenance intelligence for industrial rotating equipment**")
st.markdown("---")
st.markdown("Use the sidebar to navigate between pages:")
st.markdown("""
- **Overview** — Fleet health summary
- **Engine Detail** — Per-engine analysis
- **Model Performance** — Training metrics and comparisons
- **Drift Monitor** — Sensor drift detection
""")
