"""
Main Streamlit dashboard for IndustrialSentinel.
Predictive maintenance intelligence for industrial rotating equipment.
"""
import streamlit as st

st.set_page_config(
    page_title="IndustrialSentinel",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🏭 IndustrialSentinel")
st.sidebar.markdown("Predictive Maintenance Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
st.sidebar.markdown("""
- 📊 Fleet Overview
- 🔧 Engine Detail
- 📈 Model Performance
- 📡 Drift Monitor
""")
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | MIT License")

st.title("🏭 IndustrialSentinel")
st.markdown("### Predictive maintenance intelligence for industrial rotating equipment")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**📊 Fleet Overview**\n\nMonitor all 100 engines at a glance. See risk distribution, alerts, and fleet health status.")
with col2:
    st.info("**🔧 Engine Detail**\n\nDive deep into individual engines. View sensor telemetry, RUL predictions, and anomaly scores.")
with col3:
    st.info("**📈 Model Performance**\n\nCompare XGBoost, LSTM, and Ensemble metrics. Verify no overfitting with predicted vs true plots.")

st.markdown("---")
st.markdown("#### System Architecture")
st.markdown("""
```
Raw Data → Feature Engineering → [XGBoost + LSTM + Autoencoder] → Ensemble → API → Dashboard
                                                                              ↓
                                                                         Prometheus → Grafana
```
""")

st.markdown("#### Quick Links")
link_col1, link_col2, link_col3, link_col4 = st.columns(4)
with link_col1:
    st.markdown("[📖 API Docs](http://localhost:8000/docs)")
with link_col2:
    st.markdown("[🔬 MLflow](http://localhost:5000)")
with link_col3:
    st.markdown("[📊 Grafana](http://localhost:3000)")
with link_col4:
    st.markdown("[⚡ Prometheus](http://localhost:9090)")
