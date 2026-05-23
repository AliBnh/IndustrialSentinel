"""
Fleet health overview page for IndustrialSentinel dashboard.
"""
import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Fleet Overview", page_icon="📊", layout="wide")
st.title("📊 Fleet Health Overview")

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data(ttl=60)
def load_predictions():
    """Load test predictions from API or local file."""
    try:
        resp = requests.get(f"{API_URL}/results", timeout=10)
        if resp.status_code == 200:
            pass
    except Exception:
        pass

    # Load from processed data
    pred_path = os.environ.get("DATA_DIR", "data") + "/processed/test_predictions.csv"
    try:
        return pd.read_csv(pred_path)
    except FileNotFoundError:
        st.error("Predictions not found. Run training first.")
        return None


df = load_predictions()
if df is not None:
    # Add risk levels
    def get_risk(rul):
        if rul < 15:
            return "CRITICAL"
        elif rul < 30:
            return "HIGH"
        elif rul < 60:
            return "MEDIUM"
        return "LOW"

    df["risk_level"] = df["ensemble_pred"].apply(get_risk)
    df["alert"] = df["risk_level"].isin(["HIGH", "CRITICAL"])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Engines", len(df))
    col2.metric("🔴 Critical", len(df[df["risk_level"] == "CRITICAL"]))
    col3.metric("🟠 High Risk", len(df[df["risk_level"] == "HIGH"]))
    col4.metric("🟢 Low Risk", len(df[df["risk_level"] == "LOW"]))

    st.markdown("---")

    # Risk distribution
    st.subheader("Risk Distribution")
    risk_counts = df["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    # RUL distribution
    st.subheader("Predicted RUL Distribution")
    st.bar_chart(df.set_index("unit")["ensemble_pred"])

    # Fleet table
    st.subheader("Fleet Status Table")
    display_df = df[["unit", "ensemble_pred", "true_rul", "risk_level", "alert"]].copy()
    display_df.columns = ["Engine", "Predicted RUL", "True RUL", "Risk Level", "Alert"]
    display_df = display_df.sort_values("Predicted RUL")

    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #ffcccc" if row["Risk Level"] == "CRITICAL"
                        else "background-color: #ffe0b2" if row["Risk Level"] == "HIGH"
                        else "background-color: #fff9c4" if row["Risk Level"] == "MEDIUM"
                        else "background-color: #c8e6c9"] * len(row),
            axis=1
        ),
        use_container_width=True
    )
