"""
Fleet health overview page for IndustrialSentinel dashboard.
Shows all 100 engines at a glance with risk distribution and alerts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Fleet Overview", page_icon="📊", layout="wide")
st.title("📊 Fleet Health Overview")
st.markdown("Real-time fleet status across all monitored engines")
st.markdown("---")

DATA_DIR = os.environ.get("DATA_DIR", "data")


@st.cache_data(ttl=60)
def load_predictions():
    """Load test predictions from processed data."""
    pred_path = f"{DATA_DIR}/processed/test_predictions.csv"
    try:
        return pd.read_csv(pred_path)
    except FileNotFoundError:
        st.error("⚠️ Predictions not found. Run training first.")
        return None


def get_risk(rul):
    """Classify risk level from predicted RUL."""
    if rul < 15:
        return "🔴 CRITICAL"
    elif rul < 30:
        return "🟠 HIGH"
    elif rul < 60:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def get_risk_color(risk):
    """Get background color for risk level."""
    if "CRITICAL" in risk:
        return "#ff4444"
    elif "HIGH" in risk:
        return "#ff8800"
    elif "MEDIUM" in risk:
        return "#ffcc00"
    return "#44bb44"


df = load_predictions()
if df is not None:
    df["risk_level"] = df["ensemble_pred"].apply(get_risk)
    df["error"] = (df["ensemble_pred"] - df["true_rul"]).abs()
    df["alert"] = df["risk_level"].apply(lambda x: "CRITICAL" in x or "HIGH" in x)

    # KPI Cards
    st.markdown("### Key Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    n_critical = len(df[df["risk_level"].str.contains("CRITICAL")])
    n_high = len(df[df["risk_level"].str.contains("HIGH")])
    n_medium = len(df[df["risk_level"].str.contains("MEDIUM")])
    n_low = len(df[df["risk_level"].str.contains("LOW")])
    n_alerts = n_critical + n_high

    kpi1.metric("Total Engines", len(df))
    kpi2.metric("🔴 Critical", n_critical)
    kpi3.metric("🟠 High Risk", n_high)
    kpi4.metric("🟡 Medium", n_medium)
    kpi5.metric("🟢 Low Risk", n_low)

    if n_alerts > 0:
        st.error(f"⚠️ **{n_alerts} engines require immediate attention!**")

    st.markdown("---")

    # Two-column layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("### Predicted RUL Distribution")
        chart_df = df[["unit", "ensemble_pred"]].set_index("unit").sort_values("ensemble_pred")
        st.bar_chart(chart_df, height=300)

    with right_col:
        st.markdown("### Risk Distribution")
        risk_counts = pd.DataFrame({
            "Risk Level": ["Critical", "High", "Medium", "Low"],
            "Count": [n_critical, n_high, n_medium, n_low]
        }).set_index("Risk Level")
        st.bar_chart(risk_counts, height=300)

    st.markdown("---")

    # Prediction accuracy
    st.markdown("### Prediction Accuracy")
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    acc_col1.metric("Mean Absolute Error", f"{df['error'].mean():.1f} cycles")
    acc_col2.metric("Median Error", f"{df['error'].median():.1f} cycles")
    acc_col3.metric("Max Error", f"{df['error'].max():.1f} cycles")

    st.markdown("---")

    # Fleet table
    st.markdown("### Fleet Status Table")
    filter_risk = st.multiselect(
        "Filter by risk level",
        ["🔴 CRITICAL", "🟠 HIGH", "🟡 MEDIUM", "🟢 LOW"],
        default=["🔴 CRITICAL", "🟠 HIGH", "🟡 MEDIUM", "🟢 LOW"]
    )

    filtered = df[df["risk_level"].isin(filter_risk)]
    display_df = filtered[["unit", "ensemble_pred", "true_rul", "error", "risk_level", "alert"]].copy()
    display_df.columns = ["Engine", "Predicted RUL", "True RUL", "Error", "Risk Level", "Alert"]
    display_df = display_df.sort_values("Predicted RUL")

    st.dataframe(display_df, use_container_width=True, height=400)
