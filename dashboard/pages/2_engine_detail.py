"""
Per-engine detail page for IndustrialSentinel dashboard.
Deep dive into individual engine health, sensor telemetry, and predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Engine Detail", page_icon="🔧", layout="wide")
st.title("🔧 Engine Detail Analysis")
st.markdown("Deep dive into individual engine health and predictions")
st.markdown("---")

DATA_DIR = os.environ.get("DATA_DIR", "data")
USEFUL_SENSORS = [f"sensor{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]


@st.cache_data(ttl=60)
def load_data():
    """Load test featured data and predictions."""
    feat_path = f"{DATA_DIR}/processed/test_featured.parquet"
    pred_path = f"{DATA_DIR}/processed/test_predictions.csv"
    try:
        feat = pd.read_parquet(feat_path)
        pred = pd.read_csv(pred_path)
        return feat, pred
    except FileNotFoundError:
        return None, None


feat_df, pred_df = load_data()

if feat_df is not None and pred_df is not None:
    # Engine selector in sidebar
    engine_id = st.sidebar.selectbox(
        "🔧 Select Engine",
        sorted(feat_df["unit"].unique()),
        index=0
    )

    engine_data = feat_df[feat_df["unit"] == engine_id]
    engine_pred = pred_df[pred_df["unit"] == engine_id].iloc[0]

    # Risk level determination
    pred_rul = engine_pred['ensemble_pred']
    if pred_rul < 15:
        risk = "🔴 CRITICAL"
        risk_color = "red"
    elif pred_rul < 30:
        risk = "🟠 HIGH"
        risk_color = "orange"
    elif pred_rul < 60:
        risk = "🟡 MEDIUM"
        risk_color = "yellow"
    else:
        risk = "🟢 LOW"
        risk_color = "green"

    # Header with risk badge
    st.markdown(f"## Engine #{engine_id} — {risk}")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
    col2.metric("True RUL", f"{engine_pred['true_rul']:.1f} cycles")
    error = pred_rul - engine_pred['true_rul']
    col3.metric("Prediction Error", f"{error:+.1f} cycles")
    col4.metric("Total Cycles", f"{len(engine_data)}")

    st.markdown("---")

    # Model breakdown
    st.markdown("### Model Predictions Breakdown")
    model_col1, model_col2, model_col3 = st.columns(3)
    model_col1.metric("XGBoost", f"{engine_pred['xgb_pred']:.1f} cycles",
                      delta=f"{engine_pred['xgb_pred'] - engine_pred['true_rul']:+.1f}")
    model_col2.metric("LSTM", f"{engine_pred['lstm_pred']:.1f} cycles",
                      delta=f"{engine_pred['lstm_pred'] - engine_pred['true_rul']:+.1f}")
    model_col3.metric("Ensemble (0.3×XGB + 0.7×LSTM)", f"{pred_rul:.1f} cycles",
                      delta=f"{pred_rul - engine_pred['true_rul']:+.1f}")

    st.markdown("---")

    # Sensor telemetry
    st.markdown("### Sensor Telemetry")
    available_sensors = [s for s in USEFUL_SENSORS if s in engine_data.columns]

    sensor_tab1, sensor_tab2 = st.tabs(["📈 Individual Sensors", "📊 All Sensors"])

    with sensor_tab1:
        selected_sensors = st.multiselect(
            "Select sensors to plot",
            available_sensors,
            default=available_sensors[:4]
        )
        if selected_sensors:
            chart_data = engine_data[["cycle"] + selected_sensors].set_index("cycle")
            st.line_chart(chart_data, height=350)

    with sensor_tab2:
        if available_sensors:
            # Normalized view of all sensors
            norm_data = engine_data[["cycle"] + available_sensors].set_index("cycle")
            norm_data = (norm_data - norm_data.min()) / (norm_data.max() - norm_data.min() + 1e-8)
            st.line_chart(norm_data, height=350)
            st.caption("All sensors normalized to [0,1] for comparison")

    st.markdown("---")

    # RUL degradation curve
    if "RUL" in engine_data.columns:
        st.markdown("### RUL Degradation Curve")
        rul_data = engine_data[["cycle", "RUL"]].set_index("cycle")
        st.area_chart(rul_data, height=250)
        st.caption("True RUL decreasing over engine operational life")

    # Recommendation card
    st.markdown("---")
    st.markdown("### Maintenance Recommendation")
    if "CRITICAL" in risk:
        st.error(f"🚨 **IMMEDIATE ACTION REQUIRED** — Engine #{engine_id} has {pred_rul:.0f} cycles remaining. Schedule emergency maintenance within 24 hours. Ground this engine immediately.")
    elif "HIGH" in risk:
        st.warning(f"⚠️ **Schedule maintenance soon** — Engine #{engine_id} has {pred_rul:.0f} cycles remaining. Plan maintenance within 1 week.")
    elif "MEDIUM" in risk:
        st.info(f"📋 **Monitor closely** — Engine #{engine_id} has {pred_rul:.0f} cycles remaining. Schedule routine inspection at next opportunity.")
    else:
        st.success(f"✅ **Normal operation** — Engine #{engine_id} has {pred_rul:.0f} cycles remaining. Continue standard monitoring schedule.")

else:
    st.error("⚠️ Data not found. Run training first.")
