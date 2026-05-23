"""
Per-engine detail page for IndustrialSentinel dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Engine Detail", page_icon="🔧", layout="wide")
st.title("🔧 Engine Detail")

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
    engine_id = st.sidebar.selectbox("Select Engine", sorted(feat_df["unit"].unique()))

    engine_data = feat_df[feat_df["unit"] == engine_id]
    engine_pred = pred_df[pred_df["unit"] == engine_id].iloc[0]

    # RUL gauge
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL", f"{engine_pred['ensemble_pred']:.1f} cycles")
    col2.metric("True RUL", f"{engine_pred['true_rul']:.1f} cycles")
    error = engine_pred['ensemble_pred'] - engine_pred['true_rul']
    col3.metric("Prediction Error", f"{error:.1f} cycles")

    # Model breakdown
    st.subheader("Model Predictions")
    model_col1, model_col2, model_col3 = st.columns(3)
    model_col1.metric("XGBoost", f"{engine_pred['xgb_pred']:.1f}")
    model_col2.metric("LSTM", f"{engine_pred['lstm_pred']:.1f}")
    model_col3.metric("Ensemble", f"{engine_pred['ensemble_pred']:.1f}")

    # Sensor plots
    st.subheader("Sensor Telemetry")
    available_sensors = [s for s in USEFUL_SENSORS if s in engine_data.columns]
    selected_sensors = st.multiselect("Select sensors to plot", available_sensors, default=available_sensors[:4])

    if selected_sensors:
        chart_data = engine_data[["cycle"] + selected_sensors].set_index("cycle")
        st.line_chart(chart_data)

    # RUL trend
    if "RUL" in engine_data.columns:
        st.subheader("RUL Over Time")
        rul_data = engine_data[["cycle", "RUL"]].set_index("cycle")
        st.line_chart(rul_data)
else:
    st.error("Data not found. Run training first.")
