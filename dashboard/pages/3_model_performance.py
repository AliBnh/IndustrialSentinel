"""
Model performance page for IndustrialSentinel dashboard.
"""
import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")


@st.cache_data(ttl=60)
def load_metrics():
    """Load training metrics from metadata.json."""
    meta_path = f"{MODEL_DIR}/metadata.json"
    try:
        with open(meta_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_predictions():
    """Load test predictions."""
    pred_path = f"{DATA_DIR}/processed/test_predictions.csv"
    try:
        return pd.read_csv(pred_path)
    except FileNotFoundError:
        return None


metadata = load_metrics()
predictions = load_predictions()

if metadata:
    st.subheader("Training Metrics Comparison")

    metrics_data = {
        "Metric": ["RMSE", "MAE", "NASA Score", "Critical Zone RMSE",
                   "Detection Lead Time", "False Alert Rate"],
        "XGBoost": [
            metadata["xgb_metrics"]["rmse"],
            metadata["xgb_metrics"]["mae"],
            metadata["xgb_metrics"]["nasa_score"],
            metadata["xgb_metrics"]["critical_zone_rmse"],
            metadata["xgb_metrics"]["detection_lead_time"],
            metadata["xgb_metrics"]["false_alert_rate"],
        ],
        "LSTM": [
            metadata["lstm_metrics"]["rmse"],
            metadata["lstm_metrics"]["mae"],
            metadata["lstm_metrics"]["nasa_score"],
            metadata["lstm_metrics"]["critical_zone_rmse"],
            metadata["lstm_metrics"]["detection_lead_time"],
            metadata["lstm_metrics"]["false_alert_rate"],
        ],
        "Ensemble": [
            metadata["ensemble_metrics"]["rmse"],
            metadata["ensemble_metrics"]["mae"],
            metadata["ensemble_metrics"]["nasa_score"],
            metadata["ensemble_metrics"]["critical_zone_rmse"],
            metadata["ensemble_metrics"]["detection_lead_time"],
            metadata["ensemble_metrics"]["false_alert_rate"],
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Key metrics
    st.subheader("Ensemble Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metadata['rmse']:.2f}")
    col2.metric("MAE", f"{metadata['mae']:.2f}")
    col3.metric("NASA Score", f"{metadata['nasa_score']:.0f}")

    # Predicted vs True scatter
    if predictions is not None:
        st.subheader("Predicted vs True RUL (Ensemble)")
        scatter_data = predictions[["true_rul", "ensemble_pred"]].copy()
        scatter_data.columns = ["True RUL", "Predicted RUL"]
        st.scatter_chart(scatter_data, x="True RUL", y="Predicted RUL")

    # Training info
    st.subheader("Training Details")
    st.json({
        "training_date": metadata["training_date"],
        "n_features": metadata["n_features"],
        "hyperparameters": metadata["hyperparameters"],
    })
else:
    st.error("Metadata not found. Run training first.")
