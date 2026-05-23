"""
Prediction endpoints for IndustrialSentinel API.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path

from api.dependencies import ModelContainer, get_model_container
from api.schemas import PredictRequest, APIResponse
from src.config import (
    USEFUL_SENSORS, SEQUENCE_LENGTH, ROLLING_WINDOWS, LAG_STEPS,
    ENSEMBLE_WEIGHTS, PROCESSED_DATA_DIR, RISK_THRESHOLDS,
    ANOMALY_ALERT_THRESHOLD, ALL_COLS, INDEX_COLS, SETTING_COLS, SENSOR_COLS
)
from src.ensemble import ensemble_predict
from src.evaluation import get_risk_level

router = APIRouter()


def _get_recommendation(risk_level: str, predicted_rul: float) -> str:
    """
    Generate maintenance recommendation based on risk level.

    Args:
        risk_level: CRITICAL, HIGH, MEDIUM, or LOW.
        predicted_rul: Predicted RUL value.

    Returns:
        Recommendation string.
    """
    recommendations = {
        "CRITICAL": f"IMMEDIATE ACTION REQUIRED. Predicted failure in {predicted_rul:.0f} cycles. Schedule emergency maintenance within 24 hours.",
        "HIGH": f"Schedule maintenance soon. Predicted failure in {predicted_rul:.0f} cycles. Plan maintenance within 1 week.",
        "MEDIUM": f"Monitor closely. Predicted failure in {predicted_rul:.0f} cycles. Schedule routine inspection.",
        "LOW": f"Normal operation. Predicted failure in {predicted_rul:.0f} cycles. Continue standard monitoring.",
    }
    return recommendations.get(risk_level, "Unknown risk level")


def _engineer_single_engine(readings: list, models: ModelContainer) -> tuple:
    """
    Run feature engineering on raw readings for a single engine.

    Args:
        readings: List of cycle readings (each with 21 sensor values).
        models: Model container with scaler and feature columns.

    Returns:
        Tuple of (xgb_features, lstm_sequence, ae_sequence).
    """
    # Build DataFrame from readings
    n_cycles = len(readings)
    sensor_names = SENSOR_COLS[:21]

    df = pd.DataFrame(readings, columns=sensor_names)
    df["unit"] = 1
    df["cycle"] = range(1, n_cycles + 1)

    feature_cols = models.feature_cols
    featured = pd.DataFrame()
    featured["unit"] = df["unit"]
    featured["cycle"] = df["cycle"]

    feat_data = {}
    for sensor in USEFUL_SENSORS:
        if sensor not in df.columns:
            continue
        feat_data[sensor] = df[sensor].values

        for window in ROLLING_WINDOWS:
            col_mean = f"{sensor}_roll_mean_{window}"
            col_std = f"{sensor}_roll_std_{window}"
            series = df[sensor]
            feat_data[col_mean] = series.rolling(window, min_periods=1).mean().values
            feat_data[col_std] = series.rolling(window, min_periods=1).std().fillna(0).values

        for lag in LAG_STEPS:
            col_lag = f"{sensor}_lag_{lag}"
            feat_data[col_lag] = df[sensor].shift(lag).bfill().values

    feat_df = pd.DataFrame(feat_data)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feat_df = feat_df[feature_cols]
    feat_df = feat_df.fillna(0)

    # Scale
    scaled = models.scaler.transform(feat_df.values)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols)

    # XGBoost: use last row
    xgb_features = scaled_df.iloc[-1:].values

    # LSTM: use last SEQUENCE_LENGTH rows
    if len(scaled_df) >= SEQUENCE_LENGTH:
        lstm_seq = scaled_df.iloc[-SEQUENCE_LENGTH:].values
    else:
        pad = np.zeros((SEQUENCE_LENGTH - len(scaled_df), len(feature_cols)))
        lstm_seq = np.vstack([pad, scaled_df.values])

    lstm_seq = lstm_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols)).astype(np.float32)
    return xgb_features, lstm_seq, lstm_seq


@router.post("/predict")
def predict(request: PredictRequest,
            models: ModelContainer = Depends(get_model_container)) -> dict:
    """
    Run full prediction pipeline on submitted engine telemetry.

    Args:
        request: PredictRequest with unit ID and sensor readings.

    Returns:
        Prediction response with RUL, risk level, anomaly score.
    """
    try:
        # Validate readings have 21 sensors each
        for i, reading in enumerate(request.readings):
            if len(reading) != 21:
                raise HTTPException(
                    status_code=422,
                    detail=f"Reading {i} has {len(reading)} values, expected 21 sensor values"
                )

        xgb_features, lstm_seq, ae_seq = _engineer_single_engine(
            request.readings, models
        )

        # Predictions
        xgb_pred = float(models.xgb_model.predict(
            pd.DataFrame(xgb_features, columns=models.feature_cols)
        )[0])
        lstm_pred = float(models.lstm_model.predict(lstm_seq)[0])
        ensemble_pred = float(
            ENSEMBLE_WEIGHTS["xgboost"] * xgb_pred + ENSEMBLE_WEIGHTS["lstm"] * lstm_pred
        )
        ensemble_pred = max(0, ensemble_pred)

        # Anomaly score
        anomaly_score = float(models.ae_model.predict_anomaly_score(ae_seq)[0])
        normalized_score = min(anomaly_score / (models.ae_model.threshold * 3), 1.0)

        risk_level = get_risk_level(ensemble_pred)
        alert = risk_level in ("HIGH", "CRITICAL") or normalized_score > ANOMALY_ALERT_THRESHOLD

        return {
            "status": "ok",
            "data": {
                "predicted_rul": round(ensemble_pred, 2),
                "anomaly_score": round(normalized_score, 4),
                "risk_level": risk_level,
                "alert": alert,
                "recommendation": _get_recommendation(risk_level, ensemble_pred),
                "xgb_prediction": round(xgb_pred, 2),
                "lstm_prediction": round(lstm_pred, 2),
                "ensemble_prediction": round(ensemble_pred, 2),
                "timestamp": datetime.utcnow().isoformat(),
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo/{engine_id}")
def demo_predict(engine_id: int,
                 models: ModelContainer = Depends(get_model_container)) -> dict:
    """
    Run prediction on a real test engine by ID.

    Args:
        engine_id: Engine unit number (1-100).

    Returns:
        Prediction response plus true RUL for comparison.
    """
    if engine_id < 1 or engine_id > 100:
        raise HTTPException(status_code=422, detail="Engine ID must be between 1 and 100")

    # Load test predictions
    pred_path = PROCESSED_DATA_DIR / "test_predictions.csv"
    if not pred_path.exists():
        raise HTTPException(status_code=500, detail="Test predictions not found. Run training first.")

    predictions = pd.read_csv(pred_path)
    engine_row = predictions[predictions["unit"] == engine_id]
    if engine_row.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")

    row = engine_row.iloc[0]
    ensemble_pred = float(row["ensemble_pred"])
    true_rul = float(row["true_rul"])
    risk_level = get_risk_level(ensemble_pred)

    # Get anomaly score from test data
    test_featured_path = PROCESSED_DATA_DIR / "test_featured.parquet"
    anomaly_score = 0.0
    if test_featured_path.exists():
        test_feat = pd.read_parquet(test_featured_path)
        engine_data = test_feat[test_feat["unit"] == engine_id]
        if len(engine_data) >= SEQUENCE_LENGTH:
            seq = engine_data[models.feature_cols].iloc[-SEQUENCE_LENGTH:].values
            seq = seq.reshape(1, SEQUENCE_LENGTH, len(models.feature_cols)).astype(np.float32)
            raw_score = float(models.ae_model.predict_anomaly_score(seq)[0])
            anomaly_score = min(raw_score / (models.ae_model.threshold * 3), 1.0)

    alert = risk_level in ("HIGH", "CRITICAL") or anomaly_score > ANOMALY_ALERT_THRESHOLD

    return {
        "status": "ok",
        "data": {
            "engine_id": engine_id,
            "predicted_rul": round(ensemble_pred, 2),
            "true_rul": round(true_rul, 2),
            "anomaly_score": round(anomaly_score, 4),
            "risk_level": risk_level,
            "alert": alert,
            "recommendation": _get_recommendation(risk_level, ensemble_pred),
            "xgb_prediction": round(float(row["xgb_pred"]), 2),
            "lstm_prediction": round(float(row["lstm_pred"]), 2),
            "ensemble_prediction": round(ensemble_pred, 2),
            "timestamp": datetime.utcnow().isoformat(),
        },
        "timestamp": datetime.utcnow().isoformat()
    }
