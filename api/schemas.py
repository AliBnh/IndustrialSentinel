"""
Pydantic request/response models for IndustrialSentinel API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SensorReading(BaseModel):
    """A single cycle of sensor readings (21 sensors)."""
    values: List[float] = Field(..., min_length=21, max_length=21)


class PredictRequest(BaseModel):
    """Request payload for /predict endpoint."""
    unit: int = Field(..., ge=1, description="Engine unit ID")
    readings: List[List[float]] = Field(
        ..., min_length=1,
        description="List of cycle readings, each with 21 sensor values"
    )


class PredictionResponse(BaseModel):
    """Prediction result for a single engine."""
    predicted_rul: float
    anomaly_score: float
    risk_level: str
    alert: bool
    recommendation: str
    xgb_prediction: float
    lstm_prediction: float
    ensemble_prediction: float
    timestamp: str


class APIResponse(BaseModel):
    """Standard API response envelope."""
    status: str = "ok"
    data: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    test_inference_latency_ms: float
    model_version: str
    timestamp: str


class DriftReport(BaseModel):
    """Drift detection report."""
    sensors_checked: int
    sensors_drifted: int
    drifted_sensor_names: List[str]
    per_sensor: Dict[str, Dict[str, Any]]
