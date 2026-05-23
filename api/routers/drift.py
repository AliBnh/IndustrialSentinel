"""
Drift detection endpoint for IndustrialSentinel API.
"""
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import ModelContainer, get_model_container
from api.schemas import PredictRequest
from src.config import SENSOR_COLS, USEFUL_SENSORS
from src.drift_detector import detect_drift

router = APIRouter()


@router.post("/drift")
def check_drift(request: PredictRequest,
                models: ModelContainer = Depends(get_model_container)) -> dict:
    """
    Run drift detection on submitted engine data against training distribution.

    Args:
        request: PredictRequest with sensor readings.

    Returns:
        Per-sensor drift report with KS statistics.
    """
    try:
        # Build DataFrame from readings
        sensor_names = SENSOR_COLS[:21]
        df = pd.DataFrame(request.readings, columns=sensor_names)

        # Run drift detection
        report = detect_drift(df, models.train_stats)

        # Convert numpy types to native Python for JSON serialization
        import json

        def convert(obj):
            """Convert numpy types to native Python."""
            import numpy as np
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        report = convert(report)

        return {
            "status": "ok",
            "data": report,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
