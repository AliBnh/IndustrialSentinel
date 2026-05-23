"""
Health check endpoint for IndustrialSentinel API.
"""
import time
import numpy as np
import json
from datetime import datetime
from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_model_container
from api.schemas import APIResponse
from src.config import MODEL_DIR, METADATA_FILE, SEQUENCE_LENGTH

router = APIRouter()


@router.get("/health")
def health_check(models: ModelContainer = Depends(get_model_container)) -> dict:
    """
    Verify models are loaded and functional by running a tiny test inference.

    Returns:
        Health status with model info and test inference latency.
    """
    # Run test inference to confirm models work
    start = time.time()
    n_features = len(models.feature_cols)
    dummy_input = np.random.rand(1, SEQUENCE_LENGTH, n_features).astype(np.float32)
    _ = models.lstm_model.predict(dummy_input)
    latency_ms = (time.time() - start) * 1000

    # Get model version from metadata
    metadata_path = MODEL_DIR / METADATA_FILE
    version = "unknown"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            version = meta.get("training_date", "unknown")

    return {
        "status": "ok",
        "data": {
            "models_loaded": models.loaded,
            "test_inference_latency_ms": round(latency_ms, 2),
            "model_version": version,
        },
        "timestamp": datetime.utcnow().isoformat()
    }
