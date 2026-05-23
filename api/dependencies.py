"""
FastAPI dependency injection for model loading.
Models are loaded once at startup and injected into route handlers.
"""
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

from src.config import MODEL_DIR, SCALER_FILE, FEATURE_COLS_FILE, TRAIN_STATS_FILE
from src.models.xgboost_model import XGBoostRULModel
from src.models.lstm_model import LSTMRULModel
from src.models.autoencoder_model import AutoencoderAnomalyModel
from src.validator import validate_model_artifacts_exist

logger = logging.getLogger("api.dependencies")


class ModelContainer:
    """Container holding all loaded models and artifacts."""

    def __init__(self):
        """Initialize empty container."""
        self.xgb_model = None
        self.lstm_model = None
        self.ae_model = None
        self.scaler = None
        self.feature_cols = None
        self.train_stats = None
        self.loaded = False

    def load_all(self) -> None:
        """
        Load all model artifacts from disk.

        Raises:
            ValueError: If artifacts are missing.
        """
        validate_model_artifacts_exist()

        self.scaler = joblib.load(MODEL_DIR / SCALER_FILE)
        self.feature_cols = joblib.load(MODEL_DIR / FEATURE_COLS_FILE)
        self.train_stats = joblib.load(MODEL_DIR / TRAIN_STATS_FILE)

        self.xgb_model = XGBoostRULModel()
        self.xgb_model.load()

        n_features = len(self.feature_cols)
        self.lstm_model = LSTMRULModel(input_size=n_features)
        self.lstm_model.load()

        self.ae_model = AutoencoderAnomalyModel(input_size=n_features)
        self.ae_model.load()

        self.loaded = True
        logger.info("All models loaded successfully")


# Singleton instance
_container = None


def get_model_container() -> ModelContainer:
    """
    Get the singleton model container, loading models on first call.

    Returns:
        ModelContainer with all models loaded.
    """
    global _container
    if _container is None:
        _container = ModelContainer()
        _container.load_all()
    return _container
