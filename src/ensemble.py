"""
Weighted ensemble combiner for RUL predictions.
"""
import logging
import numpy as np

from src.config import ENSEMBLE_WEIGHTS

logger = logging.getLogger(__name__)


def ensemble_predict(xgb_pred: np.ndarray, lstm_pred: np.ndarray,
                     weights: dict = None) -> np.ndarray:
    """
    Combine XGBoost and LSTM predictions using weighted average.

    Args:
        xgb_pred: XGBoost predictions.
        lstm_pred: LSTM predictions.
        weights: Dictionary with 'xgboost' and 'lstm' weights. Defaults to config.

    Returns:
        Ensemble predictions.
    """
    w = weights or ENSEMBLE_WEIGHTS
    combined = w["xgboost"] * xgb_pred + w["lstm"] * lstm_pred
    # Clip to non-negative
    combined = np.clip(combined, 0, None)
    logger.info(
        f"Ensemble predictions: mean={combined.mean():.2f}, "
        f"std={combined.std():.2f}, range=[{combined.min():.2f}, {combined.max():.2f}]"
    )
    return combined
