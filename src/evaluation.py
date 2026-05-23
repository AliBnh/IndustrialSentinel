"""
Evaluation metrics for IndustrialSentinel RUL prediction.
"""
import logging
import numpy as np

from src.config import RISK_THRESHOLDS
from src.rul_calculator import nasa_score

logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_critical_zone_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                                threshold: int = 30) -> float:
    """
    Compute RMSE restricted to engines where true RUL < threshold.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        threshold: Critical zone threshold.

    Returns:
        Critical-zone RMSE.
    """
    mask = y_true < threshold
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_detection_lead_time(y_true: np.ndarray, y_pred: np.ndarray,
                                 alert_threshold: int = 50) -> float:
    """
    Compute mean detection lead time.
    For engines where model predicts RUL < alert_threshold, how many cycles
    before true failure does the alert fire on average.

    Args:
        y_true: True RUL values (at last cycle of each engine).
        y_pred: Predicted RUL values.
        alert_threshold: RUL threshold for triggering alert.

    Returns:
        Mean detection lead time in cycles.
    """
    # Engines where model fires alert (predicted RUL < threshold)
    alert_mask = y_pred < alert_threshold
    if alert_mask.sum() == 0:
        return 0.0
    # Lead time = true RUL at the point of alert
    lead_times = y_true[alert_mask]
    return float(np.mean(lead_times))


def compute_false_alert_rate(y_true: np.ndarray, y_pred: np.ndarray,
                              healthy_threshold: int = 60,
                              alert_threshold: int = 30) -> float:
    """
    Compute false alert rate: fraction of healthy engines that received alert.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        healthy_threshold: Engines with true RUL > this are considered healthy.
        alert_threshold: Predicted RUL below this triggers alert.

    Returns:
        False alert rate (0 to 1).
    """
    healthy_mask = y_true > healthy_threshold
    if healthy_mask.sum() == 0:
        return 0.0
    false_alerts = (y_pred[healthy_mask] < alert_threshold).sum()
    return float(false_alerts / healthy_mask.sum())


def get_risk_level(predicted_rul: float) -> str:
    """
    Determine risk level based on predicted RUL.

    Args:
        predicted_rul: Predicted remaining useful life.

    Returns:
        Risk level string: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    for level, threshold in RISK_THRESHOLDS.items():
        if predicted_rul < threshold:
            return level
    return "LOW"


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute all evaluation metrics and print formatted results.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        model_name: Name for display.

    Returns:
        Dictionary of all metrics.
    """
    metrics = {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "critical_zone_rmse": compute_critical_zone_rmse(y_true, y_pred),
        "detection_lead_time": compute_detection_lead_time(y_true, y_pred),
        "false_alert_rate": compute_false_alert_rate(y_true, y_pred),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation Results")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k:25s}: {v:.4f}")
    logger.info(f"{'='*50}\n")

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*50}\n")

    return metrics
