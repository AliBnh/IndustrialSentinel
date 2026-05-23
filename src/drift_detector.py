"""
Sensor drift detection using Kolmogorov-Smirnov test.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from pathlib import Path

from src.config import USEFUL_SENSORS, DRIFT_P_VALUE_THRESHOLD, MODEL_DIR, TRAIN_STATS_FILE

logger = logging.getLogger(__name__)


def save_training_distributions(train_df: pd.DataFrame) -> None:
    """
    Save training data sensor distributions for later drift comparison.

    Args:
        train_df: Training DataFrame with sensor columns.
    """
    distributions = {}
    for sensor in USEFUL_SENSORS:
        distributions[sensor] = train_df[sensor].values
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(distributions, MODEL_DIR / TRAIN_STATS_FILE)
    logger.info(f"Training distributions saved for {len(USEFUL_SENSORS)} sensors")


def detect_drift(new_data: pd.DataFrame, train_distributions: dict = None) -> dict:
    """
    Detect sensor drift using two-sample KS test against training distribution.

    Args:
        new_data: New data DataFrame with sensor columns.
        train_distributions: Dictionary of sensor -> training values.
            If None, loads from saved artifact.

    Returns:
        Dictionary with per-sensor drift report.
    """
    if train_distributions is None:
        train_distributions = joblib.load(MODEL_DIR / TRAIN_STATS_FILE)

    report = {}
    drifted_sensors = []

    for sensor in USEFUL_SENSORS:
        if sensor not in new_data.columns:
            continue

        train_values = train_distributions[sensor]
        new_values = new_data[sensor].values

        ks_stat, p_value = stats.ks_2samp(train_values, new_values)
        is_drifted = p_value < DRIFT_P_VALUE_THRESHOLD

        report[sensor] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": is_drifted,
        }

        if is_drifted:
            drifted_sensors.append(sensor)

    summary = {
        "sensors_checked": len(report),
        "sensors_drifted": len(drifted_sensors),
        "drifted_sensor_names": drifted_sensors,
        "per_sensor": report,
    }

    logger.info(
        f"Drift detection: {len(drifted_sensors)}/{len(report)} sensors drifted"
    )
    return summary
