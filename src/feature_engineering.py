"""
Feature engineering for IndustrialSentinel.
Rolling features, lag features, normalization, and sequence creation.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    USEFUL_SENSORS, ROLLING_WINDOWS, LAG_STEPS, SEQUENCE_LENGTH,
    MODEL_DIR, SCALER_FILE, FEATURE_COLS_FILE
)
from src.validator import validate_no_nan_inf

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, fit_scaler: bool = False,
                      scaler: MinMaxScaler = None) -> tuple:
    """
    Compute rolling and lag features for useful sensors, then normalize.

    Args:
        df: DataFrame with 'unit', 'cycle', and sensor columns.
        fit_scaler: If True, fit a new scaler on this data.
        scaler: Pre-fitted scaler to use. Required if fit_scaler=False.

    Returns:
        Tuple of (featured_df, scaler, feature_columns).
    """
    feature_cols = []
    all_new_cols = {}

    for sensor in USEFUL_SENSORS:
        # Raw sensor value
        all_new_cols[sensor] = df[sensor].values
        feature_cols.append(sensor)

        # Rolling features per engine
        grouped = df.groupby("unit")[sensor]
        for window in ROLLING_WINDOWS:
            col_mean = f"{sensor}_roll_mean_{window}"
            col_std = f"{sensor}_roll_std_{window}"
            all_new_cols[col_mean] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            ).values
            all_new_cols[col_std] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).fillna(0).values
            feature_cols.extend([col_mean, col_std])

        # Lag features per engine
        for lag in LAG_STEPS:
            col_lag = f"{sensor}_lag_{lag}"
            all_new_cols[col_lag] = grouped.shift(lag).values
            feature_cols.append(col_lag)

    # Build featured DataFrame at once
    featured = df[["unit", "cycle", "RUL"]].copy()
    feat_df = pd.DataFrame(all_new_cols, index=featured.index)
    featured = pd.concat([featured, feat_df], axis=1)

    # Fill NaN from lags with forward fill then backfill within each engine
    featured[feature_cols] = featured.groupby("unit")[feature_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Final fillna with 0 for any remaining edge cases
    featured[feature_cols] = featured[feature_cols].fillna(0)

    logger.info(f"Engineered {len(feature_cols)} features. Shape: {featured.shape}")

    # Normalize
    if fit_scaler:
        scaler = MinMaxScaler()
        featured[feature_cols] = scaler.fit_transform(featured[feature_cols])
        # Save scaler and feature columns
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODEL_DIR / SCALER_FILE)
        joblib.dump(feature_cols, MODEL_DIR / FEATURE_COLS_FILE)
        logger.info(f"Scaler fitted and saved. Features: {len(feature_cols)}")
    else:
        if scaler is None:
            raise ValueError("Must provide a scaler when fit_scaler=False")
        featured[feature_cols] = scaler.transform(featured[feature_cols])

    validate_no_nan_inf(featured[feature_cols], "Feature matrix")
    return featured, scaler, feature_cols


def create_sequences(df: pd.DataFrame, feature_cols: list,
                     target_col: str = "RUL") -> tuple:
    """
    Create LSTM input sequences of shape (N, SEQUENCE_LENGTH, num_features).
    Sequences are created per engine, respecting temporal order.

    Args:
        df: Featured DataFrame with 'unit' column.
        feature_cols: List of feature column names.
        target_col: Target column name.

    Returns:
        Tuple of (X_sequences, y_targets) as numpy arrays.
    """
    sequences = []
    targets = []

    for unit_id, group in df.groupby("unit"):
        data = group[feature_cols].values
        rul = group[target_col].values

        for i in range(SEQUENCE_LENGTH, len(data) + 1):
            sequences.append(data[i - SEQUENCE_LENGTH:i])
            targets.append(rul[i - 1])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
    return X, y


def create_test_sequences(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Create test sequences: one per engine (last SEQUENCE_LENGTH cycles).

    Args:
        df: Featured test DataFrame.
        feature_cols: List of feature column names.

    Returns:
        Tuple of (X_sequences, y_true_rul) - one sequence per engine.
    """
    sequences = []
    targets = []

    for unit_id, group in df.groupby("unit"):
        data = group[feature_cols].values
        rul = group["RUL"].values

        if len(data) >= SEQUENCE_LENGTH:
            sequences.append(data[-SEQUENCE_LENGTH:])
        else:
            # Pad with zeros at the beginning
            pad = np.zeros((SEQUENCE_LENGTH - len(data), len(feature_cols)), dtype=np.float32)
            sequences.append(np.vstack([pad, data]))
        targets.append(rul[-1])  # True RUL at last cycle

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    logger.info(f"Created test sequences: X={X.shape}, y={y.shape}")
    return X, y


def create_autoencoder_sequences(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Create sequences from the first HEALTHY_FRACTION of each engine's life.
    Used for training the autoencoder on healthy-only data.

    Args:
        df: Featured training DataFrame.
        feature_cols: List of feature column names.

    Returns:
        Array of healthy sequences.
    """
    from src.config import AE_HEALTHY_FRACTION

    sequences = []
    total_cycles = 0
    healthy_cycles = 0

    for unit_id, group in df.groupby("unit"):
        data = group[feature_cols].values
        n_healthy = int(len(data) * AE_HEALTHY_FRACTION)
        healthy_data = data[:n_healthy]
        total_cycles += len(data)
        healthy_cycles += n_healthy

        for i in range(SEQUENCE_LENGTH, len(healthy_data) + 1):
            sequences.append(healthy_data[i - SEQUENCE_LENGTH:i])

    X = np.array(sequences, dtype=np.float32)
    logger.info(
        f"Autoencoder sequences: {X.shape} from first {AE_HEALTHY_FRACTION*100:.0f}% "
        f"of engine life ({healthy_cycles}/{total_cycles} cycles)"
    )

    # Assert healthy fraction constraint
    assert healthy_cycles / total_cycles <= AE_HEALTHY_FRACTION + 0.01, \
        "Healthy fraction constraint violated"

    return X
