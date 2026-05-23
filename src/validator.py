"""
Data integrity assertions used throughout the IndustrialSentinel pipeline.
Each function raises a descriptive ValueError on failure.
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR, ALL_COLS, TRAIN_FILE, TEST_FILE, RUL_FILE, MODEL_DIR,
    XGBOOST_MODEL_FILE, LSTM_MODEL_FILE, AUTOENCODER_MODEL_FILE,
    AE_THRESHOLD_FILE, SCALER_FILE, FEATURE_COLS_FILE, SEQUENCE_LENGTH
)

logger = logging.getLogger(__name__)


def validate_raw_files(data_dir: str = None) -> None:
    """
    Validate that raw dataset files exist and have the expected schema.

    Args:
        data_dir: Path to directory containing raw files. Defaults to RAW_DATA_DIR.

    Raises:
        ValueError: If any validation fails.
    """
    raw_dir = Path(data_dir) / "raw" if data_dir else RAW_DATA_DIR

    for fname in [TRAIN_FILE, TEST_FILE, RUL_FILE]:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise ValueError(f"Required file not found: {fpath}")

    # Validate column count for train and test
    train_path = raw_dir / TRAIN_FILE
    test_path = raw_dir / TEST_FILE
    rul_path = raw_dir / RUL_FILE

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    if train_df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Train file has {train_df.shape[1]} columns, expected {len(ALL_COLS)}"
        )
    if train_df.isnull().any().any():
        raise ValueError("Train file contains null values")

    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)
    if test_df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Test file has {test_df.shape[1]} columns, expected {len(ALL_COLS)}"
        )

    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None)
    n_test_engines = test_df.iloc[:, 0].nunique()
    if rul_df.shape[0] != n_test_engines:
        raise ValueError(
            f"RUL file has {rul_df.shape[0]} rows but test has {n_test_engines} engines"
        )

    logger.info("All validations passed")
    print("All validations passed")


def validate_rul_non_negative(rul_values: np.ndarray) -> None:
    """
    Assert that all RUL values are non-negative.

    Args:
        rul_values: Array of RUL values.

    Raises:
        ValueError: If any RUL value is negative.
    """
    if np.any(rul_values < 0):
        neg_count = np.sum(rul_values < 0)
        raise ValueError(f"Found {neg_count} negative RUL values. Min: {rul_values.min()}")


def validate_no_nan_inf(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Assert that a DataFrame contains no NaN or Inf values.

    Args:
        df: DataFrame to validate.
        name: Name for error messages.

    Raises:
        ValueError: If NaN or Inf values are found.
    """
    if isinstance(df, np.ndarray):
        if np.any(np.isnan(df)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(df)):
            raise ValueError(f"{name} contains Inf values")
    else:
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"{name} contains NaN values in columns: {nan_cols}")
        numeric_df = df.select_dtypes(include=[np.number])
        if np.any(np.isinf(numeric_df.values)):
            raise ValueError(f"{name} contains Inf values")


def validate_sequence_shape(sequences: np.ndarray, expected_features: int) -> None:
    """
    Validate that sequence tensor has the expected shape.

    Args:
        sequences: Array of shape (N, window, features).
        expected_features: Expected number of features.

    Raises:
        ValueError: If shape doesn't match expectations.
    """
    if len(sequences.shape) != 3:
        raise ValueError(f"Sequences must be 3D, got shape {sequences.shape}")
    if sequences.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence window is {sequences.shape[1]}, expected {SEQUENCE_LENGTH}"
        )
    if sequences.shape[2] != expected_features:
        raise ValueError(
            f"Sequence features is {sequences.shape[2]}, expected {expected_features}"
        )


def validate_model_artifacts_exist(model_dir: Path = None) -> None:
    """
    Validate that all required model artifact files exist.

    Args:
        model_dir: Path to model directory. Defaults to MODEL_DIR.

    Raises:
        ValueError: If any artifact is missing.
    """
    mdir = model_dir or MODEL_DIR
    required_files = [
        XGBOOST_MODEL_FILE, LSTM_MODEL_FILE, AUTOENCODER_MODEL_FILE,
        AE_THRESHOLD_FILE, SCALER_FILE, FEATURE_COLS_FILE
    ]
    missing = [f for f in required_files if not (mdir / f).exists()]
    if missing:
        raise ValueError(f"Missing model artifacts in {mdir}: {missing}")


def validate_test_engine_count(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> None:
    """
    Validate that test engine count matches RUL file row count.

    Args:
        test_df: Test DataFrame with 'unit' column.
        rul_df: RUL DataFrame.

    Raises:
        ValueError: If counts don't match.
    """
    n_engines = test_df["unit"].nunique()
    n_rul = len(rul_df)
    if n_engines != n_rul:
        raise ValueError(
            f"Test has {n_engines} engines but RUL file has {n_rul} entries"
        )
