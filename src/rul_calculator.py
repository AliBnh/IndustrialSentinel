"""
RUL (Remaining Useful Life) computation and NASA scoring function.
"""
import logging
import numpy as np
import pandas as pd

from src.config import RUL_CLIP_VALUE
from src.validator import validate_rul_non_negative

logger = logging.getLogger(__name__)


def compute_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RUL for training data with piecewise linear clipping.
    RUL = max_cycle_for_engine - current_cycle, clipped at RUL_CLIP_VALUE.

    Args:
        train_df: Training DataFrame with 'unit' and 'cycle' columns.

    Returns:
        DataFrame with added 'RUL' column.
    """
    df = train_df.copy()
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP_VALUE)
    df.drop("max_cycle", axis=1, inplace=True)

    validate_rul_non_negative(df["RUL"].values)
    logger.info(f"Train RUL computed. Range: [{df['RUL'].min()}, {df['RUL'].max()}]")
    return df


def compute_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct full RUL labels for all test cycles.
    The RUL file gives the true RUL at the last recorded cycle for each engine.

    Args:
        test_df: Test DataFrame with 'unit' and 'cycle' columns.
        rul_df: DataFrame with one RUL value per test engine.

    Returns:
        DataFrame with added 'RUL' column.
    """
    df = test_df.copy()
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]

    # Map RUL values to engines (engines are numbered 1..N)
    rul_df = rul_df.copy()
    rul_df["unit"] = range(1, len(rul_df) + 1)
    rul_df.columns = ["rul_at_end", "unit"]

    df = df.merge(max_cycles, on="unit")
    df = df.merge(rul_df, on="unit")
    df["RUL"] = df["rul_at_end"] + (df["max_cycle"] - df["cycle"])
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP_VALUE)
    df.drop(["max_cycle", "rul_at_end"], axis=1, inplace=True)

    validate_rul_non_negative(df["RUL"].values)
    logger.info(f"Test RUL computed. Range: [{df['RUL'].min()}, {df['RUL'].max()}]")
    return df


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute NASA asymmetric scoring function.
    Late predictions (under-estimate RUL) are penalized more heavily.

    s_i = exp(-d_i/13) - 1  if d_i < 0 (early prediction)
    s_i = exp(d_i/10) - 1   if d_i >= 0 (late prediction)

    where d_i = predicted - true (positive means predicted > true, i.e. late)

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Total NASA score (lower is better).
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))
