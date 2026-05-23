"""
Data loading and validation for NASA CMAPSS FD001 dataset.
"""
import logging
import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_DIR, ALL_COLS, TRAIN_FILE, TEST_FILE, RUL_FILE
from src.validator import validate_raw_files

logger = logging.getLogger(__name__)


def load_raw_data(data_dir: Path = None) -> tuple:
    """
    Load and validate raw CMAPSS FD001 dataset files.

    Args:
        data_dir: Path to raw data directory. Defaults to RAW_DATA_DIR.

    Returns:
        Tuple of (train_df, test_df, rul_df) DataFrames.

    Raises:
        ValueError: If validation fails.
        FileNotFoundError: If data files are missing.
    """
    raw_dir = data_dir or RAW_DATA_DIR

    # Validate files exist and have correct schema
    validate_raw_files(str(raw_dir.parent))

    # Load train
    train_df = pd.read_csv(raw_dir / TRAIN_FILE, sep=r'\s+', header=None, names=ALL_COLS)
    logger.info(f"Loaded train data: {train_df.shape}")

    # Load test
    test_df = pd.read_csv(raw_dir / TEST_FILE, sep=r'\s+', header=None, names=ALL_COLS)
    logger.info(f"Loaded test data: {test_df.shape}")

    # Load RUL
    rul_df = pd.read_csv(raw_dir / RUL_FILE, sep=r'\s+', header=None, names=["RUL"])
    logger.info(f"Loaded RUL data: {rul_df.shape}")

    return train_df, test_df, rul_df
