"""Shared test fixtures for IndustrialSentinel."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import ALL_COLS, USEFUL_SENSORS, RAW_DATA_DIR, SENSOR_COLS


@pytest.fixture
def sample_train_df():
    """Small synthetic training DataFrame for unit tests."""
    np.random.seed(42)
    rows = []
    for unit in range(1, 4):
        n_cycles = np.random.randint(50, 80)
        for cycle in range(1, n_cycles + 1):
            row = [unit, cycle, 0.0, 0.0, 100.0]
            row += [np.random.randn() * 10 + 500 for _ in range(21)]
            rows.append(row)
    return pd.DataFrame(rows, columns=ALL_COLS)


@pytest.fixture
def sample_rul_df():
    """Sample RUL DataFrame for 3 test engines."""
    return pd.DataFrame({"RUL": [50, 30, 10]})


@pytest.fixture
def raw_data_path():
    """Path to raw data directory."""
    return RAW_DATA_DIR
