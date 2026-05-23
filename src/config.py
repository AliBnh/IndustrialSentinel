"""
Central configuration for IndustrialSentinel.
Single source of truth for all constants used across the project.
"""
import os
from pathlib import Path

# --- Directory Paths ---
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
CONFIG_DIR = PROJECT_ROOT / "config"

# --- Dataset Files ---
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

# --- Column Definitions ---
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["os1", "os2", "os3"]
SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# --- Sensor Selection ---
LOW_VARIANCE_SENSORS = ["sensor1", "sensor5", "sensor6", "sensor10", "sensor16", "sensor18", "sensor19"]
USEFUL_SENSORS = [s for s in SENSOR_COLS if s not in LOW_VARIANCE_SENSORS]
VARIANCE_THRESHOLD = 0.1  # Sensors below this variance are dropped

# --- RUL Configuration ---
RUL_CLIP_VALUE = 125  # Piecewise linear clipping ceiling

# --- Feature Engineering ---
ROLLING_WINDOWS = [5, 10, 15]
LAG_STEPS = [1, 3]
SEQUENCE_LENGTH = 30  # LSTM input window size

# --- XGBoost Hyperparameters ---
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}
XGB_EARLY_STOPPING_ROUNDS = 30
XGB_VAL_FRACTION = 0.15  # Fraction of engines for validation

# --- LSTM Hyperparameters ---
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_GRAD_CLIP = 1.0
LSTM_PATIENCE = 15  # Early stopping patience
LSTM_VAL_FRACTION = 0.15

# --- Autoencoder Hyperparameters ---
AE_HIDDEN_SIZE = 32
AE_NUM_LAYERS = 1
AE_DROPOUT = 0.2
AE_LEARNING_RATE = 0.001
AE_BATCH_SIZE = 64
AE_EPOCHS = 60
AE_HEALTHY_FRACTION = 0.30  # First 30% of engine life = healthy
AE_THRESHOLD_PERCENTILE = 95  # 95th percentile of healthy reconstruction errors

# --- Ensemble ---
ENSEMBLE_WEIGHTS = {"xgboost": 0.3, "lstm": 0.7}

# --- Risk Thresholds ---
RISK_THRESHOLDS = {
    "CRITICAL": 15,   # RUL < 15
    "HIGH": 30,       # RUL < 30
    "MEDIUM": 60,     # RUL < 60
    "LOW": float("inf"),  # RUL >= 60
}

# --- Anomaly Thresholds for Alerts ---
ANOMALY_ALERT_THRESHOLD = 0.8  # Normalized anomaly score above this triggers alert

# --- API Configuration ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "IndustrialSentinel-RUL"
MLFLOW_MODEL_NAME = "IndustrialSentinel-RUL"

# --- Drift Detection ---
DRIFT_P_VALUE_THRESHOLD = 0.05

# --- Model Artifact Filenames ---
XGBOOST_MODEL_FILE = "xgboost_rul.pkl"
LSTM_MODEL_FILE = "lstm_rul.pt"
AUTOENCODER_MODEL_FILE = "autoencoder.pt"
AE_THRESHOLD_FILE = "ae_threshold.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_COLS_FILE = "feature_cols.pkl"
METADATA_FILE = "metadata.json"
TRAIN_STATS_FILE = "train_stats.pkl"

# --- Random Seed ---
RANDOM_SEED = 42
