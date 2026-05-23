"""
XGBoost regressor wrapper for RUL prediction.
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor

from src.config import (
    XGB_PARAMS, XGB_EARLY_STOPPING_ROUNDS, XGB_VAL_FRACTION,
    MODEL_DIR, XGBOOST_MODEL_FILE, RANDOM_SEED
)

logger = logging.getLogger(__name__)


class XGBoostRULModel:
    """XGBoost regressor for Remaining Useful Life prediction."""

    def __init__(self):
        """Initialize XGBoost model with configured hyperparameters."""
        params = XGB_PARAMS.copy()
        params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
        self.model = XGBRegressor(**params)
        self.val_engines = None

    def fit(self, df: pd.DataFrame, feature_cols: list, target_col: str = "RUL") -> dict:
        """
        Train XGBoost with engine-level validation split and early stopping.

        Args:
            df: Training DataFrame with 'unit' column.
            feature_cols: List of feature column names.
            target_col: Target column name.

        Returns:
            Dictionary with training info including validation engines.
        """
        # Engine-level split to prevent data leakage
        engines = df["unit"].unique()
        np.random.seed(RANDOM_SEED)
        n_val = max(1, int(len(engines) * XGB_VAL_FRACTION))
        val_engines = np.random.choice(engines, size=n_val, replace=False)
        train_engines = [e for e in engines if e not in val_engines]

        self.val_engines = sorted(val_engines.tolist())
        logger.info(f"XGBoost split: {len(train_engines)} train, {len(val_engines)} val engines")
        logger.info(f"Validation engines: {self.val_engines}")

        train_mask = df["unit"].isin(train_engines)
        val_mask = df["unit"].isin(val_engines)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, target_col]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_pred - y_val.values) ** 2))
        logger.info(f"XGBoost validation RMSE: {val_rmse:.4f}")

        return {
            "val_engines": self.val_engines,
            "val_rmse": val_rmse,
            "best_iteration": self.model.best_iteration,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL values.

        Args:
            X: Feature DataFrame or array.

        Returns:
            Array of predicted RUL values.
        """
        return self.model.predict(X)

    def save(self, path: Path = None) -> None:
        """
        Save model to disk.

        Args:
            path: Save path. Defaults to MODEL_DIR/XGBOOST_MODEL_FILE.
        """
        save_path = path or (MODEL_DIR / XGBOOST_MODEL_FILE)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"XGBoost model saved to {save_path}")

    def load(self, path: Path = None) -> None:
        """
        Load model from disk.

        Args:
            path: Load path. Defaults to MODEL_DIR/XGBOOST_MODEL_FILE.
        """
        load_path = path or (MODEL_DIR / XGBOOST_MODEL_FILE)
        self.model = joblib.load(load_path)
        logger.info(f"XGBoost model loaded from {load_path}")

    def get_feature_importance(self, feature_cols: list) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            feature_cols: List of feature column names.

        Returns:
            DataFrame with feature names and importance scores, sorted descending.
        """
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance
        }).sort_values("importance", ascending=False)
        return fi_df
