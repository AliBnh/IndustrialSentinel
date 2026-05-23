"""Unit tests for IndustrialSentinel core modules."""
import numpy as np
import pandas as pd
import pytest

from src.config import RUL_CLIP_VALUE, USEFUL_SENSORS, ROLLING_WINDOWS, LAG_STEPS
from src.rul_calculator import compute_train_rul, nasa_score
from src.feature_engineering import engineer_features
from src.evaluation import compute_rmse, compute_mae, compute_critical_zone_rmse, get_risk_level
from src.validator import validate_rul_non_negative, validate_no_nan_inf


class TestRULCalculation:
    """Tests for RUL computation logic."""

    def test_rul_clipping(self, sample_train_df):
        """RUL values should be clipped at RUL_CLIP_VALUE."""
        result = compute_train_rul(sample_train_df)
        assert result["RUL"].max() <= RUL_CLIP_VALUE

    def test_rul_non_negative(self, sample_train_df):
        """All RUL values must be non-negative."""
        result = compute_train_rul(sample_train_df)
        assert (result["RUL"] >= 0).all()

    def test_rul_last_cycle_is_zero(self, sample_train_df):
        """The last cycle of each engine should have RUL=0."""
        result = compute_train_rul(sample_train_df)
        for unit_id, group in result.groupby("unit"):
            assert group["RUL"].iloc[-1] == 0

    def test_rul_monotonically_decreasing(self, sample_train_df):
        """RUL should decrease (or stay at clip) as cycles increase."""
        result = compute_train_rul(sample_train_df)
        for unit_id, group in result.groupby("unit"):
            rul_values = group["RUL"].values
            # After clipping region, should be strictly decreasing
            below_clip = rul_values[rul_values < RUL_CLIP_VALUE]
            if len(below_clip) > 1:
                assert all(below_clip[i] >= below_clip[i+1] for i in range(len(below_clip)-1))


class TestNASAScore:
    """Tests for NASA asymmetric scoring function."""

    def test_perfect_prediction(self):
        """Perfect predictions should give score of 0."""
        y = np.array([50.0, 30.0, 10.0])
        assert nasa_score(y, y) == 0.0

    def test_late_penalty_higher(self):
        """Late predictions (over-estimate RUL) should be penalized more."""
        y_true = np.array([50.0])
        early = nasa_score(y_true, y_true - 10)  # predict 40 (early)
        late = nasa_score(y_true, y_true + 10)   # predict 60 (late)
        assert late > early

    def test_score_non_negative(self):
        """NASA score should always be non-negative."""
        y_true = np.random.rand(100) * 125
        y_pred = np.random.rand(100) * 125
        assert nasa_score(y_true, y_pred) >= 0


class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    def test_correct_feature_count(self, sample_train_df):
        """Should produce expected number of features."""
        df = compute_train_rul(sample_train_df)
        featured, scaler, feature_cols = engineer_features(df, fit_scaler=True)
        # 14 sensors * (1 raw + 3 rolling_mean + 3 rolling_std + 2 lags) = 14 * 9 = 126
        expected = len(USEFUL_SENSORS) * (1 + len(ROLLING_WINDOWS) * 2 + len(LAG_STEPS))
        assert len(feature_cols) == expected

    def test_no_nan_after_engineering(self, sample_train_df):
        """Feature matrix should have no NaN values."""
        df = compute_train_rul(sample_train_df)
        featured, scaler, feature_cols = engineer_features(df, fit_scaler=True)
        assert not featured[feature_cols].isnull().any().any()

    def test_scaler_not_refit_on_test(self, sample_train_df):
        """Test data should use the same scaler fitted on train."""
        df = compute_train_rul(sample_train_df)
        _, scaler, feature_cols = engineer_features(df, fit_scaler=True)
        # Apply to same data with fit_scaler=False
        featured2, _, _ = engineer_features(df, fit_scaler=False, scaler=scaler)
        assert not featured2[feature_cols].isnull().any().any()

    def test_values_normalized(self, sample_train_df):
        """After scaling, values should be in [0, 1] range (approximately)."""
        df = compute_train_rul(sample_train_df)
        featured, _, feature_cols = engineer_features(df, fit_scaler=True)
        vals = featured[feature_cols].values
        assert vals.min() >= -0.01  # small tolerance
        assert vals.max() <= 1.01


class TestEvaluationMetrics:
    """Tests for evaluation metric functions."""

    def test_rmse_perfect(self):
        """RMSE of identical arrays should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_rmse_known_value(self):
        """RMSE should match hand-calculated value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])  # error of 2 on last
        expected = np.sqrt((0 + 0 + 4) / 3)
        assert abs(compute_rmse(y_true, y_pred) - expected) < 1e-6

    def test_mae_perfect(self):
        """MAE of identical arrays should be 0."""
        y = np.array([10.0, 20.0, 30.0])
        assert compute_mae(y, y) == 0.0

    def test_critical_zone_only_counts_low_rul(self):
        """Critical zone RMSE should only consider engines with true RUL < 30."""
        y_true = np.array([100.0, 50.0, 20.0, 10.0])
        y_pred = np.array([100.0, 50.0, 25.0, 15.0])
        cz_rmse = compute_critical_zone_rmse(y_true, y_pred)
        # Only last two count: errors are 5 and 5
        expected = np.sqrt((25 + 25) / 2)
        assert abs(cz_rmse - expected) < 1e-6

    def test_risk_levels(self):
        """Risk levels should match thresholds."""
        assert get_risk_level(10) == "CRITICAL"
        assert get_risk_level(20) == "HIGH"
        assert get_risk_level(45) == "MEDIUM"
        assert get_risk_level(100) == "LOW"


class TestValidator:
    """Tests for data validation functions."""

    def test_validate_rul_non_negative_passes(self):
        """Should pass for non-negative values."""
        validate_rul_non_negative(np.array([0, 10, 125]))

    def test_validate_rul_non_negative_fails(self):
        """Should raise ValueError for negative values."""
        with pytest.raises(ValueError):
            validate_rul_non_negative(np.array([10, -1, 5]))

    def test_validate_no_nan_passes(self):
        """Should pass for clean DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        validate_no_nan_inf(df)

    def test_validate_no_nan_fails(self):
        """Should raise ValueError for DataFrame with NaN."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        with pytest.raises(ValueError):
            validate_no_nan_inf(df)
