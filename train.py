"""
Master training pipeline for IndustrialSentinel.
Runs all steps from data loading through model evaluation and MLflow logging.
"""
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")

from src.config import (
    RAW_DATA_DIR, MODEL_DIR, PROCESSED_DATA_DIR, USEFUL_SENSORS,
    LOW_VARIANCE_SENSORS, VARIANCE_THRESHOLD, SENSOR_COLS, RUL_CLIP_VALUE,
    SEQUENCE_LENGTH, ENSEMBLE_WEIGHTS, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, RANDOM_SEED,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_LEARNING_RATE,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_GRAD_CLIP, AE_HEALTHY_FRACTION,
    AE_THRESHOLD_PERCENTILE, XGB_PARAMS, ROLLING_WINDOWS, METADATA_FILE
)
from src.data_loader import load_raw_data
from src.rul_calculator import compute_train_rul, compute_test_rul
from src.feature_engineering import (
    engineer_features, create_sequences, create_test_sequences,
    create_autoencoder_sequences
)
from src.models.xgboost_model import XGBoostRULModel
from src.models.lstm_model import LSTMRULModel
from src.models.autoencoder_model import AutoencoderAnomalyModel
from src.ensemble import ensemble_predict
from src.evaluation import evaluate_all
from src.drift_detector import save_training_distributions
from src.validator import (
    validate_raw_files, validate_no_nan_inf, validate_sequence_shape,
    validate_model_artifacts_exist
)


def main():
    """Run the complete training pipeline."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("  IndustrialSentinel Training Pipeline")
    logger.info(f"  Started: {start_time.isoformat()}")
    logger.info("=" * 60)

    try:
        # Step 1: Validate environment
        logger.info("\n[Step 1/15] Validating environment...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        validate_raw_files(str(RAW_DATA_DIR.parent))

        # Step 2: Load and validate raw data
        logger.info("\n[Step 2/15] Loading raw data...")
        train_df, test_df, rul_df = load_raw_data()
        logger.info(f"  Train: {train_df.shape} ({train_df['unit'].nunique()} engines)")
        logger.info(f"  Test:  {test_df.shape} ({test_df['unit'].nunique()} engines)")
        logger.info(f"  RUL:   {rul_df.shape}")

        # Step 3: EDA - verify sensor variance
        logger.info("\n[Step 3/15] Running EDA - sensor variance analysis...")
        sensor_variance = train_df[SENSOR_COLS].var()
        variance_table = pd.DataFrame({
            "sensor": SENSOR_COLS,
            "variance": [sensor_variance[s] for s in SENSOR_COLS]
        }).sort_values("variance")
        logger.info("\nSensor Variance Table:")
        logger.info(variance_table.to_string())

        # Confirm low-variance sensors
        empirical_low_var = variance_table[
            variance_table["variance"] < VARIANCE_THRESHOLD
        ]["sensor"].tolist()
        logger.info(f"\nEmpirical low-variance sensors: {empirical_low_var}")
        logger.info(f"Configured low-variance sensors: {LOW_VARIANCE_SENSORS}")

        for s in LOW_VARIANCE_SENSORS:
            assert s in empirical_low_var or sensor_variance[s] < 1.0, \
                f"Sensor {s} was expected to be low-variance but has variance {sensor_variance[s]}"
        logger.info("Sensor selection confirmed empirically.")

        # Save variance table as artifact
        variance_dict = {s: float(v) for s, v in sensor_variance.items()}

        # Step 4: Compute RUL labels
        logger.info("\n[Step 4/15] Computing RUL labels...")
        train_df = compute_train_rul(train_df)
        test_df = compute_test_rul(test_df, rul_df)

        # Step 5: Feature engineering
        logger.info("\n[Step 5/15] Engineering features...")
        train_featured, scaler, feature_cols = engineer_features(train_df, fit_scaler=True)
        test_featured, _, _ = engineer_features(test_df, fit_scaler=False, scaler=scaler)

        validate_no_nan_inf(train_featured[feature_cols], "Train features")
        validate_no_nan_inf(test_featured[feature_cols], "Test features")
        logger.info(f"  Feature columns: {len(feature_cols)}")

        # Save training distributions for drift detection
        save_training_distributions(train_df)

        # Create sequences for LSTM
        logger.info("  Creating LSTM sequences...")
        X_train_seq, y_train_seq = create_sequences(train_featured, feature_cols)
        X_test_seq, y_test_seq = create_test_sequences(test_featured, feature_cols)

        validate_sequence_shape(X_train_seq, len(feature_cols))
        validate_sequence_shape(X_test_seq, len(feature_cols))
        assert X_test_seq.shape[0] == 100, f"Expected 100 test sequences, got {X_test_seq.shape[0]}"
        logger.info(f"  Train sequences: {X_train_seq.shape}")
        logger.info(f"  Test sequences: {X_test_seq.shape}")

        # Create autoencoder sequences (healthy only)
        X_healthy = create_autoencoder_sequences(train_featured, feature_cols)
        logger.info(f"  Healthy sequences for AE: {X_healthy.shape}")

        # Step 6: Train XGBoost
        logger.info("\n[Step 6/15] Training XGBoost...")
        xgb_model = XGBoostRULModel()
        xgb_info = xgb_model.fit(train_featured, feature_cols)
        xgb_model.save()

        # XGBoost test predictions (using flat features, last cycle per engine)
        test_last = test_featured.groupby("unit").last().reset_index()
        xgb_test_pred = xgb_model.predict(test_last[feature_cols])
        xgb_test_true = test_last["RUL"].values
        xgb_metrics = evaluate_all(xgb_test_true, xgb_test_pred, "XGBoost")

        # Step 7: Train LSTM
        logger.info("\n[Step 7/15] Training LSTM...")
        lstm_model = LSTMRULModel(input_size=len(feature_cols))
        lstm_info = lstm_model.fit(X_train_seq, y_train_seq)
        lstm_model.save()

        # LSTM test predictions
        lstm_test_pred = lstm_model.predict(X_test_seq)
        lstm_metrics = evaluate_all(y_test_seq, lstm_test_pred, "LSTM")

        # Step 8: Train Autoencoder
        logger.info("\n[Step 8/15] Training LSTM Autoencoder...")
        ae_model = AutoencoderAnomalyModel(input_size=len(feature_cols))
        ae_info = ae_model.fit(X_healthy)
        ae_model.save()

        # Step 9: Evaluate on official test set
        logger.info("\n[Step 9/15] Evaluating on official test set...")
        # Already done above for individual models

        # Step 10: Ensemble predictions
        logger.info("\n[Step 10/15] Computing ensemble predictions...")
        ensemble_pred = ensemble_predict(xgb_test_pred, lstm_test_pred)
        ensemble_metrics = evaluate_all(y_test_seq, ensemble_pred, "Ensemble")

        # Step 11: Model comparison table
        logger.info("\n[Step 11/15] Model comparison...")
        print("\n" + "=" * 70)
        print(f"{'Metric':<25} {'XGBoost':>10} {'LSTM':>10} {'Ensemble':>10}")
        print("-" * 70)
        for metric in ["rmse", "mae", "nasa_score", "critical_zone_rmse",
                       "detection_lead_time", "false_alert_rate"]:
            print(f"{metric:<25} {xgb_metrics[metric]:>10.4f} "
                  f"{lstm_metrics[metric]:>10.4f} {ensemble_metrics[metric]:>10.4f}")
        print("=" * 70)

        # Step 12: MLflow logging
        logger.info("\n[Step 12/15] Logging to MLflow...")
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"train_{start_time.strftime('%Y%m%d_%H%M%S')}"):
                # Log params
                mlflow.log_params({
                    "rul_clip_value": RUL_CLIP_VALUE,
                    "sequence_length": SEQUENCE_LENGTH,
                    "rolling_windows": str(ROLLING_WINDOWS),
                    "lstm_hidden_size": LSTM_HIDDEN_SIZE,
                    "lstm_num_layers": LSTM_NUM_LAYERS,
                    "lstm_dropout": LSTM_DROPOUT,
                    "lstm_learning_rate": LSTM_LEARNING_RATE,
                    "lstm_epochs": LSTM_EPOCHS,
                    "lstm_batch_size": LSTM_BATCH_SIZE,
                    "lstm_grad_clip": LSTM_GRAD_CLIP,
                    "ae_healthy_fraction": AE_HEALTHY_FRACTION,
                    "ae_threshold_percentile": AE_THRESHOLD_PERCENTILE,
                    "ensemble_weights": str(ENSEMBLE_WEIGHTS),
                    "xgb_max_depth": XGB_PARAMS["max_depth"],
                    "xgb_n_estimators": XGB_PARAMS["n_estimators"],
                    "xgb_learning_rate": XGB_PARAMS["learning_rate"],
                    "n_features": len(feature_cols),
                    "n_useful_sensors": len(USEFUL_SENSORS),
                })

                # Log metrics
                for prefix, metrics in [("xgb_", xgb_metrics), ("lstm_", lstm_metrics),
                                         ("ensemble_", ensemble_metrics)]:
                    for k, v in metrics.items():
                        mlflow.log_metric(f"{prefix}{k}", v)

                # Log artifacts
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Feature importance plot
                fi = xgb_model.get_feature_importance(feature_cols)
                fig, ax = plt.subplots(figsize=(10, 8))
                top_fi = fi.head(20)
                ax.barh(range(len(top_fi)), top_fi["importance"].values)
                ax.set_yticks(range(len(top_fi)))
                ax.set_yticklabels(top_fi["feature"].values)
                ax.set_xlabel("Importance")
                ax.set_title("XGBoost Top 20 Feature Importance")
                plt.tight_layout()
                fi_path = PROCESSED_DATA_DIR / "feature_importance.png"
                fig.savefig(fi_path)
                plt.close()
                mlflow.log_artifact(str(fi_path))

                # LSTM training curve
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(lstm_model.train_losses, label="Train Loss")
                ax.plot(lstm_model.val_losses, label="Val Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE Loss")
                ax.set_title("LSTM Training Curve")
                ax.legend()
                plt.tight_layout()
                curve_path = PROCESSED_DATA_DIR / "lstm_training_curve.png"
                fig.savefig(curve_path)
                plt.close()
                mlflow.log_artifact(str(curve_path))

                # Predicted vs True scatter
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(y_test_seq, ensemble_pred, alpha=0.6, s=50)
                ax.plot([0, 130], [0, 130], "r--", label="Perfect")
                ax.set_xlabel("True RUL")
                ax.set_ylabel("Predicted RUL")
                ax.set_title("Ensemble: Predicted vs True RUL")
                ax.legend()
                plt.tight_layout()
                scatter_path = PROCESSED_DATA_DIR / "pred_vs_true.png"
                fig.savefig(scatter_path)
                plt.close()
                mlflow.log_artifact(str(scatter_path))

                # Log variance table
                variance_path = PROCESSED_DATA_DIR / "sensor_variance.json"
                with open(variance_path, "w") as f:
                    json.dump(variance_dict, f, indent=2)
                mlflow.log_artifact(str(variance_path))

                # Register model
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    MLFLOW_MODEL_NAME
                )

            logger.info("MLflow logging complete.")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

        # Step 13: Already done in step 12 (model registry)

        # Step 14: Write metadata.json
        logger.info("\n[Step 14/15] Writing metadata.json...")
        metadata = {
            "training_date": start_time.isoformat(),
            "dataset": "NASA CMAPSS FD001",
            "n_train_engines": int(train_df["unit"].nunique()),
            "n_test_engines": int(test_df["unit"].nunique()),
            "n_features": len(feature_cols),
            "hyperparameters": {
                "rul_clip_value": RUL_CLIP_VALUE,
                "sequence_length": SEQUENCE_LENGTH,
                "rolling_windows": ROLLING_WINDOWS,
                "ensemble_weights": ENSEMBLE_WEIGHTS,
                "xgb_params": XGB_PARAMS,
                "lstm_hidden_size": LSTM_HIDDEN_SIZE,
                "lstm_num_layers": LSTM_NUM_LAYERS,
                "lstm_dropout": LSTM_DROPOUT,
                "lstm_learning_rate": LSTM_LEARNING_RATE,
                "lstm_epochs_trained": lstm_info["epochs_trained"],
                "ae_healthy_fraction": AE_HEALTHY_FRACTION,
                "ae_threshold_percentile": AE_THRESHOLD_PERCENTILE,
                "ae_threshold_value": ae_info["threshold"],
            },
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "ensemble_metrics": ensemble_metrics,
            "rmse": ensemble_metrics["rmse"],
            "mae": ensemble_metrics["mae"],
            "nasa_score": ensemble_metrics["nasa_score"],
            "critical_zone_rmse": ensemble_metrics["critical_zone_rmse"],
            "detection_lead_time": ensemble_metrics["detection_lead_time"],
            "false_alert_rate": ensemble_metrics["false_alert_rate"],
            "artifacts": {
                "xgboost_model": "xgboost_rul.pkl",
                "lstm_model": "lstm_rul.pt",
                "autoencoder_model": "autoencoder.pt",
                "ae_threshold": "ae_threshold.pkl",
                "scaler": "scaler.pkl",
                "feature_cols": "feature_cols.pkl",
                "train_stats": "train_stats.pkl",
            },
            "xgb_training_info": {
                "val_engines": xgb_info["val_engines"],
                "val_rmse": xgb_info["val_rmse"],
                "best_iteration": xgb_info["best_iteration"],
            },
            "ae_training_info": {
                "n_healthy_sequences": ae_info["n_healthy_sequences"],
                "threshold": ae_info["threshold"],
            },
        }

        metadata_path = MODEL_DIR / METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata written to {metadata_path}")

        # Save test predictions for dashboard
        test_results = pd.DataFrame({
            "unit": test_last["unit"].values,
            "true_rul": y_test_seq,
            "xgb_pred": xgb_test_pred,
            "lstm_pred": lstm_test_pred,
            "ensemble_pred": ensemble_pred,
        })
        test_results.to_csv(PROCESSED_DATA_DIR / "test_predictions.csv", index=False)

        # Save test featured data for demo endpoint
        test_featured.to_parquet(PROCESSED_DATA_DIR / "test_featured.parquet", index=False)

        # Validate all artifacts exist
        validate_model_artifacts_exist()

        # Step 15: Final summary
        logger.info("\n[Step 15/15] Final Summary")
        duration = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"  Ensemble MAE: {ensemble_metrics['mae']:.4f}")
        print(f"  NASA Score: {ensemble_metrics['nasa_score']:.4f}")
        print(f"  Critical Zone RMSE: {ensemble_metrics['critical_zone_rmse']:.4f}")
        print(f"  Detection Lead Time: {ensemble_metrics['detection_lead_time']:.4f}")
        print(f"  False Alert Rate: {ensemble_metrics['false_alert_rate']:.4f}")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
