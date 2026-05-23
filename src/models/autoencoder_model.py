"""
LSTM Autoencoder for anomaly detection in engine degradation.
Trained on healthy-only data to detect deviations from normal behavior.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib

from src.config import (
    AE_HIDDEN_SIZE, AE_NUM_LAYERS, AE_DROPOUT, AE_LEARNING_RATE,
    AE_BATCH_SIZE, AE_EPOCHS, AE_THRESHOLD_PERCENTILE, SEQUENCE_LENGTH,
    MODEL_DIR, AUTOENCODER_MODEL_FILE, AE_THRESHOLD_FILE
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder for sequence reconstruction."""

    def __init__(self, input_size: int, hidden_size: int = AE_HIDDEN_SIZE,
                 num_layers: int = AE_NUM_LAYERS, dropout: float = AE_DROPOUT):
        """
        Args:
            input_size: Number of input features per timestep.
            hidden_size: Hidden dimension for LSTM layers.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode the input sequence.

        Args:
            x: Input tensor (batch, seq_len, features).

        Returns:
            Reconstructed tensor of same shape.
        """
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decode - repeat encoded representation for each timestep
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))

        # Output projection
        reconstruction = self.output_layer(decoder_out)
        return reconstruction


class AutoencoderAnomalyModel:
    """LSTM Autoencoder wrapper for anomaly detection."""

    def __init__(self, input_size: int):
        """
        Args:
            input_size: Number of input features per timestep.
        """
        self.input_size = input_size
        self.model = LSTMAutoencoder(input_size).to(device)
        self.threshold = None

    def fit(self, X_healthy: np.ndarray) -> dict:
        """
        Train autoencoder on healthy-only sequences and calibrate threshold.

        Args:
            X_healthy: Healthy sequences (N, seq_len, features).

        Returns:
            Dictionary with training info.
        """
        logger.info(f"Training autoencoder on {len(X_healthy)} healthy sequences")

        dataset = TensorDataset(torch.FloatTensor(X_healthy))
        loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=AE_LEARNING_RATE)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None

        for epoch in range(AE_EPOCHS):
            self.model.train()
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(dataset)

            if epoch % 10 == 0:
                logger.info(f"AE Epoch {epoch}/{AE_EPOCHS} - Loss: {epoch_loss:.6f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        # Calibrate threshold using 95th percentile of healthy reconstruction errors
        errors = self._compute_reconstruction_errors(X_healthy)
        self.threshold = float(np.percentile(errors, AE_THRESHOLD_PERCENTILE))
        logger.info(
            f"AE threshold calibrated at {AE_THRESHOLD_PERCENTILE}th percentile: "
            f"{self.threshold:.6f}"
        )

        return {
            "best_loss": best_loss,
            "threshold": self.threshold,
            "n_healthy_sequences": len(X_healthy),
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
        }

    def _compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sequence reconstruction error (MSE).

        Args:
            X: Input sequences (N, seq_len, features).

        Returns:
            Array of reconstruction errors per sequence.
        """
        self.model.eval()
        errors = []
        with torch.no_grad():
            dataset = TensorDataset(torch.FloatTensor(X))
            loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE)
            for (batch,) in loader:
                batch = batch.to(device)
                reconstruction = self.model(batch)
                mse = ((reconstruction - batch) ** 2).mean(dim=(1, 2))
                errors.append(mse.cpu().numpy())
        return np.concatenate(errors)

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (reconstruction errors) for sequences.

        Args:
            X: Input sequences (N, seq_len, features).

        Returns:
            Array of anomaly scores.
        """
        return self._compute_reconstruction_errors(X)

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Determine if sequences are anomalous based on calibrated threshold.

        Args:
            X: Input sequences (N, seq_len, features).

        Returns:
            Boolean array (True = anomaly).
        """
        scores = self.predict_anomaly_score(X)
        return scores > self.threshold

    def save(self, path: Path = None) -> None:
        """
        Save model and threshold.

        Args:
            path: Save path for model. Defaults to MODEL_DIR/AUTOENCODER_MODEL_FILE.
        """
        save_path = path or (MODEL_DIR / AUTOENCODER_MODEL_FILE)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.input_size,
        }, save_path)
        # Save threshold separately
        threshold_path = save_path.parent / AE_THRESHOLD_FILE
        joblib.dump(self.threshold, threshold_path)
        logger.info(f"Autoencoder saved to {save_path}, threshold to {threshold_path}")

    def load(self, path: Path = None) -> None:
        """
        Load model and threshold.

        Args:
            path: Load path. Defaults to MODEL_DIR/AUTOENCODER_MODEL_FILE.
        """
        load_path = path or (MODEL_DIR / AUTOENCODER_MODEL_FILE)
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        self.input_size = checkpoint["input_size"]
        self.model = LSTMAutoencoder(self.input_size).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Load threshold
        threshold_path = load_path.parent / AE_THRESHOLD_FILE
        self.threshold = joblib.load(threshold_path)
        logger.info(f"Autoencoder loaded from {load_path}, threshold: {self.threshold:.6f}")
