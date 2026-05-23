"""
PyTorch LSTM regressor for RUL prediction.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_LEARNING_RATE,
    LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_GRAD_CLIP, LSTM_PATIENCE,
    LSTM_VAL_FRACTION, MODEL_DIR, LSTM_MODEL_FILE, RANDOM_SEED
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNet(nn.Module):
    """LSTM network with 2-layer MLP head for RUL regression."""

    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS, dropout: float = LSTM_DROPOUT):
        """
        Args:
            input_size: Number of input features.
            hidden_size: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Takes (batch, seq_len, features), returns (batch, 1)."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


class LSTMRULModel:
    """LSTM model wrapper with train/predict/save/load interface."""

    def __init__(self, input_size: int):
        """
        Args:
            input_size: Number of input features per timestep.
        """
        self.input_size = input_size
        self.model = LSTMNet(input_size).to(device)
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train LSTM with early stopping and learning rate scheduling.

        Args:
            X_train: Training sequences (N, seq_len, features).
            y_train: Training targets.
            X_val: Validation sequences.
            y_val: Validation targets.

        Returns:
            Dictionary with training history.
        """
        # If no val set provided, split from training data by engine count
        if X_val is None:
            n_val = max(1, int(len(X_train) * LSTM_VAL_FRACTION))
            np.random.seed(RANDOM_SEED)
            indices = np.random.permutation(len(X_train))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(LSTM_EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = self.model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), LSTM_GRAD_CLIP)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_dataset)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = self.model(X_batch).squeeze()
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * len(X_batch)
            val_loss /= len(val_dataset)

            scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{LSTM_EPOCHS} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= LSTM_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        logger.info(f"LSTM training complete. Best val loss: {best_val_loss:.4f}")
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL values from sequences.

        Args:
            X: Input sequences (N, seq_len, features).

        Returns:
            Array of predicted RUL values.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = []
            # Batch prediction to avoid memory issues
            for i in range(0, len(X_tensor), LSTM_BATCH_SIZE):
                batch = X_tensor[i:i + LSTM_BATCH_SIZE]
                pred = self.model(batch).squeeze().cpu().numpy()
                if pred.ndim == 0:
                    pred = np.array([pred.item()])
                predictions.append(pred)
        return np.concatenate(predictions)

    def save(self, path: Path = None) -> None:
        """
        Save model state dict.

        Args:
            path: Save path. Defaults to MODEL_DIR/LSTM_MODEL_FILE.
        """
        save_path = path or (MODEL_DIR / LSTM_MODEL_FILE)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, save_path)
        logger.info(f"LSTM model saved to {save_path}")

    def load(self, path: Path = None) -> None:
        """
        Load model state dict.

        Args:
            path: Load path. Defaults to MODEL_DIR/LSTM_MODEL_FILE.
        """
        load_path = path or (MODEL_DIR / LSTM_MODEL_FILE)
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        self.input_size = checkpoint["input_size"]
        self.model = LSTMNet(self.input_size).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"LSTM model loaded from {load_path}")
