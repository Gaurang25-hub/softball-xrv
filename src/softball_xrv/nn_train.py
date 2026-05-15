"""Preprocess features and train fold-level neural-network experiments for xRV."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from softball_xrv.models.neural_net import PitchNN
from softball_xrv.models.neural_net_losses import (
    hard_ce_plus_expected_distance_loss,
    make_soft_target_matrix,
    make_xrv_distance_matrix,
    soft_label_cross_entropy_loss,
)

NN_EXPERIMENTS = {
    "PyTorch_HardCE_plus_ExpectedDistance": "hard_ce_distance",
    "PyTorch_SoftLabelCE": "soft_label_ce",
    "PyTorch_HybridCE_SharpSoftCE_Distance": "hybrid_ce_sharp_soft_distance",
}


@dataclass(frozen=True)
class NNTrainConfig:
    """
    Hyperparameters for one fold-level neural-network training run.
    The config is frozen so each fold receives a stable, immutable set of
    architecture, optimizer, loss, and early-stopping settings.
    
    """

    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.20
    use_layer_norm: bool = True
    lambda_distance: float = 0.20
    soft_tau: float = 0.05
    sharp_tau: float = 0.005
    random_state: int = 42


def preprocess_nn_features(
    train_part_df: pd.DataFrame,
    val_part_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    NN-only preprocessing.
    Fit medians and scaler on inner-train only.
    Apply those same train medians/scaler to validation.
    """

    X_train = train_part_df[list(feature_cols)].copy()
    X_val = val_part_df[list(feature_cols)].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train.median(axis=0).fillna(0.0)

    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=np.float32)
    X_val_scaled = np.ascontiguousarray(X_val_scaled, dtype=np.float32)

    if X_train_scaled.shape[1] != len(feature_cols):
        raise ValueError(
            f"NN preprocessing changed feature count: "
            f"expected {len(feature_cols)}, got {X_train_scaled.shape[1]}"
        )

    if not np.isfinite(X_train_scaled).all():
        raise ValueError("X_train_scaled contains NaN or inf after preprocessing.")

    if not np.isfinite(X_val_scaled).all():
        raise ValueError("X_val_scaled contains NaN or inf after preprocessing.")

    return X_train_scaled, X_val_scaled


def compute_nn_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str,
    distance_matrix: torch.Tensor,
    soft_target_matrix: torch.Tensor,
    soft_target_matrix_sharp: torch.Tensor,
    lambda_distance: float,
) -> torch.Tensor:
    """Compute the configured NN loss from logits and integer class targets."""

    if loss_type == "hard_ce_distance":
        return hard_ce_plus_expected_distance_loss(
            logits=logits,
            y_true=y_true,
            distance_matrix=distance_matrix,
            lambda_distance=lambda_distance,
        )

    if loss_type == "soft_label_ce":
        return soft_label_cross_entropy_loss(
            logits=logits,
            y_true=y_true,
            soft_target_matrix=soft_target_matrix,
        )

    if loss_type == "hybrid_ce_sharp_soft_distance":
        ce_loss = F.cross_entropy(logits, y_true)

        soft_loss = soft_label_cross_entropy_loss(
            logits=logits,
            y_true=y_true,
            soft_target_matrix=soft_target_matrix_sharp,
        )

        probs = torch.softmax(logits, dim=1)
        row_distances = distance_matrix[y_true]
        expected_distance = (probs * row_distances).sum(dim=1).mean()

        return ce_loss + (lambda_distance * expected_distance) + (0.05 * soft_loss)

    raise ValueError(f"Unknown loss_type: {loss_type}")


def train_nn_one_fold(
    train_part_df: pd.DataFrame,
    val_part_df: pd.DataFrame,
    feature_cols: Sequence[str],
    loss_type: str,
    xrv_class_values: np.ndarray,
    target_col: str = "target_class",
    config: NNTrainConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Train one NN on one CV fold and return validation predictions.
    This does not compute final metrics and does not save reports.
    modeling.py will handle those parts.
    """

    if config is None:
        config = NNTrainConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    y_train = train_part_df[target_col].to_numpy(dtype=np.int64)
    y_val = val_part_df[target_col].to_numpy(dtype=np.int64)

    X_train_np, X_val_np = preprocess_nn_features(
        train_part_df=train_part_df,
        val_part_df=val_part_df,
        feature_cols=feature_cols,
    )

    X_train_tensor = torch.from_numpy(X_train_np)
    y_train_tensor = torch.from_numpy(y_train)

    X_val_tensor = torch.from_numpy(X_val_np).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    actual_batch_size = min(config.batch_size, len(X_train_tensor))

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = len(xrv_class_values)

    model = PitchNN(
        input_dim=X_train_np.shape[1],
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_layer_norm=config.use_layer_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    distance_matrix = make_xrv_distance_matrix(xrv_class_values).to(device)
    soft_target_matrix = make_soft_target_matrix(
        distance_matrix,
        tau=config.soft_tau,
    ).to(device)
    soft_target_matrix_sharp = make_soft_target_matrix(
        distance_matrix,
        tau=config.sharp_tau,
    ).to(device)

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_count = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        epochs_trained = epoch + 1

        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))

            logits = model(xb)

            loss = compute_nn_loss(
                logits=logits,
                y_true=yb,
                loss_type=loss_type,
                distance_matrix=distance_matrix,
                soft_target_matrix=soft_target_matrix,
                soft_target_matrix_sharp=soft_target_matrix_sharp,
                lambda_distance=config.lambda_distance,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite NN training loss at epoch {epoch + 1}: {loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = compute_nn_loss(
                logits=val_logits,
                y_true=y_val_tensor,
                loss_type=loss_type,
                distance_matrix=distance_matrix,
                soft_target_matrix=soft_target_matrix,
                soft_target_matrix_sharp=soft_target_matrix_sharp,
                lambda_distance=config.lambda_distance,
            )

        if not torch.isfinite(val_loss):
            raise ValueError(
                f"Non-finite NN validation loss at epoch {epoch + 1}: "
                f"{val_loss.item()}"
            )

        val_loss_value = float(val_loss.detach().cpu().item())
        avg_train_loss = float(np.mean(batch_losses))

        if verbose:
            print(
                f"Epoch {epoch + 1}/{config.max_epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss_value:.4f}",
                flush=True,
            )

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_proba = torch.softmax(val_logits, dim=1).detach().cpu().numpy()

    if not np.isfinite(val_proba).all():
        raise ValueError("Validation probabilities contain NaN or inf.")

    val_proba = np.clip(val_proba, 1e-15, 1.0)
    val_proba = val_proba / val_proba.sum(axis=1, keepdims=True)

    val_pred = val_proba.argmax(axis=1)

    if verbose:
        print(
            f"Best checkpoint | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss={best_val_loss:.4f} | "
            f"epochs_trained={epochs_trained}",
            flush=True,
        )

    return {
        "y_true": y_val,
        "y_pred": val_pred,
        "y_proba": val_proba,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }
