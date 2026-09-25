"""Manual linear regression trained with NumPy batch gradient descent."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_pipeline import FEATURE_COLUMNS, REGRESSION_TARGET, RANDOM_SEED, dataframe_to_features


# These settings are intentionally near the top for easy live verification changes.
LEARNING_RATE = 0.01
EPOCHS = 1000
TEST_SIZE = 0.20


def _gradient_descent(
    features: np.ndarray,
    target: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
) -> tuple[np.ndarray, float, list[float]]:
    """Fit weights and bias using the batch gradient descent equations."""

    sample_count, feature_count = features.shape
    weights = np.zeros(feature_count, dtype=float)
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        predictions = features @ weights + bias
        errors = predictions - target
        loss = float(np.mean(errors**2))
        losses.append(loss)

        # These are the derivatives of mean squared error.
        weight_gradient = (2.0 / sample_count) * (features.T @ errors)
        bias_gradient = (2.0 / sample_count) * np.sum(errors)
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weights, bias, losses


def run_regression(
    dataframe: pd.DataFrame,
    artifacts_dir: str | Path,
    models_dir: str | Path,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Train, evaluate, plot, and save the NumPy regression model."""

    if len(dataframe) < 5:
        raise ValueError("Regression needs at least 5 rows for a train/test split.")

    features = dataframe_to_features(dataframe)
    target = dataframe[REGRESSION_TARGET].to_numpy(dtype=float)
    row_indexes = np.arange(len(dataframe))

    train_indexes, test_indexes = train_test_split(
        row_indexes, test_size=TEST_SIZE, random_state=random_seed
    )
    x_train = features[train_indexes]
    x_test = features[test_indexes]
    y_train = target[train_indexes]
    y_test = target[test_indexes]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Scaling the target only makes gradient descent numerically stable. Metrics
    # and predictions are converted back to the original kilogram units below.
    target_mean = float(np.mean(y_train))
    target_scale = float(np.std(y_train))
    if target_scale == 0:
        target_scale = 1.0
    y_train_scaled = (y_train - target_mean) / target_scale

    weights, bias, losses = _gradient_descent(
        x_train_scaled, y_train_scaled, LEARNING_RATE, EPOCHS
    )
    test_predictions_scaled = x_test_scaled @ weights + bias
    test_predictions = test_predictions_scaled * target_scale + target_mean

    metrics = {
        "random_seed": random_seed,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "feature_columns": FEATURE_COLUMNS,
        "train_size": int(len(train_indexes)),
        "test_size": int(len(test_indexes)),
        "MAE": float(mean_absolute_error(y_test, test_predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "R2": float(r2_score(y_test, test_predictions)) if len(y_test) >= 2 else None,
        "final_training_loss": float(losses[-1]),
        "target_scaling": "training mean and standard deviation used for stable gradient descent",
    }

    artifacts_path = Path(artifacts_dir)
    models_path = Path(models_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    with (artifacts_path / "regression_metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color="navy")
    plt.title("Regression Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error on scaled target")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifacts_path / "regression_loss.png", dpi=160)
    plt.close()

    np.savez(
        models_path / "regression_model.npz",
        weights=weights,
        bias=np.array([bias]),
        target_mean=np.array([target_mean]),
        target_scale=np.array([target_scale]),
    )
    joblib.dump(scaler, models_path / "regression_scaler.joblib")
    return metrics

