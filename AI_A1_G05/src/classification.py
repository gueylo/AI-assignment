"""Dispatch-attention classification using logistic regression."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_pipeline import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    RANDOM_SEED,
    dataframe_to_features,
)


# This setting is intentionally easy to find and change for live verification.
CLASSIFICATION_THRESHOLD = 0.50
TEST_SIZE = 0.20


def _can_stratify(labels: np.ndarray, test_size: float) -> bool:
    """Return whether a safe stratified split is possible for this dataset."""

    class_counts = np.bincount(labels.astype(int), minlength=2)
    class_count = int(np.count_nonzero(class_counts))
    test_count = int(np.ceil(len(labels) * test_size))
    train_count = len(labels) - test_count
    return (
        class_count >= 2
        and int(class_counts.min()) >= 2
        and test_count >= class_count
        and train_count >= class_count
    )


def run_classification(
    dataframe: pd.DataFrame,
    artifacts_dir: str | Path,
    models_dir: str | Path,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Train, evaluate, plot, and save the classification model."""

    features = dataframe_to_features(dataframe)
    labels = dataframe[CLASSIFICATION_TARGET].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Classification needs both 0 and 1 examples in dispatch_attention.")
    if len(dataframe) < 5:
        raise ValueError("Classification needs at least 5 rows for a train/test split.")

    row_indexes = np.arange(len(dataframe))
    stratify = labels if _can_stratify(labels, TEST_SIZE) else None
    train_indexes, test_indexes = train_test_split(
        row_indexes,
        test_size=TEST_SIZE,
        random_state=random_seed,
        stratify=stratify,
    )

    x_train = features[train_indexes]
    x_test = features[test_indexes]
    y_train = labels[train_indexes]
    y_test = labels[test_indexes]
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "The training split contains only one dispatch_attention class; "
            "add more examples before training."
        )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
    classifier.fit(x_train_scaled, y_train)
    probabilities = classifier.predict_proba(x_test_scaled)[:, 1]
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "random_seed": random_seed,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "train_size": int(len(train_indexes)),
        "test_size": int(len(test_indexes)),
        "stratified_split_used": stratify is not None,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "scenario_interpretation": (
            "A false negative means the program predicts that dispatch attention is not "
            "required even though it was actually required. In this operational scenario, "
            "a false negative may be particularly costly because a consignment needing "
            "intervention could be missed."
        ),
    }

    artifacts_path = Path(artifacts_dir)
    models_path = Path(models_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    with (artifacts_path / "classification_metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Dispatch Attention Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, int(matrix[row, column]), ha="center", va="center")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(artifacts_path / "confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(
        {
            "classifier": classifier,
            "threshold": CLASSIFICATION_THRESHOLD,
            "feature_columns": FEATURE_COLUMNS,
        },
        models_path / "classification_model.joblib",
    )
    joblib.dump(scaler, models_path / "classification_scaler.joblib")
    return metrics

