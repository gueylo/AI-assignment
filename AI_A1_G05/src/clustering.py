"""KMeans clustering using only the six operational input features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_pipeline import FEATURE_COLUMNS, IDENTIFIER_COLUMN, RANDOM_SEED, dataframe_to_features


K_VALUES = [2, 3, 4, 5]


def run_clustering(
    dataframe: pd.DataFrame,
    artifacts_dir: str | Path,
    models_dir: str | Path,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Evaluate KMeans values, select the best one, and save cluster labels."""

    features = dataframe_to_features(dataframe)
    if len(dataframe) < 3:
        raise ValueError("Clustering needs at least 3 rows to calculate a silhouette score.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    silhouette_scores: dict[str, float | None] = {}
    valid_models: dict[int, KMeans] = {}
    for k in K_VALUES:
        if k >= len(dataframe):
            silhouette_scores[str(k)] = None
            continue
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(scaled_features)
        if len(np.unique(labels)) < 2:
            silhouette_scores[str(k)] = None
            continue
        silhouette_scores[str(k)] = float(silhouette_score(scaled_features, labels))
        valid_models[k] = model

    valid_scores = {
        k: score for k, score in silhouette_scores.items() if score is not None
    }
    if not valid_scores:
        raise ValueError("No valid k value was available for silhouette evaluation.")

    best_k = int(max(valid_scores, key=valid_scores.get))
    selected_model = valid_models[best_k]
    cluster_labels = selected_model.labels_

    artifacts_path = Path(artifacts_dir)
    models_path = Path(models_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    clusters = pd.DataFrame(
        {
            IDENTIFIER_COLUMN: dataframe[IDENTIFIER_COLUMN].to_numpy(),
            "cluster": cluster_labels.astype(int),
        }
    )
    clusters.to_csv(artifacts_path / "clusters.csv", index=False)

    # PCA is used only to display six-dimensional clusters on a two-dimensional plot.
    plotting_coordinates = PCA(n_components=2, random_state=random_seed).fit_transform(
        scaled_features
    )
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        plotting_coordinates[:, 0],
        plotting_coordinates[:, 1],
        c=cluster_labels,
        cmap="tab10",
        edgecolor="black",
        alpha=0.85,
    )
    plt.title(f"KMeans Operating-Profile Clusters (k={best_k})")
    plt.xlabel("PCA component 1 (plotting only)")
    plt.ylabel("PCA component 2 (plotting only)")
    plt.colorbar(scatter, label="Cluster label")
    plt.tight_layout()
    plt.savefig(artifacts_path / "cluster_plot.png", dpi=160)
    plt.close()

    metrics = {
        "random_seed": random_seed,
        "feature_columns": FEATURE_COLUMNS,
        "evaluated_k_values": K_VALUES,
        "silhouette_scores": silhouette_scores,
        "selected_k": best_k,
        "interpretation": (
            "Clusters are mathematical groups based on similar standardized operating "
            "characteristics in this dataset; they are not good/bad farm labels."
        ),
    }
    with (artifacts_path / "clustering_metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    joblib.dump(scaler, models_path / "clustering_scaler.joblib")
    joblib.dump(selected_model, models_path / "clustering_model.joblib")
    return metrics

