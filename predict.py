"""Make one regression, classification, and clustering prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.data_pipeline import FEATURE_COLUMNS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict from one JSON farm record.")
    parser.add_argument("--record", required=True, help="JSON object containing six input fields.")
    parser.add_argument("--models", default="models", help="Folder containing trained models.")
    return parser.parse_args()


def parse_record(record_text: str) -> dict[str, float]:
    try:
        record = json.loads(record_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON: {error.msg}") from error

    if not isinstance(record, dict):
        raise ValueError("The record must be a JSON object.")

    missing_fields = [field for field in FEATURE_COLUMNS if field not in record]
    if missing_fields:
        raise ValueError("Missing required field(s): " + ", ".join(missing_fields))

    extra_fields = [field for field in record if field not in FEATURE_COLUMNS]
    if extra_fields:
        raise ValueError("Unexpected field(s): " + ", ".join(extra_fields))

    values: dict[str, float] = {}
    for field in FEATURE_COLUMNS:
        value = record[field]
        if isinstance(value, bool):
            raise ValueError(f"Field '{field}' must be numeric, not true/false.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Field '{field}' must be numeric.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"Field '{field}' must be finite.")
        values[field] = numeric_value
    return values


def make_prediction(record: dict[str, float], models_dir: str | Path) -> dict:
    models_path = Path(models_dir)
    row = np.array([[record[field] for field in FEATURE_COLUMNS]], dtype=float)

    regression_scaler = joblib.load(models_path / "regression_scaler.joblib")
    regression_model = np.load(models_path / "regression_model.npz")
    regression_features = regression_scaler.transform(row)
    regression_scaled = (
        regression_features @ regression_model["weights"] + regression_model["bias"][0]
    )
    regression_prediction = (
        regression_scaled[0] * regression_model["target_scale"][0]
        + regression_model["target_mean"][0]
    )

    classification_scaler = joblib.load(models_path / "classification_scaler.joblib")
    classification_bundle = joblib.load(models_path / "classification_model.joblib")
    classification_features = classification_scaler.transform(row)
    classifier = classification_bundle["classifier"]
    threshold = float(classification_bundle["threshold"])
    probability = float(classifier.predict_proba(classification_features)[0, 1])
    classification_prediction = int(probability >= threshold)

    clustering_scaler = joblib.load(models_path / "clustering_scaler.joblib")
    clustering_model = joblib.load(models_path / "clustering_model.joblib")
    cluster_label = int(
        clustering_model.predict(clustering_scaler.transform(row))[0]
    )

    with (models_path / "model_metadata.json").open("r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    return {
        "regression_prediction": round(float(regression_prediction), 6),
        "classification_prediction": classification_prediction,
        "classification_probability": round(probability, 6),
        "cluster_label": cluster_label,
        "group_code": metadata["group_code"],
        "model_version": metadata["model_version"],
    }


def main() -> int:
    arguments = parse_arguments()
    record = parse_record(arguments.record)
    result = make_prediction(record, arguments.models)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, FileNotFoundError) as error:
        print(f"Validation/model error: {error}", file=sys.stderr)
        raise SystemExit(2)

