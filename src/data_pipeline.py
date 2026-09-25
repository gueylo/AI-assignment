"""Loading, validation, and reporting for the assignment dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SEED = 42
MODEL_VERSION = "1.0"

FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]
REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"
IDENTIFIER_COLUMN = "record_id"

REQUIRED_COLUMNS = [
    IDENTIFIER_COLUMN,
    *FEATURE_COLUMNS,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
]
NUMERIC_COLUMNS = [*FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET]


class DataValidationError(ValueError):
    """Raised when the supplied CSV cannot safely be used."""


def calculate_sha256(csv_path: Path) -> str:
    """Return the SHA-256 hash of the original CSV bytes."""

    digest = hashlib.sha256()
    with csv_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a CSV and perform the checks required by the assignment."""

    path = Path(csv_path)
    if not path.is_file():
        raise DataValidationError(f"CSV file was not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as error:  # pandas can raise several parser-related errors.
        raise DataValidationError(f"Could not read CSV file '{path}': {error}") from error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise DataValidationError(
            "CSV is missing required columns: " + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise DataValidationError("CSV contains no data rows.")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_count = int(converted.isna().sum() - dataframe[column].isna().sum())
        if invalid_count > 0:
            raise DataValidationError(
                f"Column '{column}' contains {invalid_count} non-numeric value(s)."
            )
        dataframe[column] = converted

    missing_values = dataframe[REQUIRED_COLUMNS].isna().sum()
    missing_with_values = {
        column: int(count) for column, count in missing_values.items() if count > 0
    }
    if missing_with_values:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_with_values.items()
        )
        raise DataValidationError(f"CSV contains missing required values: {details}")

    invalid_attention = ~dataframe[CLASSIFICATION_TARGET].isin([0, 1])
    if invalid_attention.any():
        raise DataValidationError(
            f"Column '{CLASSIFICATION_TARGET}' must contain only 0 or 1."
        )

    return dataframe


def dataframe_to_features(dataframe: pd.DataFrame) -> np.ndarray:
    """Return the six model input columns in the fixed training order."""

    return dataframe[FEATURE_COLUMNS].to_numpy(dtype=float)


def _json_safe(value: Any) -> Any:
    """Convert NumPy values into values accepted by json.dump."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_data_report(dataframe: pd.DataFrame, csv_path: str | Path, group_code: str) -> dict:
    """Build a dynamic report from the loaded data and original CSV."""

    missing_values = {
        column: int(count)
        for column, count in dataframe[REQUIRED_COLUMNS].isna().sum().items()
    }
    descriptive_columns = [*FEATURE_COLUMNS, REGRESSION_TARGET, CLASSIFICATION_TARGET]
    descriptive_statistics = dataframe[descriptive_columns].describe().to_dict()

    return _json_safe(
        {
            "group_code": group_code,
            "row_count": int(len(dataframe)),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "missing_values": missing_values,
            "duplicate_count": int(dataframe.duplicated().sum()),
            "descriptive_statistics": descriptive_statistics,
            "sha256": calculate_sha256(Path(csv_path)),
        }
    )


def save_data_report(
    dataframe: pd.DataFrame, csv_path: str | Path, output_dir: str | Path, group_code: str
) -> dict:
    """Create the output folder and save data_report.json."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_data_report(dataframe, csv_path, group_code)
    with (output_path / "data_report.json").open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    return report

