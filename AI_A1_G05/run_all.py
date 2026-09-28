"""Command-line entry point for the complete assignment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classification import run_classification
from src.clustering import run_clustering
from src.data_pipeline import (
    MODEL_VERSION,
    RANDOM_SEED,
    load_and_validate_csv,
    save_data_report,
)
from src.regression import run_regression


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SWE 3513 AI analysis pipeline.")
    parser.add_argument("--data", required=True, help="Path to the lecturer-provided CSV.")
    parser.add_argument(
        "--output", default="artifacts", help="Folder for generated reports, plots, and CSV files."
    )
    parser.add_argument(
        "--models", default="models", help="Folder for reusable trained models."
    )
    parser.add_argument("--group", required=True, help="Group code, for example AI-G03.")
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})."
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_dir = Path(arguments.output)
    models_dir = Path(arguments.models)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_and_validate_csv(arguments.data)
    report = save_data_report(dataframe, arguments.data, output_dir, arguments.group)
    print(f"Group code: {arguments.group}")
    print(f"Rows loaded: {report['row_count']}")
    print(f"Original CSV SHA-256: {report['sha256']}")

    regression_metrics = run_regression(dataframe, output_dir, models_dir, arguments.seed)
    print(
        "Regression complete: "
        f"MAE={regression_metrics['MAE']:.3f}, "
        f"RMSE={regression_metrics['RMSE']:.3f}, "
        f"R2={regression_metrics['R2']}"
    )

    classification_metrics = run_classification(
        dataframe, output_dir, models_dir, arguments.seed
    )
    print(
        "Classification complete: "
        f"accuracy={classification_metrics['accuracy']:.3f}, "
        f"F1={classification_metrics['f1_score']:.3f}"
    )

    clustering_metrics = run_clustering(dataframe, output_dir, models_dir, arguments.seed)
    print(f"Clustering complete: selected k={clustering_metrics['selected_k']}")

    metadata = {
        "group_code": arguments.group,
        "model_version": MODEL_VERSION,
        "random_seed": arguments.seed,
    }
    with (models_dir / "model_metadata.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    print("\nPipeline complete.")
    print(f"Artifacts: {output_dir}")
    print(f"Models: {models_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"Pipeline error: {error}")
        raise SystemExit(2)

