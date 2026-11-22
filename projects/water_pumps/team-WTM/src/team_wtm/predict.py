"""Predict pump status for new records with the saved final model.

Run with (from the team-WTM folder):
    python -m team_wtm.predict
    python -m team_wtm.predict --records path/to/records.csv --output out.csv
    python -m team_wtm.predict --format json --output out.json

This module only predicts. It never fits or retrains the model and never
reads the labelled holdout. It does not depend on Streamlit, so the app and
the command line can share it.

The default run (no --records) is the completion check: it predicts the
DrivenData test records, compares the result with SubmissionFormat.csv,
checks that a reloaded model gives identical predictions and writes a
totals-only report (no individual records) to reports/sprint2.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from team_wtm.cleaning import PROJECT_DIR, RAW_DATA_DIR
from team_wtm.evaluation import TARGET_CLASSES
from team_wtm.features import FEATURE_OPTIONS, select_features
from team_wtm.final_model import (
    FINAL_FEATURES,
    FINAL_MODEL,
    FINAL_MODEL_REPORT,
    MODEL_PATH,
    versions,
)
from team_wtm.model_comparison import REPORT_DIR, add_model_columns

ID_COLUMN = "id"
TARGET_COLUMN = "status_group"
OUTPUT_COLUMNS = [ID_COLUMN, TARGET_COLUMN]
ALLOWED_LABELS = set(TARGET_CLASSES)

# The test file has had several names; the first one found is used.
RECORD_FILE_NAMES = [
    "test_features.csv.csv",
    "Test_set_values.csv",
    "test_features.csv",
]
SUBMISSION_PATH = RAW_DATA_DIR / "SubmissionFormat.csv"
# data/**/*.csv is ignored by Git, so predictions stay local.
PREDICTIONS_PATH = PROJECT_DIR / "data" / "processed" / "test_predictions.csv"
CHECK_REPORT_PATH = REPORT_DIR / "prediction_check.json"

# Package versions compared with the ones saved when the model was trained.
CHECKED_PACKAGES = ["pandas", "numpy", "scikit_learn", "joblib"]


def find_records_path(raw_dir=RAW_DATA_DIR):
    """Return the first test records file that exists in raw_dir."""
    for name in RECORD_FILE_NAMES:
        path = Path(raw_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No test records found in {raw_dir}; tried {RECORD_FILE_NAMES}"
    )


def check_versions(report_path=FINAL_MODEL_REPORT):
    """Warn for each package whose version differs from training time.

    Returns the list of warning messages (empty when all versions match).
    """
    report_path = Path(report_path)
    if not report_path.exists():
        message = f"No training report at {report_path}; package versions not checked."
        warnings.warn(message, stacklevel=2)
        return [message]
    trained = json.loads(report_path.read_text())["versions"]
    current = versions()
    messages = []
    for package in CHECKED_PACKAGES:
        if trained.get(package) != current[package]:
            message = (
                f"{package} version differs: model trained with "
                f"{trained.get(package)}, now {current[package]}."
            )
            warnings.warn(message, stacklevel=2)
            messages.append(message)
    return messages


def load_model(path=MODEL_PATH, report_path=FINAL_MODEL_REPORT):
    """Load the saved final pipeline and warn about package version changes.

    Pass report_path=None to skip the version check (already done).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run 'python -m team_wtm.final_model' first."
        )
    if report_path is not None:
        check_versions(report_path)
    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "feature_names_in_"):
        raise TypeError(f"{path} does not contain a fitted pipeline.")
    return model


def validate_records(records):
    """Stop early if the records cannot be used for prediction."""
    if not isinstance(records, pd.DataFrame):
        raise TypeError(
            f"Records must be a pandas DataFrame, not {type(records).__name__}."
        )
    if records.empty:
        raise ValueError("Records are empty; there is nothing to predict.")
    if TARGET_COLUMN in records.columns:
        # The target must never be passed in, even by accident.
        raise ValueError(
            f"Records must not contain the target column '{TARGET_COLUMN}'."
        )
    if ID_COLUMN not in records.columns:
        raise ValueError(f"Records must contain an '{ID_COLUMN}' column.")
    if records[ID_COLUMN].isna().any():
        raise ValueError("Records contain missing ids.")
    if records[ID_COLUMN].duplicated().any():
        raise ValueError("Records contain duplicate ids.")


def check_feature_columns(features, expected_columns):
    """Refuse features that are not exactly the model's columns, in order."""
    expected = list(expected_columns)
    actual = list(features.columns)
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        raise ValueError(
            f"Feature columns do not match the model. "
            f"Missing: {missing}. Unexpected: {extra}."
        )
    if actual != expected:
        raise ValueError(
            f"Feature columns are in the wrong order. Expected: {expected}."
        )


def prepare_features(records, expected_columns):
    """Clean the records and build the model's features in the trained order.

    Reuses the shared steps (clean_data, add_feature_columns, select_features)
    so prediction matches training exactly. Rows keep their original order.
    """
    expected = list(expected_columns)
    if sorted(expected) != sorted(FEATURE_OPTIONS[FINAL_FEATURES]):
        raise ValueError(
            f"The model does not use the '{FINAL_FEATURES}' features "
            f"of {FINAL_MODEL}: {expected}"
        )
    features = select_features(add_model_columns(records), FINAL_FEATURES)
    # Put the columns in the order the model was trained with.
    features = features[expected]
    check_feature_columns(features, expected)
    return features


def validate_predictions(predictions, records):
    """Check one valid label per record, with ids in the original order."""
    if list(predictions.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {OUTPUT_COLUMNS}.")
    if len(predictions) != len(records):
        raise ValueError("There must be exactly one prediction per record.")
    if not np.array_equal(
        predictions[ID_COLUMN].to_numpy(), records[ID_COLUMN].to_numpy()
    ):
        raise ValueError("Prediction ids are not the record ids in the same order.")
    if predictions.isna().any().any():
        raise ValueError("Predictions contain missing values.")
    unknown = set(predictions[TARGET_COLUMN]) - ALLOWED_LABELS
    if unknown:
        raise ValueError(f"Unknown predicted labels: {sorted(unknown)}")


def predict_records(records, model):
    """Return a DataFrame with columns id, status_group: one row per record.

    Only model.predict is called; the model is never fitted here.
    """
    validate_records(records)
    features = prepare_features(records, model.feature_names_in_)
    labels = model.predict(features)
    predictions = pd.DataFrame(
        {
            ID_COLUMN: records[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: labels,
        }
    )
    validate_predictions(predictions, records)
    return predictions


def compare_with_submission(predictions, submission):
    """Compare predictions with SubmissionFormat.csv. Returns totals only."""
    return {
        "rows_match": len(predictions) == len(submission),
        "ids_match_in_order": bool(
            np.array_equal(
                predictions[ID_COLUMN].to_numpy(), submission[ID_COLUMN].to_numpy()
            )
        ),
        "columns_match": list(predictions.columns) == list(submission.columns),
    }


def save_predictions(predictions, path, file_format="csv"):
    """Write predictions as CSV or JSON (a list of {id, status_group})."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        predictions.to_csv(path, index=False)
    elif file_format == "json":
        predictions.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {file_format}. Use 'csv' or 'json'.")
    return path


def relative(path):
    """Show a path relative to the project if possible (forward slashes)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.name


def run_completion_check(
    records_path,
    model_path,
    output_path,
    file_format,
    submission_path=SUBMISSION_PATH,
    report_path=CHECK_REPORT_PATH,
):
    """Predict the test records, run every check and write a totals-only report."""
    records = pd.read_csv(records_path)
    submission = pd.read_csv(submission_path)
    version_warnings = check_versions()

    model = load_model(model_path, report_path=None)
    predictions = predict_records(records, model)
    # Load the model file a second time and check the labels are identical.
    reloaded = load_model(model_path, report_path=None)
    same_after_reload = predictions.equals(predict_records(records, reloaded))

    checks = compare_with_submission(predictions, submission)
    checks["labels_valid"] = set(predictions[TARGET_COLUMN]) <= ALLOWED_LABELS
    checks["no_missing_values"] = not predictions.isna().any().any()
    checks["same_predictions_after_reload"] = bool(same_after_reload)

    report = {
        "scope": "Predictions on unlabelled test records only; no fitting, "
        "no holdout scoring. Totals only, no individual records.",
        "model": FINAL_MODEL,
        "features": FINAL_FEATURES,
        "model_file": relative(model_path),
        "records_file": relative(records_path),
        "submission_file": relative(submission_path),
        "records_rows": len(records),
        "records_columns": records.shape[1],
        "prediction_rows": len(predictions),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "label_counts": {
            label: int((predictions[TARGET_COLUMN] == label).sum())
            for label in TARGET_CLASSES
        },
        "version_warnings": version_warnings,
        "versions": versions(),
    }
    save_predictions(predictions, output_path, file_format)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict pump status with the final model."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="CSV of records without status_group. "
        "Default: the test records in data/raw (runs the completion check).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Saved model file (default: models/final_pipeline.joblib).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PREDICTIONS_PATH,
        help="Where to write predictions "
        "(default: data/processed/test_predictions.csv).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output file format (default: csv).",
    )
    args = parser.parse_args(argv)

    if args.records is None:
        report = run_completion_check(
            find_records_path(), args.model, args.output, args.format
        )
        print(f"Predicted {report['prediction_rows']} records.")
        for name, passed in report["checks"].items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        print(f"Label counts: {report['label_counts']}")
        print(f"Saved predictions to {relative(args.output)}")
        print(f"Saved totals-only report to {relative(CHECK_REPORT_PATH)}")
        if not report["all_checks_passed"]:
            raise SystemExit("Completion check failed.")
        return

    records = pd.read_csv(args.records)
    predictions = predict_records(records, load_model(args.model))
    save_predictions(predictions, args.output, args.format)
    print(f"Predicted {len(predictions)} records; saved to {args.output}")


if __name__ == "__main__":
    main()
