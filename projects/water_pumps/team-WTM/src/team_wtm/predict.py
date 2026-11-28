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

import hashlib
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from team_wtm.artifacts import read_json, sha256_file, write_json
from team_wtm.cleaning import RAW_DATA_DIR
from team_wtm.config import SETTINGS
from team_wtm.evaluation import TARGET_CLASSES
from team_wtm.features import FEATURE_OPTIONS, select_features
from team_wtm.final_model import (
    FINAL_FEATURES,
    FINAL_MODEL_REPORT,
    MODEL_PATH,
    versions,
)
from team_wtm.model_comparison import MODELS, REPORT_DIR, add_model_columns

ID_COLUMN = "id"
TARGET_COLUMN = "status_group"
OUTPUT_COLUMNS = [ID_COLUMN, TARGET_COLUMN]
ALLOWED_LABELS = set(TARGET_CLASSES)

# The test file has had several names; the first one found is used.
RECORD_FILE_NAMES = SETTINGS.test_records

SUBMISSION_PATH = SETTINGS.submission_format
# data/**/*.csv is ignored by Git, so predictions stay local.
PREDICTIONS_PATH = SETTINGS.predictions_file
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
    return _version_warnings(read_json(report_path)["versions"])


def _version_warnings(trained):
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


def _load_model_details(path, report_path=None):
    """Read the model and its identity together, using the same bytes for verification."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run 'python -m team_wtm.final_model' first."
        )
    if report_path is None:
        if path.resolve() == MODEL_PATH.resolve():
            report_path = FINAL_MODEL_REPORT
        elif path.with_suffix(".json").is_file():
            report_path = path.with_suffix(".json")
    manifest = read_json(report_path) if report_path is not None else None
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest is not None and manifest.get("model_sha256") != digest:
            raise ValueError(f"Model checksum missing or different in {report_path}.")
        stream.seek(0)
        model = joblib.load(stream)
    if not hasattr(model, "predict") or not hasattr(model, "feature_names_in_"):
        raise TypeError(f"{path} does not contain a fitted pipeline.")

    columns = list(model.feature_names_in_)
    estimator = getattr(model, "named_steps", {}).get("model", model)
    if manifest is None:
        # A standalone custom model has no recorded run or training environment.
        identity = {
            "run_id": None,
            "model": type(estimator).__name__,
            "features": _feature_option(columns),
            "model_sha256": digest,
        }
        return model, identity, []

    settings = manifest.get("settings", {})
    if not isinstance(settings, dict) or not isinstance(manifest.get("versions"), dict):
        raise TypeError("Model manifest requires settings and training versions.")
    model_name, option = settings.get("model"), settings.get("features")
    if not isinstance(model_name, str) or model_name not in MODELS:
        raise ValueError("Unknown model in manifest.")
    if not isinstance(option, str) or option not in FEATURE_OPTIONS:
        raise ValueError("Unknown feature option in manifest.")
    if settings.get("feature_columns") != columns or sorted(columns) != sorted(
        FEATURE_OPTIONS[option]
    ):
        raise ValueError("Manifest columns differ from the pipeline or feature option.")
    registered = MODELS[model_name]()
    if (
        type(estimator) is not type(registered)
        or estimator.get_params() != settings.get("model_settings")
        or estimator.get_params().get("class_weight")
        != registered.get_params().get("class_weight")
    ):
        raise ValueError("Manifest estimator settings differ from the saved model.")
    if "random_seed" in settings and settings[
        "random_seed"
    ] != estimator.get_params().get("random_state"):
        raise ValueError("Manifest seed differs from the saved estimator.")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ValueError("Model manifest requires a run_id.")
    identity = {
        "run_id": manifest["run_id"],
        "model": model_name,
        "features": option,
        "model_sha256": digest,
    }
    return model, identity, _version_warnings(manifest["versions"])


def load_model(path=MODEL_PATH, report_path=None):
    """Load a trusted pipeline; verify the configured or accompanying model manifest."""
    model, _, _ = _load_model_details(path, report_path)
    return model


def _feature_option(columns):
    for name, group in FEATURE_OPTIONS.items():
        if sorted(columns) == sorted(group):
            return name
    raise ValueError("The model does not use a supported feature option.")


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


def prepare_features(records, expected_columns, feature_option=FINAL_FEATURES):
    """Clean the records and build the model's features in the trained order.

    Reuses the shared steps (clean_data, add_feature_columns, select_features)
    so prediction matches training exactly. Rows keep their original order.
    """
    expected = list(expected_columns)
    if sorted(expected) != sorted(FEATURE_OPTIONS[feature_option]):
        raise ValueError(
            f"The model does not use the '{feature_option}' features: {expected}"
        )
    features = select_features(add_model_columns(records), feature_option)
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
    columns = list(model.feature_names_in_)
    features = prepare_features(records, columns, _feature_option(columns))
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
        return path.relative_to(SETTINGS.project_root).as_posix()
    except ValueError:
        return path.name


def run_completion_check(
    records_path,
    model_path,
    output_path,
    file_format,
    submission_path=SUBMISSION_PATH,
    report_path=None,
    *,
    model_report=None,
):
    """Predict the test records, run every check and write a totals-only report."""
    if report_path is None:
        if Path(model_path).resolve() == MODEL_PATH.resolve():
            report_path = CHECK_REPORT_PATH
        else:
            report_path = Path(str(output_path) + ".check.json")
    records = pd.read_csv(records_path)
    submission = pd.read_csv(submission_path)
    model, identity, version_warnings = _load_model_details(model_path, model_report)
    predictions = predict_records(records, model)
    # Load the model file a second time and check the labels are identical.
    reloaded, reloaded_identity, _ = _load_model_details(model_path, model_report)
    if identity != reloaded_identity:
        raise ValueError("Model identity changed during the completion check.")
    same_after_reload = predictions.equals(predict_records(records, reloaded))

    checks = compare_with_submission(predictions, submission)
    checks["labels_valid"] = set(predictions[TARGET_COLUMN]) <= ALLOWED_LABELS
    checks["no_missing_values"] = not predictions.isna().any().any()
    checks["same_predictions_after_reload"] = bool(same_after_reload)

    report = {
        "scope": "Predictions on unlabelled test records only; no fitting, "
        "no holdout scoring. Totals only, no individual records.",
        **identity,
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
    report["predictions_sha256"] = sha256_file(Path(output_path))
    report["report_file"] = relative(report_path)
    write_json(report_path, report)
    return report


def run_prediction(
    records_path=None,
    model_path=MODEL_PATH,
    output_path=PREDICTIONS_PATH,
    file_format="csv",
    *,
    model_report=None,
):
    """Write predictions and return their count, plus checks for supplied records."""
    if records_path is None:
        return run_completion_check(
            find_records_path(),
            model_path,
            output_path,
            file_format,
            model_report=model_report,
        )

    records = pd.read_csv(records_path)
    predictions = predict_records(records, load_model(model_path, model_report))
    save_predictions(predictions, output_path, file_format)
    return {"prediction_rows": len(predictions)}


def main(argv=None):
    from team_wtm.cli import main as run_cli

    run_cli(["predict", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
