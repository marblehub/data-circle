"""Run the majority-class baseline with python -m team_wtm.baseline."""

import json
import logging
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from team_wtm.artifacts import sha256_file
from team_wtm.config import SETTINGS
from team_wtm.data.preparation import load_training_data
from team_wtm.evaluation.splits import (
    CLASS_NAMES,
    N_FOLDS,
    RANDOM_SEED,
    RESULT_COLUMNS,
    TARGET_CLASSES,
    development_folds,
    make_split_plan,
    score_predictions,
)
from team_wtm.runtime import track_workflow

logger = logging.getLogger(__name__)

REPORT_DIR = SETTINGS.report_dir
SPLIT_PATH = SETTINGS.split_plan


def run_baseline(training, plan):
    """Learn the most common class separately in each training fold."""
    rows = []
    fold_details = []

    for fold, fit_rows, validation_rows in development_folds(training, plan):
        fit_labels = training.iloc[fit_rows]["status_group"]
        validation_labels = training.iloc[validation_rows]["status_group"]

        # This model ignores features. The zeros only tell it how many rows to use.
        fit_features = np.zeros((len(fit_rows), 1))
        validation_features = np.zeros((len(validation_rows), 1))
        model = DummyClassifier(strategy="most_frequent")
        model.fit(fit_features, fit_labels)
        predictions = model.predict(validation_features)

        scores = score_predictions(validation_labels, predictions)
        # Save the matrix in JSON; keep individual scores in the CSV table.
        matrix = scores.pop("confusion_matrix")
        row = {
            "model": "majority_baseline",
            "features": "none",
            "fold": fold,
            "train_rows": len(fit_rows),
            "validation_rows": len(validation_rows),
        }
        row.update(scores)
        row["random_seed"] = RANDOM_SEED
        row["notes"] = "Development validation only; holdout not scored."
        rows.append(row)

        fold_details.append(
            {
                "fold": fold,
                "predicted_class": str(predictions[0]),
                "confusion_matrix": matrix,
            }
        )

    if not rows:
        raise ValueError("No development folds were available.")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results, fold_details


def _write_baseline_reports():
    # 1. Join training values and labels by ID. Stop if any IDs do not match.
    training = load_training_data()

    # 2. Make the shared splits and score only the development folds.
    plan = make_split_plan(training)
    results, fold_details = run_baseline(training, plan)

    # 3. Keep ID-level assignments local and save aggregate scores for the team.
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(SPLIT_PATH, index=False)
    (REPORT_DIR / "baseline").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "comparisons").mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "baseline/baseline_cv_scores.csv", index=False)

    # Keep any changes teammates have made to the comparison template.
    template_path = REPORT_DIR / "comparisons/model_comparison_template.csv"
    if not template_path.exists():
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(template_path, index=False)

    # 4. Calculate the average and spread of scores across the five folds.
    metric_columns = ["accuracy", "macro_f1"]
    for class_name in CLASS_NAMES.values():
        for metric in ["precision", "recall", "f1"]:
            metric_columns.append(f"{class_name}_{metric}")

    mean_scores = {}
    score_spread = {}
    for column in metric_columns:
        mean_scores[column] = float(results[column].mean())
        score_spread[column] = float(results[column].std(ddof=1))

    summary = {
        "scope": "Recorded pump condition; not future failure prediction.",
        "settings": {
            "random_seed": RANDOM_SEED,
            "outer_folds": N_FOLDS,
            "development_folds": N_FOLDS,
            "holdout_outer_fold": 0,
        },
        "rows": {
            "all": len(plan),
            "development": int(plan["split"].eq("development").sum()),
            "holdout": int(plan["split"].eq("holdout").sum()),
        },
        "feature_groups": int(plan["group"].nunique()),
        "repeated_feature_rows": int(len(plan) - plan["group"].nunique()),
        "holdout_scored": False,
        "supplied_test_file_loaded": False,
        "class_order": list(TARGET_CLASSES),
        "confusion_matrix_rows": "actual",
        "confusion_matrix_columns": "predicted",
        "cv_mean": mean_scores,
        "cv_std": score_spread,
        "fold_details": fold_details,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    summary_path = REPORT_DIR / "baseline/baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def run_baseline_workflow(*, on_progress=None):
    """Create the split plan and baseline reports and return their summary."""
    with track_workflow(
        "baseline", config=SETTINGS, on_progress=on_progress, report_dir=REPORT_DIR
    ) as run:
        run.record_inputs(
            {
                "training_values": SETTINGS.training_values,
                "training_labels": SETTINGS.training_labels,
            }
        )
        run.progress("Creating the split plan and majority-class baseline")
        summary = _write_baseline_reports()
        run.details["metrics"] = summary["cv_mean"]
        run.archive(
            [
                REPORT_DIR / "baseline/baseline_cv_scores.csv",
                REPORT_DIR / "baseline/baseline_summary.json",
            ],
            REPORT_DIR,
        )
        run.details["split_plan"] = {
            "path": str(SPLIT_PATH),
            "sha256": sha256_file(SPLIT_PATH),
        }
        logger.info("Saved baseline results to %s", REPORT_DIR / "baseline")
        logger.info("Saved split plan to %s", SPLIT_PATH)
        logger.info(
            "Baseline accuracy %.4f; macro F1 %.4f",
            summary["cv_mean"]["accuracy"],
            summary["cv_mean"]["macro_f1"],
        )
        return summary


def main(argv=None):
    from team_wtm.cli import main as run_cli

    run_cli(["baseline", *([] if argv is None else argv)])


if __name__ == "__main__":
    main(sys.argv[1:])
