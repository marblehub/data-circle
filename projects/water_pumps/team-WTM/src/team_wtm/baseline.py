"""Run the majority-class baseline with python -m team_wtm.baseline."""

import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from team_wtm.cleaning import PROJECT_DIR, load_data, merge_data
from team_wtm.evaluation import (
    CLASS_NAMES,
    N_FOLDS,
    RANDOM_SEED,
    RESULT_COLUMNS,
    TARGET_CLASSES,
    development_folds,
    make_split_plan,
    score_predictions,
)

REPORT_DIR = PROJECT_DIR / "reports" / "sprint2"
SPLIT_PATH = PROJECT_DIR / "data" / "processed" / "sprint2_splits.csv"


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


def main():
    # 1. Join training values and labels by ID. Stop if any IDs do not match.
    values, labels = load_data()
    training = merge_data(values, labels, keep_merge_indicator=True)
    if not training["_merge"].eq("both").all():
        raise ValueError(
            "Training values and labels must contain exactly the same IDs."
        )
    training = training.drop(columns="_merge")

    # 2. Make the shared splits and score only the development folds.
    plan = make_split_plan(training)
    results, fold_details = run_baseline(training, plan)

    # 3. Keep ID-level assignments local and save aggregate scores for the team.
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(SPLIT_PATH, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "baseline_cv_scores.csv", index=False)

    # Keep any changes teammates have made to the comparison template.
    template_path = REPORT_DIR / "model_comparison_template.csv"
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
    summary_path = REPORT_DIR / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved {len(results)} development-fold results to {REPORT_DIR}")
    print(f"Local split plan: {SPLIT_PATH}")
    print(f"Accuracy: {mean_scores['accuracy']:.4f}")
    print(f"Macro F1: {mean_scores['macro_f1']:.4f}")
    print("The labelled holdout was not scored. The supplied test file was not loaded.")


if __name__ == "__main__":
    main()
