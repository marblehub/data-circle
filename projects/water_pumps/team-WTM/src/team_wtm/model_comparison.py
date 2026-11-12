"""Train and compare models with the shared folds (Issue #17).

Run with:
    uv run python -m team_wtm.model_comparison            # four models, 'base'
    uv run python -m team_wtm.model_comparison quantity   # quantity check (RF)
    uv run python -m team_wtm.model_comparison location   # location check (RF)

Reuses team code instead of rewriting it:
- PM-1 (evaluation.py): development folds and score_predictions.
- PM-2 (model_data_preparation.py): build_preprocessor.
- PM-3 (features.py): add_feature_columns and select_features.

Only the development folds are used here. The labelled holdout is not scored.
"""

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from model_data_preparation import build_preprocessor
from team_wtm.cleaning import PROJECT_DIR, clean_data, load_data, merge_data
from team_wtm.evaluation import (
    RANDOM_SEED,
    RESULT_COLUMNS,
    TARGET_CLASSES,
    development_folds,
    score_predictions,
)
from team_wtm.features import FEATURE_OPTIONS, add_feature_columns, select_features

SPLIT_PATH = PROJECT_DIR / "data" / "processed" / "sprint2_splits.csv"
REPORT_DIR = PROJECT_DIR / "reports" / "sprint2"

# Scores summarised over folds for the comparison table.
SUMMARY_METRICS = [
    "macro_f1",
    "accuracy",
    "repair_precision",
    "repair_recall",
    "repair_f1",
    "missed_repairs",
    "false_repair_alarms",
]


# Each entry makes a fresh, unfitted model. A new model is made for every fold.
MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED
    ),
    "logistic_regression_balanced": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "random_forest_balanced": lambda: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
}


def load_training():
    """Join training values and labels by ID. Stop if any IDs do not match."""
    values, labels = load_data()
    training = merge_data(values, labels, keep_merge_indicator=True)
    if not training["_merge"].eq("both").all():
        raise ValueError("Training values and labels must have the same IDs.")
    return training.drop(columns="_merge")


def add_model_columns(training):
    """Apply the fixed cleaning rules and add PM-3's feature columns.

    These rules work row by row and learn nothing from the data, so they are
    safe to apply before splitting. Row order is unchanged.
    """
    return add_feature_columns(clean_data(training))


def categories_as_text(features):
    """Turn True/False values into "True"/"False" text; keep missing values.

    The encoder cannot mix True/False with text such as "__MISSING__".
    This step learns nothing, so it is safe inside cross-validation.
    """
    output = features.copy()
    for column in output.columns:
        is_flag = pd.api.types.is_bool_dtype(output[column])
        if is_flag or not pd.api.types.is_numeric_dtype(output[column]):
            column_values = output[column]
            output[column] = column_values.astype(object).where(
                column_values.isna(), column_values.astype(str)
            )
    return output


def build_pipeline(fit_features, model):
    """Join the preparation steps and the model into one unfitted Pipeline.

    Learned steps (filling missing values, scaling, encoding) are fitted only
    when Pipeline.fit is called, which we do on the fitting rows only.
    """
    return Pipeline(
        steps=[
            ("categories_as_text", FunctionTransformer(categories_as_text)),
            ("preprocessor", build_preprocessor(fit_features)),
            ("model", model),
        ]
    )


def run_cross_validation(training, plan, model_name, feature_option="base"):
    """Fit and score one model on each development fold.

    Returns one result row per fold (shared CSV columns) and the confusion
    matrices, which stay out of the CSV.
    """
    features = select_features(add_model_columns(training), feature_option)
    target = training["status_group"]

    rows = []
    matrices = []
    for fold, fit_rows, validation_rows in development_folds(training, plan):
        fit_features = features.iloc[fit_rows]
        validation_features = features.iloc[validation_rows]

        # A fresh pipeline per fold: nothing learned in one fold reaches the next.
        pipeline = build_pipeline(fit_features, MODELS[model_name]())
        pipeline.fit(fit_features, target.iloc[fit_rows])
        predictions = pipeline.predict(validation_features)

        scores = score_predictions(target.iloc[validation_rows], predictions)
        matrices.append({"fold": fold, "confusion_matrix": scores.pop("confusion_matrix")})
        row = {
            "model": model_name,
            "features": feature_option,
            "fold": fold,
            "train_rows": len(fit_rows),
            "validation_rows": len(validation_rows),
        }
        row.update(scores)
        row["random_seed"] = RANDOM_SEED
        row["notes"] = "Development validation only; holdout not scored."
        rows.append(row)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS), matrices


def summarize_folds(results, metrics=SUMMARY_METRICS):
    """Mean and sample standard deviation (ddof=1) of each score over folds.

    Returns one row per model and feature option, with <metric>_mean and
    <metric>_std columns, plus the number of folds.
    """
    grouped = results.groupby(["model", "features"], sort=False)[metrics]
    means = grouped.mean().add_suffix("_mean")
    spreads = grouped.std(ddof=1).add_suffix("_std")
    summary = pd.concat([means, spreads], axis=1)
    # Put each mean next to its std, in the order of `metrics`.
    ordered = [f"{metric}_{part}" for metric in metrics for part in ("mean", "std")]
    summary = summary[ordered]
    summary.insert(0, "folds", grouped.size())
    return summary


def comparison_table(results, metrics=SUMMARY_METRICS):
    """Put all models side by side, best mean macro F1 first."""
    summary = summarize_folds(results, metrics)
    return summary.sort_values("macro_f1_mean", ascending=False)


def save_reports(results, matrices, name, feature_options):
    """Save fold rows (CSV), the comparison table (CSV) and a JSON summary.

    `matrices` maps "model / features" -> confusion matrices for each fold.
    """
    # 1. One row per fold, in the shared template columns.
    results.to_csv(REPORT_DIR / f"{name}_cv_scores.csv", index=False)

    # 2. Mean and std per model, best macro F1 first.
    table = comparison_table(results)
    table.to_csv(REPORT_DIR / f"{name}_comparison_table.csv")

    # 3. Confusion matrices and settings go in JSON, as in PM-1's report.
    summary = {
        "scope": "Development folds only; the labelled holdout was not scored.",
        "settings": {
            "random_seed": RANDOM_SEED,
            "development_folds": int(results["fold"].nunique()),
            "feature_options": {
                option: FEATURE_OPTIONS[option] for option in feature_options
            },
            "std": "sample standard deviation (ddof=1) over folds",
        },
        "holdout_scored": False,
        "class_order": list(TARGET_CLASSES),
        "confusion_matrix_rows": "actual",
        "confusion_matrix_columns": "predicted",
        "comparison": table.reset_index().to_dict(orient="records"),
        "confusion_matrices": matrices,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    summary_path = REPORT_DIR / f"{name}_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    shown = ["macro_f1_mean", "macro_f1_std", "accuracy_mean", "repair_recall_mean",
             "repair_precision_mean", "missed_repairs_mean", "false_repair_alarms_mean"]
    print(table[shown].round(4).to_string())
    print(f"Saved results to {REPORT_DIR}")
    print("The labelled holdout was not scored.")


def run_models(training, plan):
    """All four models on the 'base' option, next to PM-1's baseline."""
    # Start with PM-1's baseline rows so the table includes the baseline.
    all_rows = [pd.read_csv(REPORT_DIR / "baseline_cv_scores.csv")]
    baseline_summary = json.loads((REPORT_DIR / "baseline_summary.json").read_text())
    matrices = {"majority_baseline / none": baseline_summary["fold_details"]}

    for model_name in MODELS:
        print(f"Training {model_name} on 5 folds ...")
        rows, model_matrices = run_cross_validation(training, plan, model_name)
        all_rows.append(rows)
        matrices[f"{model_name} / base"] = model_matrices
    results = pd.concat(all_rows, ignore_index=True)[RESULT_COLUMNS]
    save_reports(results, matrices, "model", ["base"])


QUANTITY_MODELS = ["random_forest", "random_forest_balanced"]
QUANTITY_OPTIONS = ["full", "full_no_quantity"]


def run_quantity_check(training, plan):
    """Random Forests on PM-3's 'full' and 'full_no_quantity' options.

    'full' vs 'full_no_quantity' differ only in quantity/quantity_group.
    The 'base' rows are reused from the model run, on the same folds.
    """
    base_rows = pd.read_csv(REPORT_DIR / "model_cv_scores.csv")
    base_rows = base_rows[base_rows["model"].isin(QUANTITY_MODELS)]
    base_summary = json.loads((REPORT_DIR / "model_comparison_summary.json").read_text())
    all_rows = [base_rows]
    matrices = {
        f"{model} / base": base_summary["confusion_matrices"][f"{model} / base"]
        for model in QUANTITY_MODELS
    }

    for option in QUANTITY_OPTIONS:
        for model_name in QUANTITY_MODELS:
            print(f"Training {model_name} with '{option}' on 5 folds ...")
            rows, model_matrices = run_cross_validation(
                training, plan, model_name, feature_option=option
            )
            all_rows.append(rows)
            matrices[f"{model_name} / {option}"] = model_matrices
    results = pd.concat(all_rows, ignore_index=True)[RESULT_COLUMNS]
    save_reports(results, matrices, "quantity", ["base", *QUANTITY_OPTIONS])


def run_location_check(training, plan):
    """RF balanced on 'base_age_missing' ('full' without the location group).

    The 'base' and 'full' rows are reused from the quantity check, same folds.
    """
    model_name = "random_forest_balanced"
    earlier = pd.read_csv(REPORT_DIR / "quantity_cv_scores.csv")
    earlier = earlier[earlier["model"].eq(model_name)
                      & earlier["features"].isin(["base", "full"])]
    earlier_summary = json.loads(
        (REPORT_DIR / "quantity_comparison_summary.json").read_text()
    )
    matrices = {
        f"{model_name} / {option}":
            earlier_summary["confusion_matrices"][f"{model_name} / {option}"]
        for option in ["base", "full"]
    }

    print(f"Training {model_name} with 'base_age_missing' on 5 folds ...")
    rows, model_matrices = run_cross_validation(
        training, plan, model_name, feature_option="base_age_missing"
    )
    matrices[f"{model_name} / base_age_missing"] = model_matrices
    results = pd.concat([earlier, rows], ignore_index=True)[RESULT_COLUMNS]
    save_reports(results, matrices, "location", ["base", "base_age_missing", "full"])


EXPERIMENTS = {
    "models": run_models,
    "quantity": run_quantity_check,
    "location": run_location_check,
}


def main():
    parser = argparse.ArgumentParser(description="Issue #17 model comparisons.")
    parser.add_argument(
        "experiment", nargs="?", default="models", choices=list(EXPERIMENTS),
        help="models: four models on 'base'. quantity / location: RF feature checks.",
    )
    experiment = parser.parse_args().experiment

    training = load_training()
    # Reuse PM-1's saved split plan so every model sees the same folds.
    plan = pd.read_csv(SPLIT_PATH)
    EXPERIMENTS[experiment](training, plan)


if __name__ == "__main__":
    main()
