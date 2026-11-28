"""Train and compare models with the shared folds (Issue #17).

Run with:
    uv run python -m team_wtm.model_comparison            # four models, 'base'
    uv run python -m team_wtm.model_comparison quantity   # quantity check (RF)
    uv run python -m team_wtm.model_comparison location   # location check (RF)

Reuses team code instead of rewriting it:
- PM-1 (evaluation.py): development folds and score_predictions.
- PM-2 (model_data_preparation.py): training join, preparation and pipeline builder.
- PM-3 (features.py): add_feature_columns and select_features.

Only the development folds are used here. The labelled holdout is not scored.
"""

import json
import logging
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from team_wtm.config import SETTINGS
from team_wtm.data import preparation
from team_wtm.evaluation.splits import (
    RANDOM_SEED,
    RESULT_COLUMNS,
    TARGET_CLASSES,
    development_folds,
    score_predictions,
)
from team_wtm.modeling import preprocessing
from team_wtm.modeling.estimators import MODELS
from team_wtm.modeling.features import FEATURE_OPTIONS, select_features
from team_wtm.runtime import track_workflow

logger = logging.getLogger(__name__)

SPLIT_PATH = SETTINGS.split_plan
REPORT_DIR = SETTINGS.report_dir

# Existing notebooks and saved pipelines refer to these names in this module.
load_training = preparation.load_training_data
add_model_columns = preparation.prepare_features
categories_as_text = preprocessing.categories_as_text
build_pipeline = preprocessing.build_pipeline

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


def run_cross_validation(
    training, plan, model_name, feature_option="base", *, on_progress=logger.info
):
    """Fit and score one model on each development fold.

    Returns one result row per fold (shared CSV columns) and the confusion
    matrices, which stay out of the CSV.
    """
    features = select_features(add_model_columns(training), feature_option)
    target = training["status_group"]

    rows = []
    matrices = []
    for fold, fit_rows, validation_rows in development_folds(training, plan):
        on_progress(f"Fitting {model_name}, {feature_option}, development fold {fold}")
        fit_features = features.iloc[fit_rows]
        validation_features = features.iloc[validation_rows]

        # A fresh pipeline per fold: nothing learned in one fold reaches the next.
        pipeline = build_pipeline(fit_features, MODELS[model_name]())
        pipeline.fit(fit_features, target.iloc[fit_rows])
        predictions = pipeline.predict(validation_features)

        scores = score_predictions(target.iloc[validation_rows], predictions)
        matrices.append(
            {"fold": fold, "confusion_matrix": scores.pop("confusion_matrix")}
        )
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


def save_reports(results, matrices, name, feature_options, *, folder="comparisons"):
    """Save fold rows (CSV), the comparison table (CSV) and a JSON summary.

    `matrices` maps "model / features" -> confusion matrices for each fold.
    """
    directory = REPORT_DIR / folder
    directory.mkdir(parents=True, exist_ok=True)

    # 1. One row per fold, in the shared template columns.
    results.to_csv(directory / f"{name}_cv_scores.csv", index=False)

    # 2. Mean and std per model, best macro F1 first.
    table = comparison_table(results)
    table.to_csv(directory / f"{name}_comparison_table.csv")

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
    summary_path = directory / f"{name}_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return table


def run_models(training, plan, *, on_progress=logger.info):
    """All four models on the 'base' option, next to PM-1's baseline."""
    # Start with PM-1's baseline rows so the table includes the baseline.
    all_rows = [pd.read_csv(REPORT_DIR / "baseline/baseline_cv_scores.csv")]
    baseline_summary = json.loads(
        (REPORT_DIR / "baseline/baseline_summary.json").read_text()
    )
    matrices = {"majority_baseline / none": baseline_summary["fold_details"]}

    for model_name in SETTINGS.candidates:
        on_progress(f"Training {model_name} on the shared folds ...")
        rows, model_matrices = run_cross_validation(
            training, plan, model_name, on_progress=on_progress
        )
        all_rows.append(rows)
        matrices[f"{model_name} / base"] = model_matrices
    results = pd.concat(all_rows, ignore_index=True)[RESULT_COLUMNS]
    return save_reports(results, matrices, "model", ["base"])


QUANTITY_MODELS = ["random_forest", "random_forest_balanced"]
QUANTITY_OPTIONS = ["full", "full_no_quantity"]


def run_quantity_check(training, plan, *, on_progress=logger.info):
    """Random Forests on PM-3's 'full' and 'full_no_quantity' options.

    'full' vs 'full_no_quantity' differ only in quantity/quantity_group.
    The 'base' rows are reused from the model run, on the same folds.
    """
    base_rows = pd.read_csv(REPORT_DIR / "comparisons/model_cv_scores.csv")
    base_rows = base_rows[base_rows["model"].isin(QUANTITY_MODELS)]
    base_summary = json.loads(
        (REPORT_DIR / "comparisons/model_comparison_summary.json").read_text()
    )
    all_rows = [base_rows]
    missing = set(QUANTITY_MODELS) - set(base_rows["model"])
    if missing:
        raise ValueError(
            f"Missing base comparison for {sorted(missing)}. Include both Random Forest "
            "variants in comparison.candidates and run 'team-wtm compare' first."
        )
    matrices = {
        f"{model} / base": base_summary["confusion_matrices"][f"{model} / base"]
        for model in QUANTITY_MODELS
    }

    for option in QUANTITY_OPTIONS:
        for model_name in QUANTITY_MODELS:
            on_progress(
                f"Training {model_name} with '{option}' on the shared folds ..."
            )
            rows, model_matrices = run_cross_validation(
                training,
                plan,
                model_name,
                feature_option=option,
                on_progress=on_progress,
            )
            all_rows.append(rows)
            matrices[f"{model_name} / {option}"] = model_matrices
    results = pd.concat(all_rows, ignore_index=True)[RESULT_COLUMNS]
    return save_reports(results, matrices, "quantity", ["base", *QUANTITY_OPTIONS])


def run_location_check(training, plan, *, on_progress=logger.info):
    """RF balanced on 'base_age_missing' ('full' without the location group).

    The 'base' and 'full' rows are reused from the quantity check, same folds.
    """
    model_name = "random_forest_balanced"
    earlier = pd.read_csv(REPORT_DIR / "comparisons/quantity_cv_scores.csv")
    earlier = earlier[
        earlier["model"].eq(model_name) & earlier["features"].isin(["base", "full"])
    ]
    earlier_summary = json.loads(
        (REPORT_DIR / "comparisons/quantity_comparison_summary.json").read_text()
    )
    matrices = {
        f"{model_name} / {option}": earlier_summary["confusion_matrices"][
            f"{model_name} / {option}"
        ]
        for option in ["base", "full"]
    }

    on_progress(
        f"Training {model_name} with 'base_age_missing' on the shared folds ..."
    )
    rows, model_matrices = run_cross_validation(
        training,
        plan,
        model_name,
        feature_option="base_age_missing",
        on_progress=on_progress,
    )
    matrices[f"{model_name} / base_age_missing"] = model_matrices
    results = pd.concat([earlier, rows], ignore_index=True)[RESULT_COLUMNS]
    return save_reports(
        results, matrices, "location", ["base", "base_age_missing", "full"]
    )


EXPERIMENTS = {
    "models": run_models,
    "quantity": run_quantity_check,
    "location": run_location_check,
}


def run_comparison(experiment="models", *, on_progress=None):
    """Archive each development comparison and return its ranked table."""
    with track_workflow(
        f"compare-{experiment}",
        config=SETTINGS,
        on_progress=on_progress,
        report_dir=REPORT_DIR,
    ) as run:
        unknown = set(SETTINGS.candidates).difference(MODELS)
        if unknown:
            raise ValueError(f"Unknown comparison candidates: {sorted(unknown)}")
        dependencies = {
            "models": (
                "baseline/baseline_cv_scores.csv",
                "baseline/baseline_summary.json",
            ),
            "quantity": (
                "comparisons/model_cv_scores.csv",
                "comparisons/model_comparison_summary.json",
            ),
            "location": (
                "comparisons/quantity_cv_scores.csv",
                "comparisons/quantity_comparison_summary.json",
            ),
        }
        run.record_inputs(
            {
                "training_values": SETTINGS.training_values,
                "training_labels": SETTINGS.training_labels,
                "split_plan": SPLIT_PATH,
                **{name: REPORT_DIR / name for name in dependencies[experiment]},
            }
        )
        training = load_training()
        plan = pd.read_csv(SPLIT_PATH)
        table = EXPERIMENTS[experiment](training, plan, on_progress=run.progress)
        prefix = {"models": "model", "quantity": "quantity", "location": "location"}[
            experiment
        ]
        run.archive(
            [
                REPORT_DIR / f"comparisons/{prefix}_cv_scores.csv",
                REPORT_DIR / f"comparisons/{prefix}_comparison_table.csv",
                REPORT_DIR / f"comparisons/{prefix}_comparison_summary.json",
            ],
            REPORT_DIR,
        )
        run.details["comparison"] = table.reset_index().to_dict(orient="records")
        logger.info(
            "Saved development comparison %s to %s",
            experiment,
            REPORT_DIR / "comparisons",
        )
        return table


def main(argv=None):
    from team_wtm.cli import main as run_cli

    run_cli(["compare", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
