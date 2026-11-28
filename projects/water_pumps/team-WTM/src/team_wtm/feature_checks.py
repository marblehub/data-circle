"""PM-3 feature and location checks (Issue #16, cards 13, 15, 16 and 17).

Run with:
    uv run python -m team_wtm.feature_checks ladder    # card 13: add one group at a time
    uv run python -m team_wtm.feature_checks h6        # card 15: one field vs combined
    uv run python -m team_wtm.feature_checks groups    # card 16: results by region and age
    uv run python -m team_wtm.feature_checks regions   # card 17: train and validate on different regions
    uv run python -m team_wtm.feature_checks charts    # charts from the saved results

Reuses team code instead of rewriting it:
- PM-1 (evaluation.py): the saved split plan, development folds and score_predictions.
- PM-4 (model_comparison.py): load_training, add_model_columns, build_pipeline,
  run_cross_validation, save_reports and the model settings.
- PM-3 (features.py): the feature options.

Only development records are used. The labelled holdout is never scored here.
Individual predictions are saved under data/processed/ (ignored by Git);
only aggregate tables are written to reports/sprint2/.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from team_wtm.cleaning import PROJECT_DIR
from team_wtm.evaluation import (
    RANDOM_SEED,
    RESULT_COLUMNS,
    development_folds,
    score_predictions,
)
from team_wtm.features import select_features
from team_wtm.model_comparison import (
    MODELS,
    REPORT_DIR,
    SPLIT_PATH,
    add_model_columns,
    build_pipeline,
    load_training,
    run_cross_validation,
    save_reports,
)

# The final setup chosen in #17. All PM-3 checks use it, so results are comparable.
MODEL_NAME = "random_forest_balanced"
FINAL_OPTION = "full"
REPAIR = "functional needs repair"

# D20: groups with fewer than 30 records of the class of interest are flagged.
MIN_REPAIRS = 30
N_REGION_FOLDS = 5

PREDICTION_PATH = PROJECT_DIR / "data" / "processed" / "pm3_out_of_fold_predictions.csv"
REGION_PREDICTION_PATH = PROJECT_DIR / "data" / "processed" / "pm3_region_fold_predictions.csv"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures" / "sprint2"


# ---------------------------------------------------------------------------
# Cards 13 and 15: more feature options on the same folds
# ---------------------------------------------------------------------------

def reuse_rows(file_name, options):
    """Fold rows and confusion matrices already saved by #17 for the final model."""
    rows = pd.read_csv(REPORT_DIR / f"{file_name}_cv_scores.csv")
    rows = rows[rows["model"].eq(MODEL_NAME) & rows["features"].isin(options)]
    summary = json.loads((REPORT_DIR / f"{file_name}_comparison_summary.json").read_text())
    matrices = {
        f"{MODEL_NAME} / {option}": summary["confusion_matrices"][f"{MODEL_NAME} / {option}"]
        for option in options
    }
    return rows, matrices


def run_options(training, plan, new_options, reused_file, reused_options, name):
    """Train the final model on new options; put them next to reused #17 rows."""
    rows, matrices = reuse_rows(reused_file, reused_options)
    all_rows = [rows]
    for option in new_options:
        print(f"Training {MODEL_NAME} with '{option}' on 5 folds ...")
        option_rows, option_matrices = run_cross_validation(
            training, plan, MODEL_NAME, feature_option=option
        )
        all_rows.append(option_rows)
        matrices[f"{MODEL_NAME} / {option}"] = option_matrices
    results = pd.concat(all_rows, ignore_index=True)[RESULT_COLUMNS]
    return save_reports(results, matrices, name, [*reused_options, *new_options])


def run_ladder(training, plan):
    """Card 13: split the age and missing-flag step into two steps."""
    return run_options(
        training, plan,
        new_options=["base_age"],
        reused_file="location",
        reused_options=["base", "base_age_missing", "full"],
        name="pm3_ladder",
    )


def run_h6(training, plan):
    """Card 15 (H6): one field alone against the combined feature groups."""
    return run_options(
        training, plan,
        new_options=["single_quantity", "single_waterpoint_type"],
        reused_file="quantity",
        reused_options=["full"],
        name="pm3_h6",
    )


# ---------------------------------------------------------------------------
# Card 16: results by region and by known / unknown age
# ---------------------------------------------------------------------------

def out_of_fold_predictions(training, plan, model_name=MODEL_NAME, option=FINAL_OPTION):
    """Predict each development record once, from the fold where it is validated.

    The same fitting as model_comparison.run_cross_validation, but the
    predictions are kept so that results can be split into groups.
    """
    columns = add_model_columns(training)
    features = select_features(columns, option)
    target = training["status_group"]
    parts = []
    for fold, fit_rows, validation_rows in development_folds(training, plan):
        print(f"  fold {fold}: fitting on {len(fit_rows)} records ...")
        pipeline = build_pipeline(features.iloc[fit_rows], MODELS[model_name]())
        pipeline.fit(features.iloc[fit_rows], target.iloc[fit_rows])
        parts.append(pd.DataFrame({
            "id": training["id"].iloc[validation_rows].to_numpy(),
            "fold": fold,
            "actual": target.iloc[validation_rows].to_numpy(),
            "predicted": pipeline.predict(features.iloc[validation_rows]),
            "region": columns["region"].iloc[validation_rows].to_numpy(),
            "pump_age_missing": columns["pump_age_missing"].iloc[validation_rows].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def group_results(predictions, by, min_repairs=MIN_REPAIRS):
    """Scores for each group, with record counts and a small-group flag.

    `by` is one column name or a list of column names in `predictions`.
    A group is flagged when it has fewer than `min_repairs` pumps that
    really need repair: its repair scores are then too uncertain to compare.
    """
    keys = [by] if isinstance(by, str) else list(by)
    rows = []
    for group, part in predictions.groupby(keys, sort=True, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        scores = score_predictions(part["actual"], part["predicted"])
        repairs = int(part["actual"].eq(REPAIR).sum())
        row = dict(zip(keys, group))
        row.update({
            "records": len(part),
            "repair_records": repairs,
            "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "repair_recall": scores["repair_recall"] if repairs else np.nan,
            "repair_precision": scores["repair_precision"],
            "missed_repairs": scores["missed_repairs"],
            "false_repair_alarms": scores["false_repair_alarms"],
            "too_small": repairs < min_repairs,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def overall_fold_scores(predictions):
    """Fold scores from the kept predictions, to check they match #17."""
    rows = []
    for fold, part in predictions.groupby("fold"):
        scores = score_predictions(part["actual"], part["predicted"])
        rows.append({"fold": fold, "macro_f1": scores["macro_f1"],
                     "repair_recall": scores["repair_recall"],
                     "missed_repairs": scores["missed_repairs"]})
    return pd.DataFrame(rows)


def run_groups(training, plan):
    """Card 16: final model results by region, by age availability, and both."""
    print(f"Training {MODEL_NAME} with '{FINAL_OPTION}' and keeping predictions ...")
    predictions = out_of_fold_predictions(training, plan)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_PATH, index=False)  # local only, ignored by Git

    predictions["age"] = np.where(predictions["pump_age_missing"], "unknown", "known")
    tables = {
        "region": group_results(predictions, "region"),
        "age": group_results(predictions, "age"),
        "region_age": group_results(predictions, ["region", "age"]),
    }
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / f"pm3_groups_by_{name}.csv", index=False)

    folds = overall_fold_scores(predictions)
    summary = {
        "scope": "Development folds only (out-of-fold predictions); holdout not scored.",
        "model": MODEL_NAME,
        "features": FINAL_OPTION,
        "min_repairs_for_reliable_group": MIN_REPAIRS,
        "check_against_17_macro_f1_mean": float(folds["macro_f1"].mean()),
        "fold_scores": folds.to_dict(orient="records"),
    }
    (REPORT_DIR / "pm3_groups_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(tables["age"].round(3).to_string(index=False))
    print(f"Mean macro F1 over folds: {folds['macro_f1'].mean():.4f} (should match #17)")


# ---------------------------------------------------------------------------
# Card 17: train on some regions, validate on other regions
# ---------------------------------------------------------------------------

def region_folds(training, plan, n_folds=N_REGION_FOLDS):
    """Folds over development records where each region is in one fold only.

    Unlike the main folds (which only keep identical records together),
    a validation region is never seen during fitting. The holdout stays out.
    """
    aligned = plan.set_index("id").loc[training["id"]].reset_index()
    development = np.flatnonzero(aligned["split"].eq("development").to_numpy())
    regions = training["region"].iloc[development].to_numpy()
    folds = []
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (fit, validation) in enumerate(
        splitter.split(development, groups=regions)
    ):
        fit_rows, validation_rows = development[fit], development[validation]
        if set(regions[fit]) & set(regions[validation]):
            raise ValueError("A region is in both fitting and validation rows.")
        folds.append((fold, fit_rows.tolist(), validation_rows.tolist(),
                      sorted(set(regions[validation]))))
    return folds


def run_regions(training, plan):
    """Card 17: the final setup when the validation regions are unseen."""
    columns = add_model_columns(training)
    features = select_features(columns, FINAL_OPTION)
    target = training["status_group"]
    rows, parts, matrices = [], [], []
    for fold, fit_rows, validation_rows, regions in region_folds(training, plan):
        print(f"Region fold {fold}: validating on {', '.join(regions)} ...")
        pipeline = build_pipeline(features.iloc[fit_rows], MODELS[MODEL_NAME]())
        pipeline.fit(features.iloc[fit_rows], target.iloc[fit_rows])
        predicted = pipeline.predict(features.iloc[validation_rows])
        scores = score_predictions(target.iloc[validation_rows], predicted)
        matrices.append({"fold": fold, "regions": regions,
                         "confusion_matrix": scores.pop("confusion_matrix")})
        row = {"model": MODEL_NAME, "features": f"{FINAL_OPTION} (unseen regions)",
               "fold": fold, "train_rows": len(fit_rows),
               "validation_rows": len(validation_rows)}
        row.update(scores)
        row["random_seed"] = RANDOM_SEED
        row["notes"] = "Validation regions: " + ", ".join(regions)
        rows.append(row)
        parts.append(pd.DataFrame({
            "id": training["id"].iloc[validation_rows].to_numpy(),
            "fold": fold,
            "actual": target.iloc[validation_rows].to_numpy(),
            "predicted": predicted,
            "region": columns["region"].iloc[validation_rows].to_numpy(),
        }))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results.to_csv(REPORT_DIR / "pm3_region_cv_scores.csv", index=False)
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(REGION_PREDICTION_PATH, index=False)  # local only

    # Per region: unseen-region scores next to the main-fold scores.
    unseen = group_results(predictions, "region")
    main = pd.read_csv(REPORT_DIR / "pm3_groups_by_region.csv")
    keep = ["region", "macro_f1", "repair_recall", "missed_repairs"]
    by_region = main[["region", "records", "repair_records", "too_small", *keep[1:]]].merge(
        unseen[keep], on="region", suffixes=("_main", "_unseen"))
    by_region.to_csv(REPORT_DIR / "pm3_region_transfer_by_region.csv", index=False)

    main_folds = pd.read_csv(REPORT_DIR / "quantity_cv_scores.csv")
    main_folds = main_folds[main_folds["model"].eq(MODEL_NAME)
                            & main_folds["features"].eq(FINAL_OPTION)]
    metrics = ["macro_f1", "accuracy", "repair_recall", "repair_precision"]
    comparison = pd.DataFrame({
        "split": ["main folds (identical records kept together)",
                  "region folds (validation regions unseen)"],
        **{f"{m}_mean": [main_folds[m].mean(), results[m].mean()] for m in metrics},
        **{f"{m}_std": [main_folds[m].std(ddof=1), results[m].std(ddof=1)] for m in metrics},
    })
    comparison.to_csv(REPORT_DIR / "pm3_region_comparison_table.csv", index=False)
    summary = {
        "scope": "Development records only; holdout not scored.",
        "model": MODEL_NAME,
        "features": FINAL_OPTION,
        "region_folds": N_REGION_FOLDS,
        "grouping": "GroupKFold by region; each region is validated once and never fitted on in that fold.",
        "difference_from_main_split": (
            "Main folds only keep identical records together, so nearby pumps in the "
            "same region appear in fitting and validation. Region folds hold out whole regions."
        ),
        "comparison": comparison.to_dict(orient="records"),
        "confusion_matrices": matrices,
    }
    (REPORT_DIR / "pm3_region_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparison.round(3).to_string(index=False))


# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "ladder": run_ladder,
    "h6": run_h6,
    "groups": run_groups,
    "regions": run_regions,
}


def main():
    parser = argparse.ArgumentParser(description="Issue #16 feature and location checks.")
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "charts"])
    experiment = parser.parse_args().experiment
    if experiment == "charts":
        from team_wtm.feature_check_charts import make_charts
        make_charts()
        return
    training = load_training()
    plan = pd.read_csv(SPLIT_PATH)  # PM-1's saved plan: same folds for every check
    result = EXPERIMENTS[experiment](training, plan)
    if result is not None:
        from team_wtm.cli import print_comparison

        print_comparison(result, REPORT_DIR)


if __name__ == "__main__":
    main()
