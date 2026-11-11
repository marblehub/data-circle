"""Shared data splits and scores for Sprint 2."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from team_wtm.cleaning import clean_data

RANDOM_SEED = 42
N_FOLDS = 5
TARGET_CLASSES = ("functional", "functional needs repair", "non functional")

# Short names keep the CSV column headings readable.
CLASS_NAMES = {
    "functional": "functional",
    "functional needs repair": "repair",
    "non functional": "non_functional",
}
RESULT_COLUMNS = [
    "model",
    "features",
    "fold",
    "train_rows",
    "validation_rows",
    "accuracy",
    "macro_f1",
    "missed_repairs",
    "false_repair_alarms",
    "functional_precision",
    "functional_recall",
    "functional_f1",
    "functional_support",
    "repair_precision",
    "repair_recall",
    "repair_f1",
    "repair_support",
    "non_functional_precision",
    "non_functional_recall",
    "non_functional_f1",
    "non_functional_support",
    "random_seed",
    "notes",
]


def validate_training(data):
    """Stop if IDs or labels cannot be used safely."""
    if "id" not in data.columns or "status_group" not in data.columns:
        raise ValueError("Training data must contain id and status_group.")
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Training IDs must be present and unique.")
    if not data["status_group"].isin(TARGET_CLASSES).all():
        raise ValueError("Every training row needs one of the three known labels.")


def make_split_plan(training, n_folds=N_FOLDS, random_seed=RANDOM_SEED):
    """Reserve a labelled holdout and make folds for model comparisons.

    With five folds, the holdout contains about 20% of the records.
    The remaining records get five development folds.
    """
    validate_training(training)

    # Sorting first gives the same split even if the CSV rows change order.
    ordered = training.sort_values("id").reset_index(drop=True)
    features = ordered.drop(columns=["id", "status_group"])
    features = clean_data(features)
    if features.empty:
        raise ValueError("At least one feature is needed to group matching records.")

    # Give matching features the same group number, even when labels differ.
    # dropna=False keeps records with matching missing values in their groups.
    feature_columns = features.columns.tolist()
    groups = features.groupby(
        feature_columns, dropna=False, sort=False, observed=True
    ).ngroup()
    target = ordered["status_group"]
    class_counts = target.value_counts()
    for label in TARGET_CLASSES:
        if class_counts.get(label, 0) < n_folds:
            raise ValueError("Each class needs at least one record per fold.")

    # Stratification aims for similar class proportions. Groups cannot be split.
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_seed
    )
    row_counts = np.zeros((len(ordered), 1))
    outer_splits = list(splitter.split(row_counts, target, groups))
    development_rows, holdout_rows = outer_splits[0]

    plan = pd.DataFrame({"id": ordered["id"], "group": groups})
    plan["split"] = "development"
    plan.loc[holdout_rows, "split"] = "holdout"
    # -1 marks records that must stay out of cross-validation.
    plan["cv_fold"] = -1

    development_target = target.iloc[development_rows]
    development_groups = groups.iloc[development_rows]
    class_counts = development_target.value_counts()
    for label in TARGET_CLASSES:
        if class_counts.get(label, 0) < n_folds:
            raise ValueError("Development data need more records for these folds.")

    development_splits = splitter.split(
        row_counts[development_rows], development_target, development_groups
    )
    for fold, split in enumerate(development_splits):
        _, validation_rows = split
        # These positions refer to development rows; map them to the full table.
        full_table_rows = development_rows[validation_rows]
        plan.loc[full_table_rows, "cv_fold"] = fold

    # Check that no matching-feature group was divided between splits or folds.
    splits_per_group = plan.groupby("group")["split"].nunique()
    if splits_per_group.max() != 1:
        raise ValueError("A feature group crossed the holdout boundary.")
    development_plan = plan[plan["split"] == "development"]
    folds_per_group = development_plan.groupby("group")["cv_fold"].nunique()
    if folds_per_group.max() != 1:
        raise ValueError("A feature group crossed development folds.")
    return plan


def development_folds(training, plan):
    """Return fitting and validation row positions for each development fold.

    Use these positions with training.iloc[...]. The holdout stays excluded.
    """
    validate_training(training)
    if plan["id"].duplicated().any() or set(training["id"]) != set(plan["id"]):
        raise ValueError("The split plan must match training IDs exactly.")

    # Match by ID, then number rows in the same order as the training table.
    aligned = plan.set_index("id").loc[training["id"]].reset_index()
    is_development = aligned["split"] == "development"
    if not aligned["split"].isin(["development", "holdout"]).all():
        raise ValueError("Unknown split name.")
    if (aligned.loc[is_development, "cv_fold"] < 0).any():
        raise ValueError("Only development records can have a validation fold.")
    if (aligned.loc[~is_development, "cv_fold"] != -1).any():
        raise ValueError("Only development records can have a validation fold.")

    splits_per_group = aligned.groupby("group")["split"].nunique()
    if splits_per_group.max() != 1:
        raise ValueError("A feature group crossed the holdout boundary.")
    development_plan = aligned[is_development]
    folds_per_group = development_plan.groupby("group")["cv_fold"].nunique()
    if folds_per_group.max() != 1:
        raise ValueError("A feature group crossed development folds.")

    folds = []
    fold_numbers = sorted(development_plan["cv_fold"].unique())
    for fold in fold_numbers:
        fit_mask = is_development & (aligned["cv_fold"] != fold)
        validation_mask = is_development & (aligned["cv_fold"] == fold)
        fit_rows = aligned.index[fit_mask].tolist()
        validation_rows = aligned.index[validation_mask].tolist()
        if not fit_rows or not validation_rows:
            raise ValueError("Each fold needs fitting and validation records.")
        folds.append((int(fold), fit_rows, validation_rows))
    return folds


def score_predictions(actual, predicted):
    """Score all three classes, including classes the model never predicts."""
    actual = pd.Series(list(actual))
    predicted = pd.Series(list(predicted))
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels need the same nonzero length.")
    if not actual.isin(TARGET_CLASSES).all():
        raise ValueError("Scores require the three known label names.")
    if not predicted.isin(TARGET_CLASSES).all():
        raise ValueError("Scores require the three known label names.")

    # zero_division=0 gives a score of zero when a class is never predicted.
    report = classification_report(
        actual,
        predicted,
        labels=list(TARGET_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    repair = "functional needs repair"
    missed_repairs = (actual == repair) & (predicted != repair)
    false_alarms = (actual != repair) & (predicted == repair)
    matrix = confusion_matrix(actual, predicted, labels=list(TARGET_CLASSES))

    scores = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "missed_repairs": int(missed_repairs.sum()),
        "false_repair_alarms": int(false_alarms.sum()),
        # Rows show actual labels; columns show predictions, in TARGET_CLASSES order.
        "confusion_matrix": matrix.tolist(),
    }
    for label, short_name in CLASS_NAMES.items():
        class_scores = report[label]
        scores[f"{short_name}_precision"] = float(class_scores["precision"])
        scores[f"{short_name}_recall"] = float(class_scores["recall"])
        scores[f"{short_name}_f1"] = float(class_scores["f1-score"])
        scores[f"{short_name}_support"] = int(class_scores["support"])
    return scores
