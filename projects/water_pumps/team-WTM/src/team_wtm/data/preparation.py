"""Load records, validate joins and prepare the selected model columns."""

from __future__ import annotations

import pandas as pd

from team_wtm.data.cleaning import (
    DEFAULT_LABELS_PATH,
    DEFAULT_VALUES_PATH,
    clean_data,
    load_data,
    merge_data,
)
from team_wtm.modeling.features import add_feature_columns, select_features

ID_COLUMN = "id"
TARGET_COLUMN = "status_group"


def merge_training_data(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    require_complete_match: bool = True,
) -> pd.DataFrame:
    """Join values and labels using the shared validated one-to-one join."""
    merged = merge_data(values, labels, how="outer", keep_merge_indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", ID_COLUMN]
    if require_complete_match and not unmatched.empty:
        raise ValueError(
            f"Training values and labels contain {len(unmatched)} unmatched IDs"
        )
    return merged.drop(columns="_merge")


def load_training_data(
    values_path=DEFAULT_VALUES_PATH, labels_path=DEFAULT_LABELS_PATH
):
    """Load the configured CSVs and require a complete one-to-one ID join.

    The shared merge sorts by ID, matching the existing training workflows.
    Pass explicit paths to load another dataset with the same validation.
    """
    values, labels = load_data(values_path, labels_path)
    return merge_training_data(values, labels)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the row-by-row cleaning and column-engineering rules.

    These steps learn nothing from the data and preserve row order and index.
    """
    prepared = clean_data(data)
    # add_feature_columns delegates age and quality flags to the shared
    # cleaning.add_quality_features implementation.
    return add_feature_columns(prepared)


def prepare_model_inputs(
    data: pd.DataFrame,
    *,
    option: str = "full",
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Return predictors, optional target, and IDs in their original row order.

    Use the same feature option for training and supplied test data. The shared
    select_features function raises when any required feature is unavailable.
    ID and target are never included among predictors.
    """
    if ID_COLUMN not in data.columns:
        raise KeyError(f"Data must contain an '{ID_COLUMN}' column")
    if require_target and TARGET_COLUMN not in data.columns:
        raise KeyError(f"Data must contain a '{TARGET_COLUMN}' column")

    prepared = prepare_features(data)
    X = select_features(prepared, option)
    forbidden = {ID_COLUMN, TARGET_COLUMN}.intersection(X.columns)
    if forbidden:
        raise ValueError(f"ID/target columns cannot be predictors: {sorted(forbidden)}")

    record_ids = prepared[ID_COLUMN].copy()
    y = prepared[TARGET_COLUMN].copy() if TARGET_COLUMN in prepared.columns else None
    if not X.index.equals(record_ids.index):
        raise ValueError("Predictors and record IDs are not aligned")
    if y is not None and not X.index.equals(y.index):
        raise ValueError("Predictors, target and record IDs are not aligned")
    return X, y, record_ids
