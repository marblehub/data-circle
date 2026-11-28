"""Reusable PM-2 preparation built on the shared Team WTM rules.

Cleaning, pump-age derivation, feature engineering, feature options, and split
assignments come from the shared project modules. This module intentionally
does not define a second feature list or split strategy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from team_wtm.cleaning import (
    DEFAULT_LABELS_PATH,
    DEFAULT_VALUES_PATH,
    clean_data,
    load_data,
    merge_data,
)
from team_wtm.features import add_feature_columns, select_features

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


def load_training_data(values_path=DEFAULT_VALUES_PATH, labels_path=DEFAULT_LABELS_PATH):
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
    # export_cleaned.add_quality_features implementation.
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


def _categorical_values_as_text(features: pd.DataFrame) -> pd.DataFrame:
    """Convert categories to text and use NaN for every missing value."""
    output = features.copy()
    for column in output.columns:
        values = output[column]
        output[column] = values.astype(str).astype(object).where(values.notna(), np.nan)
    return output


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer from training columns only.

    Fit only on the fitting rows of a training fold. Validation and supplied
    test data must only be transformed with the fitted transformer.
    """
    if X_train.empty:
        raise ValueError("X_train must contain at least one row")
    if X_train.columns.duplicated().any():
        raise ValueError("X_train contains duplicate column names")

    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "as_text",
                    FunctionTransformer(
                        _categorical_values_as_text,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=0.01,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("X_train must contain at least one feature")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def categories_as_text(features):
    """Support category conversion in previously saved pipelines and notebooks.

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
    """Join the preprocessor and model into one unfitted Pipeline.

    The preprocessor normalizes categorical values in its as_text step.

    Learned steps (filling missing values, scaling, encoding) are fitted only
    when Pipeline.fit is called, which we do on the fitting rows only.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(fit_features)),
            ("model", model),
        ]
    )
