"""Fit imputers, scaling, categorical encoding and model pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from team_wtm.config import SETTINGS


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
                        min_frequency=SETTINGS.minimum_category_frequency,
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
