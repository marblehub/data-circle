"""Reusable data-preparation utilities for the WTM water-pumps project.

The functions in this module mirror the initial PM-2 modelling workflow.
Exploratory analysis remains in the notebook; this module contains the
repeatable steps needed to prepare modelling inputs and build a baseline
preprocessing pipeline.

Important:
- The record ID and target are never included as model predictors.
- Duplicate records are not removed. Identical feature rows are kept in the
  same train/validation split to reduce leakage.
- Imputers, scalers and encoders are fitted only when the training pipeline
  is fitted, never on validation or test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "id"
TARGET_COLUMN = "status_group"

# Feature groups and baseline selection follow the decisions documented
# in PM2_model_data_preparation.ipynb.
FEATURE_GROUPS = {
    "technical": [
        "extraction_type",
        "extraction_type_class",
        "waterpoint_type",
        "source_type",
    ],
    "management": [
        "management",
        "management_group",
        "payment",
        "payment_type",
        "scheme_management",
        "permit",
        "public_meeting",
    ],
    "water_quality": ["water_quality"],
    "numeric": ["amount_tsh", "gps_height", "population"],
    "age": ["pump_age"],
    "quantity": ["quantity", "quantity_group"],
    "location": ["region", "basin", "lga", "longitude", "latitude"],
}

ACTIVE_GROUPS = [
    "technical",
    "management",
    "water_quality",
    "numeric",
    "age",
    "location",
]

EXCLUDED_HIGH_CARDINALITY = [
    "wpt_name",
    "subvillage",
    "scheme_name",
    "installer",
    "funder",
    "ward",
]

NUMERIC_COLUMNS = [
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "population",
    "construction_year",
]


def merge_training_data(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    require_complete_match: bool = True,
) -> pd.DataFrame:
    """Merge training features and labels by ID, validating a one-to-one join.

    By default, raise an error if any ID appears in only one input table.
    The original rows and labels are retained; no duplicate removal is done.
    """
    for name, frame in (("values", values), ("labels", labels)):
        if ID_COLUMN not in frame.columns:
            raise KeyError(f"{name} must contain an '{ID_COLUMN}' column")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")

    merged = values.merge(
        labels,
        on=ID_COLUMN,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged.loc[merged["_merge"] != "both", ID_COLUMN].tolist()
    if require_complete_match and unmatched:
        raise ValueError(
            f"Training values and labels contain {len(unmatched)} unmatched IDs"
        )

    return merged.drop(columns="_merge")


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature preparation used by the PM-2 baseline.

    This function does not fit statistical transformations. It preserves
    the source columns, converts selected numeric columns, treats zero
    population/construction year as unknown, derives pump_age, and drops
    num_private as specified in the notebook.
    """
    if ID_COLUMN not in data.columns:
        raise KeyError(f"Data must contain an '{ID_COLUMN}' column")

    prepared = data.copy()

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column], errors="coerce"
            )

    if "population" in prepared.columns:
        prepared["population"] = prepared["population"].replace(0, np.nan)

    if "construction_year" in prepared.columns:
        prepared["construction_year"] = prepared[
            "construction_year"
        ].replace(0, np.nan)

    if "date_recorded" in prepared.columns and "construction_year" in prepared.columns:
        recorded_date = pd.to_datetime(
            prepared["date_recorded"], errors="coerce"
        )
        prepared["pump_age"] = (
            recorded_date.dt.year - prepared["construction_year"]
        )
        prepared.loc[prepared["pump_age"] < 0, "pump_age"] = np.nan

    if "num_private" in prepared.columns:
        prepared = prepared.drop(columns="num_private")

    return prepared


def get_baseline_feature_names(
    data: pd.DataFrame,
    *,
    active_groups: list[str] | None = None,
) -> list[str]:
    """Return available baseline predictors in the agreed feature groups."""
    groups = ACTIVE_GROUPS if active_groups is None else active_groups
    unknown_groups = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown feature groups: {unknown_groups}")

    selected: list[str] = []
    for group in groups:
        selected.extend(FEATURE_GROUPS[group])

    excluded = set(EXCLUDED_HIGH_CARDINALITY)
    available = set(data.columns)

    # Preserve the order in FEATURE_GROUPS while removing absent, excluded
    # and repeated column names.
    return list(
        dict.fromkeys(
            column
            for column in selected
            if column in available and column not in excluded
        )
    )


def prepare_model_inputs(
    data: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate predictors, target and IDs without leaking ID/target.

    Returns X, y and record_ids in the same row order.
    """
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Data is missing required columns: {sorted(missing)}")

    selected = (
        get_baseline_feature_names(data)
        if feature_names is None
        else list(feature_names)
    )
    invalid = [
        column
        for column in selected
        if column in {ID_COLUMN, TARGET_COLUMN} or column not in data.columns
    ]
    if invalid:
        raise ValueError(
            "Predictor list contains the target/ID or unavailable columns: "
            f"{invalid}"
        )
    if not selected:
        raise ValueError("No predictor columns were selected")

    X = data.loc[:, selected].copy()
    y = data[TARGET_COLUMN].copy()
    record_ids = data[ID_COLUMN].copy()

    # Apply the agreed zero-as-unknown rule even if the input has not already
    # passed through prepare_features().
    if "population" in X.columns:
        X["population"] = pd.to_numeric(
            X["population"], errors="coerce"
        ).replace(0, np.nan)

    return X, y, record_ids


def split_model_data(
    X: pd.DataFrame,
    y: pd.Series,
    record_ids: pd.Series | None = None,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    split_number: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """Create a stratified split that keeps identical feature rows together.

    The default returns the first fold, matching the notebook workflow.
    Duplicate rows are not dropped. Their shared group is used only to stop
    identical predictor combinations crossing the train/validation boundary.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indexes")
    if record_ids is not None and not X.index.equals(record_ids.index):
        raise ValueError("X and record_ids must have matching indexes")
    if not 0 <= split_number < n_splits:
        raise ValueError("split_number must be between 0 and n_splits - 1")

    if X.empty:
        raise ValueError("Cannot split an empty feature table")

    groups = X.groupby(
        list(X.columns), dropna=False, sort=False
    ).ngroup()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = splitter.split(X, y, groups=groups)
    train_idx, validation_idx = next(
        split for fold, split in enumerate(splits) if fold == split_number
    )

    X_train = X.iloc[train_idx].copy()
    X_validation = X.iloc[validation_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_validation = y.iloc[validation_idx].copy()

    id_train = (
        record_ids.iloc[train_idx].copy()
        if record_ids is not None
        else None
    )
    id_validation = (
        record_ids.iloc[validation_idx].copy()
        if record_ids is not None
        else None
    )

    train_groups = set(groups.iloc[train_idx])
    validation_groups = set(groups.iloc[validation_idx])
    if not train_groups.isdisjoint(validation_groups):
        raise RuntimeError("Identical feature groups crossed the data split")

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        id_train,
        id_validation,
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build numerical and categorical transformations from training columns.

    The returned transformer is unfitted. Fit it only on training data, or
    preferably use it inside a model Pipeline and call Pipeline.fit(X_train,
    y_train). Validation/test data must only be passed to transform/predict.
    """
    if X_train.empty:
        raise ValueError("X_train must contain at least one row")
    if X_train.columns.duplicated().any():
        raise ValueError("X_train contains duplicate column names")

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = [
        column
        for column in X_train.columns
        if column not in numerical_features
    ]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
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
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(transformers=transformers)


def build_baseline_pipeline(
    X_train: pd.DataFrame,
) -> Pipeline:
    """Build the PM-2 Logistic Regression baseline as one leakage-safe pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
