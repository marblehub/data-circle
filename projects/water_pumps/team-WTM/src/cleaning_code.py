"""Reusable data-loading, quality-checking, and cleaning functions for the water-pump project.

Expected raw files:
    data/raw/Training_set_values.csv
    data/raw/Training_set_labels.csv

Run this module directly to execute the default pipeline, or import its functions
from another script or notebook.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd


# Resolve paths from this file, not from the current working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
DEFAULT_VALUES_PATH = RAW_DATA_DIR / "Training_set_values.csv"
DEFAULT_LABELS_PATH = RAW_DATA_DIR / "Training_set_labels.csv"


def load_data(
    values_path: str | Path = DEFAULT_VALUES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feature and target CSV files and return them separately."""
    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)

    for name, frame in (("values", values), ("labels", labels)):
        if "id" not in frame.columns:
            raise ValueError(f"The {name} dataset must contain an 'id' column.")
        if frame["id"].isna().any():
            raise ValueError(f"The {name} dataset contains missing IDs.")
        if frame["id"].duplicated().any():
            raise ValueError(f"The {name} dataset contains duplicate IDs.")

    return values, labels


def merge_data(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    how: str = "outer",
    keep_merge_indicator: bool = False,
) -> pd.DataFrame:
    """Merge feature and label data on id, optionally retaining the merge indicator."""
    merged = values.merge(
        labels,
        how=how,
        on="id",
        sort=True,
        indicator=True,
        validate="one_to_one",
    )

    if keep_merge_indicator:
        return merged
    return merged.drop(columns="_merge")


def check_data_quality(
    data: pd.DataFrame,
    target_column: str = "status_group",
    exclude_from_duplicate_check: Iterable[str] = ("id",),
) -> dict[str, pd.Series | int]:
    """Print and return key data-quality summaries without changing the DataFrame."""
    missing = data.isna().sum().sort_values(ascending=False)
    excluded = [col for col in exclude_from_duplicate_check if col in data.columns]
    duplicate_count = data.drop(columns=excluded).duplicated().sum()

    print("Missing values by column:")
    print(missing)
    print(f"Duplicate rows (excluding {excluded}): {duplicate_count}")

    summary: dict[str, pd.Series | int] = {
        "missing_values": missing,
        "duplicate_rows": int(duplicate_count),
    }

    if target_column in data.columns:
        target_distribution = data[target_column].value_counts(dropna=False)
        print(f"\nTarget distribution ({target_column}):")
        print(target_distribution)
        summary["target_distribution"] = target_distribution

    return summary


def inspect_columns(data: pd.DataFrame) -> dict[str, pd.DataFrame | list[str]]:
    """Print categorical unique values and return categorical/numeric summaries."""
    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    categorical_columns = [
        col for col in categorical_columns if col != "date_recorded"
    ]
    numeric_columns = data.select_dtypes(include="number").columns.tolist()

    print("Categorical columns:")
    print(categorical_columns)
    for column in categorical_columns:
        print(f"\n{column}:")
        print(data[column].dropna().unique())

    numeric_summary = (
        data[numeric_columns].describe() if numeric_columns else pd.DataFrame()
    )
    print("\nNumerical summary:")
    print(numeric_summary)

    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_summary": numeric_summary,
    }


def clean_data(
    data: pd.DataFrame,
    zero_to_missing: Iterable[str] = ("population", "construction_year"),
    date_column: str = "date_recorded",
    drop_columns: Iterable[str] = ("num_private",),
    copy: bool = True,
) -> pd.DataFrame:
    """Apply agreed cleaning rules and return the cleaned DataFrame.

    Zero values are replaced with missing values only in the columns listed in
    zero_to_missing. Invalid dates are converted to NaT rather than raising an
    error. Columns absent from drop_columns or zero_to_missing are ignored.
    """
    cleaned = data.copy() if copy else data

    for column in zero_to_missing:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, float("nan"))

    if date_column in cleaned.columns:
        cleaned[date_column] = pd.to_datetime(
            cleaned[date_column], errors="coerce"
        )

    columns_to_drop = [col for col in drop_columns if col in cleaned.columns]
    if columns_to_drop:
        cleaned = cleaned.drop(columns=columns_to_drop)

    return cleaned


def run_pipeline(
    values_path: str | Path = DEFAULT_VALUES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    merge_how: str = "outer",
    run_inspection: bool = True,
) -> pd.DataFrame:
    """Load, merge, report quality, optionally inspect, clean, and return data."""
    values, labels = load_data(values_path, labels_path)
    data = merge_data(values, labels, how=merge_how)

    check_data_quality(data)
    if run_inspection:
        inspect_columns(data)

    cleaned = clean_data(data)
    print("\nCleaning complete.")
    print(f"Rows: {len(cleaned):,}; columns: {len(cleaned.columns):,}")
    return cleaned


if __name__ == "__main__":
    train_data = run_pipeline()
