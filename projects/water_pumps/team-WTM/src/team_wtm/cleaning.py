"""Reusable data-loading, quality-checking, and cleaning functions for the water-pump project.

Expected raw files:
    data/raw/Training_set_values.csv
    data/raw/Training_set_labels.csv

Run ``python -m team_wtm.cleaning`` to execute the default pipeline, or import
its functions from another script or notebook. ``notebooks/data_audit.ipynb``
uses these same functions to explain the audit findings and compare raw and
cleaned data.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Resolve paths from this file, not from the current working directory.
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
DEFAULT_VALUES_PATH = RAW_DATA_DIR / "Training_set_values.csv"
DEFAULT_LABELS_PATH = RAW_DATA_DIR / "Training_set_labels.csv"


def _validate_ids(data: pd.DataFrame, name: str) -> None:
    """Require an ID for every row and a one-to-one values/labels relationship."""
    if "id" not in data.columns:
        raise ValueError(f"The {name} dataset must contain an 'id' column.")
    if data["id"].isna().any():
        raise ValueError(f"The {name} dataset contains missing IDs.")
    if data["id"].duplicated().any():
        raise ValueError(f"The {name} dataset contains duplicate IDs.")


def load_data(
    values_path: str | Path = DEFAULT_VALUES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feature and target CSVs, rejecting absent, missing or duplicate IDs.

    Default paths are relative to the project, so callers can run from either
    the repository root or the notebooks directory.
    """
    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)

    for name, frame in (("values", values), ("labels", labels)):
        _validate_ids(frame, name)

    return values, labels


def merge_data(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    how: str = "outer",
    keep_merge_indicator: bool = False,
) -> pd.DataFrame:
    """Join validated IDs, retaining unmatched rows by default for auditing.

    Set keep_merge_indicator=True to inspect ``left_only``, ``right_only`` and
    ``both`` counts. Remove that audit-only column before inspecting or cleaning
    features. Unmatched IDs are reported, not silently removed or imputed.
    """
    _validate_ids(values, "values")
    _validate_ids(labels, "labels")
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


def build_data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    """Describe each column without modifying data; unique counts exclude nulls."""
    return pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).to_numpy(),
            "Missing Values": data.isna().sum().to_numpy(),
            "Unique Values": data.nunique().to_numpy(),
        }
    )


def check_data_quality(
    data: pd.DataFrame,
    target_column: str = "status_group",
    exclude_from_duplicate_check: Iterable[str] = ("id",),
    verbose: bool = True,
) -> dict[str, pd.Series | int]:
    """Return null counts, duplicate counts and the available target distribution.

    Duplicates exclude IDs by default but include all other columns, including
    the target. Use verbose=False when displaying the returned values in a
    notebook. Missing targets remain visible in the distribution.
    """
    missing = data.isna().sum().sort_values(ascending=False)
    excluded = [col for col in exclude_from_duplicate_check if col in data.columns]
    duplicate_count = data.drop(columns=excluded).duplicated().sum()

    if verbose:
        print("Missing values by column:")
        print(missing)
        print(f"Duplicate rows (excluding {excluded}): {duplicate_count}")

    summary: dict[str, pd.Series | int] = {
        "missing_values": missing,
        "duplicate_rows": int(duplicate_count),
    }

    if target_column in data.columns:
        target_distribution = data[target_column].value_counts(dropna=False)
        if verbose:
            print(f"\nTarget distribution ({target_column}):")
            print(target_distribution)
        summary["target_distribution"] = target_distribution

    return summary


def inspect_columns(
    data: pd.DataFrame, verbose: bool = True
) -> dict[str, pd.DataFrame | list[str]]:
    """Return categorical cardinality and numeric summaries without changing data.

    Handle object, string and category dtypes; date_recorded is excluded from
    categorical inspection because it represents a date. Set verbose=False to
    display the returned summaries separately in a notebook.
    """
    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != "date_recorded"]
    numeric_columns = data.select_dtypes(include="number").columns.tolist()

    cardinality_summary = pd.DataFrame(
        {
            "Unique Values": data[categorical_columns].nunique(),
            "Missing Values": data[categorical_columns].isna().sum(),
        }
    ).sort_values("Unique Values", ascending=False)

    numeric_summary = (
        data[numeric_columns].describe() if numeric_columns else pd.DataFrame()
    )
    if verbose:
        print("Categorical columns:")
        print(categorical_columns)
        for column in categorical_columns:
            print(f"\n{column}:")
            print(data[column].dropna().unique())
        print("\nNumerical summary:")
        print(numeric_summary)

    return {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "cardinality_summary": cardinality_summary,
        "numeric_summary": numeric_summary,
    }


def investigate_duplicates(
    data: pd.DataFrame, target_column: str = "status_group"
) -> dict[str, pd.Series | pd.DataFrame]:
    """Compare complete observations and feature matches, keeping missing values.

    IDs identify records, not duplicate content. Feature matches exclude both
    ID and target so conflicting labels are visible. Group numbers are local
    to this report; no records are removed or declared to be the same pump.
    """
    observations = data.drop(columns=["id"], errors="ignore")
    features = observations.drop(columns=[target_column], errors="ignore")
    if features.shape[1] == 0:
        raise ValueError("Duplicate investigation requires at least one feature.")

    group_ids = features.groupby(
        features.columns.tolist(), dropna=False, sort=False, observed=True
    ).ngroup()
    members = pd.DataFrame({"Group": group_ids, "Rows": 1}, index=data.index)
    groups = members.groupby("Group", sort=False)["Rows"].sum().to_frame()
    if target_column in data:
        members["Target"] = data[target_column]
        groups["Distinct labels"] = members.groupby("Group")["Target"].nunique(
            dropna=False
        )
    else:
        groups["Distinct labels"] = pd.NA
    groups = groups.loc[groups["Rows"].gt(1)].copy()
    groups["Conflicting labels"] = groups["Distinct labels"].gt(1)
    summary = pd.Series(
        {
            "Repeated observations excluding id": int(observations.duplicated().sum()),
            "Repeated feature rows excluding id and target": int(
                features.duplicated().sum()
            ),
            "Feature duplicate groups": len(groups),
            "Rows in feature duplicate groups": int(groups["Rows"].sum()),
            "Feature duplicate groups with conflicting labels": int(
                groups["Conflicting labels"].sum()
            ),
        },
        name="Count",
    )
    return {"summary": summary, "groups": groups}


def check_suspicious_values(data: pd.DataFrame) -> pd.DataFrame:
    """Count screening flags without changing data or asserting their cause.

    Near-zero latitude uses an absolute tolerance of 1e-6 degrees. The broad
    Tanzania coordinate envelope (29..41 longitude, -12.5..-0.9 latitude) is
    a screening rule, not a country-boundary test. Years before 1900 are flagged
    for review, not automatically invalidated. Missing values are separate
    from these flags and percentages use all input rows as their denominator.
    """
    records = []

    def add(check: str, mask: pd.Series) -> None:
        count = int(mask.fillna(False).sum())
        records.append(
            {
                "Check": check,
                "Rows": count,
                "Percent of rows": 100 * count / len(data) if len(data) else 0.0,
            }
        )

    numeric = {}
    for column in (
        "population",
        "construction_year",
        "gps_height",
        "longitude",
        "latitude",
    ):
        if column in data:
            numeric[column] = pd.to_numeric(data[column], errors="coerce")
            add(
                f"{column}: non-numeric value",
                data[column].notna() & numeric[column].isna(),
            )
    for column in ("population", "construction_year", "gps_height"):
        if column in numeric:
            add(f"{column}: zero", numeric[column].eq(0))
            add(f"{column}: negative", numeric[column].lt(0))
    if "construction_year" in numeric:
        year = numeric["construction_year"]
        add("construction_year: positive year before 1900", year.gt(0) & year.lt(1900))
        add("construction_year: fractional year", year.mod(1).ne(0) & year.notna())
    if "date_recorded" in data:
        dates = pd.to_datetime(data["date_recorded"], errors="coerce")
        add(
            "date_recorded: unparseable non-missing value",
            data["date_recorded"].notna() & dates.isna(),
        )
        if "construction_year" in numeric:
            add(
                "construction_year: after recording year",
                numeric["construction_year"].gt(dates.dt.year),
            )
    if "longitude" in numeric and "latitude" in numeric:
        longitude, latitude = numeric["longitude"], numeric["latitude"]
        present = longitude.notna() & latitude.notna()
        add(
            "coordinates: longitude zero and latitude near zero",
            longitude.eq(0) & latitude.abs().le(1e-6),
        )
        add(
            "coordinates: outside global bounds",
            present & (~longitude.between(-180, 180) | ~latitude.between(-90, 90)),
        )
        add(
            "coordinates: outside broad Tanzania envelope",
            present & (~longitude.between(29, 41) | ~latitude.between(-12.5, -0.9)),
        )
    for column in ("funder", "installer"):
        if column in data:
            text = data[column].astype("string").str.strip().str.casefold()
            add(f"{column}: literal zero", text.eq("0"))
            add(
                f"{column}: blank or unknown token",
                text.isin(["", "unknown", "none", "n/a", "na"]),
            )
    if "status_group" in data:
        add(
            "status_group: unexpected non-missing label",
            data["status_group"].notna()
            & ~data["status_group"].isin(
                ["functional", "functional needs repair", "non functional"]
            ),
        )
    return pd.DataFrame.from_records(
        records, columns=["Check", "Rows", "Percent of rows"]
    ).set_index("Check")


def inspect_categorical_variants(data: pd.DataFrame) -> pd.DataFrame:
    """Find candidate aliases differing only in case or whitespace.

    Report counts and distinct spellings; keep raw values unchanged. Normalized
    matches are candidates for source review, not proof of identical entities.
    Abbreviations, spelling differences and semantic aliases require separate
    investigation and are not merged by this function.
    """
    records = []
    columns = data.select_dtypes(include=["object", "category", "string"]).columns
    for column in columns:
        if column == "date_recorded":
            continue
        raw = data[column].dropna().astype("string")
        normalized = raw.str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
        pairs = pd.DataFrame({"Raw": raw, "Normalized": normalized})
        for name, group in pairs.groupby("Normalized", sort=True):
            variants = tuple(sorted(group["Raw"].unique()))
            if len(variants) > 1:
                records.append(
                    {
                        "Column": column,
                        "Normalized": name,
                        "Variants": variants,
                        "Rows": len(group),
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=["Column", "Normalized", "Variants", "Rows"]
    )


def clean_data(
    data: pd.DataFrame,
    zero_to_missing: Iterable[str] = ("population", "construction_year"),
    date_column: str = "date_recorded",
    drop_columns: Iterable[str] = ("num_private",),
    copy: bool = True,
) -> pd.DataFrame:
    """Apply agreed cleaning rules and return the cleaned DataFrame.

    Zero values are replaced with missing values only in the columns listed in
    zero_to_missing: population and construction_year by default, following the
    audit's unknown-value convention. Other zeros (including gps_height) remain.
    Invalid dates are converted to NaT rather than raising an error. num_private
    is dropped by default as the project's current feature-selection decision,
    not because all its values are invalid. Pass drop_columns=() to retain it.

    Absent optional columns are ignored. No rows, IDs or labels are removed or
    imputed. The input is preserved unless copy=False, which permits mutation.
    """
    cleaned = data.copy() if copy else data

    for column in zero_to_missing:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, float("nan"))

    if date_column in cleaned.columns:
        cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")

    columns_to_drop = [col for col in drop_columns if col in cleaned.columns]
    if columns_to_drop:
        cleaned = cleaned.drop(columns=columns_to_drop)

    return cleaned


def derive_pump_age(data: pd.DataFrame) -> pd.Series:
    """Derive usable age at recording without rewriting source years or dates.

    Unknown/nonpositive construction years, invalid dates and negative ages
    produce missing age. A waterpoint recorded in its construction year is 0.
    """
    year = pd.to_numeric(data["construction_year"], errors="coerce")
    recorded = pd.to_datetime(data["date_recorded"], errors="coerce")
    age = recorded.dt.year - year
    return age.where(year.gt(0) & age.ge(0)).rename("pump_age")


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
    print("\nDuplicate investigation:")
    print(investigate_duplicates(data)["summary"])
    print("\nSuspicious-value screening (flags may overlap):")
    print(check_suspicious_values(data))
    if run_inspection:
        inspect_columns(data)
        print("\nCandidate categorical variants (first 20 groups):")
        print(inspect_categorical_variants(data).head(20).to_string(index=False))

    cleaned = clean_data(data)
    print("\nCleaning complete.")
    print(f"Rows: {len(cleaned):,}; columns: {len(cleaned.columns):,}")
    return cleaned


if __name__ == "__main__":
    train_data = run_pipeline()
