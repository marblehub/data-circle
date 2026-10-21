"""Export a reproducible local cleaned training snapshot and provenance.

Run python -m team_wtm.export_cleaned. Raw inputs and unresolved records are
preserved. Learned imputation, encoding and scaling belong in training folds.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from team_wtm.cleaning import (
    DEFAULT_LABELS_PATH,
    DEFAULT_VALUES_PATH,
    PROJECT_DIR,
    clean_data,
    derive_pump_age,
    load_data,
    merge_data,
)

DEFAULT_OUTPUT = PROJECT_DIR / "data" / "processed" / "training_cleaned.parquet"
TARGET_CLASSES = ("functional", "functional needs repair", "non functional")
QUALITY_FLAGS = (
    "construction_year_missing",
    "population_missing",
    "pump_age_missing",
    "pump_age_invalid",
    "coordinate_placeholder",
    "coordinate_screen_failed",
)
RULES_VERSION = "sprint1-cleaned-snapshot-v1"


def sha256_file(path: Path) -> str:
    """Hash file bytes, with bounded memory use."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_quality_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic flags and usable age while retaining source columns.

    The broad audit coordinate envelope is a screening rule, not an exact
    Tanzania boundary. Original coordinates and nonzero construction years
    stay available for review.
    """
    required = {
        "population",
        "construction_year",
        "date_recorded",
        "longitude",
        "latitude",
    }
    missing = required - set(cleaned.columns)
    if missing:
        raise ValueError(f"Missing quality-feature columns: {sorted(missing)}")
    output = cleaned.copy()
    year = pd.to_numeric(output["construction_year"], errors="coerce")
    population = pd.to_numeric(output["population"], errors="coerce")
    longitude = pd.to_numeric(output["longitude"], errors="coerce")
    latitude = pd.to_numeric(output["latitude"], errors="coerce")
    output["construction_year_missing"] = year.isna()
    output["population_missing"] = population.isna()
    output["pump_age"] = derive_pump_age(output)
    output["pump_age_missing"] = output["pump_age"].isna()
    output["pump_age_invalid"] = output["pump_age_missing"] & year.notna()
    output["coordinate_placeholder"] = longitude.eq(0) & latitude.abs().le(1e-6)
    output["coordinate_screen_failed"] = ~(
        longitude.between(29, 41) & latitude.between(-12.5, -0.9)
    )
    return output


def _code_provenance() -> dict:
    """Record exact source hashes as well as the current repository revision."""
    files = [Path(__file__), Path(__file__).with_name("cleaning.py")]
    relative = [str(p.relative_to(PROJECT_DIR)) for p in files]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_files_dirty": bool(status.stdout.strip())
        if status.returncode == 0
        else None,
        "source_sha256": {
            name: sha256_file(p) for name, p in zip(relative, files, strict=True)
        },
        "uv_lock_sha256": sha256_file(PROJECT_DIR / "uv.lock"),
    }


def export_cleaned_dataset(
    values_path=DEFAULT_VALUES_PATH,
    labels_path=DEFAULT_LABELS_PATH,
    output_path=DEFAULT_OUTPUT,
    *,
    overwrite=False,
) -> dict:
    """Export all matched, labelled training records plus additive quality flags.

    A missing/extra ID or unexpected target stops the export. Files are staged
    and round-trip checked before replacement. Existing outputs require an
    explicit overwrite; metadata is installed last and contains the data hash.
    """
    values_path, labels_path, output_path = (
        Path(p).resolve() for p in (values_path, labels_path, output_path)
    )
    if output_path.suffix.lower() != ".parquet":
        raise ValueError("The cleaned output must have a .parquet extension.")
    metadata_path = output_path.with_suffix(".metadata.json")
    if {output_path, metadata_path} & {values_path, labels_path}:
        raise ValueError("Outputs must not overwrite source inputs.")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            "A snapshot or metadata file exists; use --overwrite to regenerate."
        )
    hashes = {"values": sha256_file(values_path), "labels": sha256_file(labels_path)}
    values, labels = load_data(values_path, labels_path)
    merged = merge_data(values, labels, keep_merge_indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Features and labels must contain exactly the same IDs.")
    raw = merged.drop(columns="_merge")
    if "status_group" not in raw or not raw["status_group"].isin(TARGET_CLASSES).all():
        raise ValueError("Every row needs a valid training target label.")
    cleaned = add_quality_features(clean_data(raw))
    # Nullable Boolean preserves true/false/null through Arrow without changing values.
    for column in ("public_meeting", "permit"):
        if column in cleaned:
            cleaned[column] = cleaned[column].astype("boolean")
    for role, path in (("values", values_path), ("labels", labels_path)):
        if sha256_file(path) != hashes[role]:
            raise RuntimeError(
                "An input file changed during the export; rerun on stable inputs."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".clean-export-", dir=output_path.parent
    ) as temporary:
        staged = Path(temporary) / output_path.name
        cleaned.to_parquet(staged, engine="pyarrow", compression="zstd", index=False)
        reloaded = pd.read_parquet(staged, engine="pyarrow")
        assert_frame_equal(cleaned, reloaded, check_exact=True)
        metadata = {
            "metadata_version": 1,
            "rules_version": RULES_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "inputs": {
                role: {
                    "filename": path.name,
                    "sha256": hashes[role],
                    "rows": len(frame),
                    "columns": len(frame.columns),
                }
                for role, path, frame in (
                    ("values", values_path, values),
                    ("labels", labels_path, labels),
                )
            },
            "code": _code_provenance(),
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": importlib.metadata.version("pyarrow"),
            },
            "output": {
                "filename": output_path.name,
                "sha256": sha256_file(staged),
                "rows": len(cleaned),
                "columns": len(cleaned.columns),
                "schema": [
                    {"column": col, "dtype": str(dtype)}
                    for col, dtype in cleaned.dtypes.items()
                ],
            },
            "rules": {
                "zero_to_missing": ["population", "construction_year"],
                "parse_date": "date_recorded; invalid dates become NaT",
                "nullable_boolean_columns": [
                    col for col in ("public_meeting", "permit") if col in cleaned
                ],
                "dropped_columns": ["num_private"] if "num_private" in raw else [],
                "age": "recording year minus construction year; requires positive year and nonnegative age",
                "coordinate_envelope": {
                    "longitude": [29, 41],
                    "latitude": [-12.5, -0.9],
                },
                "coordinate_placeholder": "longitude == 0 and abs(latitude) <= 1e-6",
                "rows": "all matched training records retained, including duplicates",
                "uncertain_values": "retain elevation/amount zeros, nonzero source years, coordinates and raw categories",
                "learned_preprocessing": "none; fit imputation, grouping, encoding and scaling on training folds",
            },
            "quality": {
                "flag_counts": {
                    name: int(cleaned[name].sum()) for name in QUALITY_FLAGS
                },
                "missing_counts": {
                    col: int(count) for col, count in cleaned.isna().sum().items()
                },
                "target_counts": {
                    status: int(cleaned["status_group"].eq(status).sum())
                    for status in TARGET_CLASSES
                },
                "repeated_rows_excluding_id": int(
                    raw.drop(columns="id").duplicated().sum()
                ),
                "round_trip_verified": True,
            },
        }
        staged_metadata = Path(temporary) / metadata_path.name
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staged, output_path)
        os.replace(staged_metadata, metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing local snapshot and metadata.",
    )
    args = parser.parse_args()
    metadata = export_cleaned_dataset(
        args.values, args.labels, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "metadata": str(args.output.resolve().with_suffix(".metadata.json")),
                "rows": metadata["output"]["rows"],
                "columns": metadata["output"]["columns"],
                "flag_counts": metadata["quality"]["flag_counts"],
                "git_commit": metadata["code"]["git_commit"],
                "source_files_dirty": metadata["code"]["source_files_dirty"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
