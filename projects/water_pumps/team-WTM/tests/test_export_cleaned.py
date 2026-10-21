"""Checks for local snapshot integrity and reproducible provenance."""

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from team_wtm.cleaning import clean_data, load_data, merge_data
from team_wtm.export_cleaned import (
    QUALITY_FLAGS,
    add_quality_features,
    export_cleaned_dataset,
    sha256_file,
)


@pytest.fixture
def inputs(tmp_path):
    values = pd.DataFrame(
        {
            "id": [8, 2, 5, 1, 7, 9],
            "construction_year": [2000, 0, 2015, 2013, 2001, 2000],
            "date_recorded": ["2013-01-01"] * 4 + ["bad-date", "2013-01-01"],
            "population": [0, 5, 10, 1, 20, 0],
            "gps_height": [0, -3, 100, 50, 60, 0],
            "amount_tsh": [0, 0, 25, 0, 50, 0],
            "longitude": [0, 35, 35, 35, 35, 0],
            "latitude": [-2e-8, -6, -6, -0.95, None, -2e-8],
            "installer": ["0", "DWE", "DWE", "DWE", "unknown", "0"],
            "public_meeting": [True, False, None, True, False, True],
            "permit": [None, True, False, True, True, None],
            "num_private": [0] * 6,
        }
    )
    labels = pd.DataFrame(
        {
            "id": [1, 2, 5, 7, 8, 9],
            "status_group": [
                "functional",
                "functional needs repair",
                "non functional",
                "functional",
                "functional",
                "functional",
            ],
        }
    )
    values_path, labels_path = tmp_path / "values.csv", tmp_path / "labels.csv"
    values.to_csv(values_path, index=False)
    labels.to_csv(labels_path, index=False)
    return values_path, labels_path, tmp_path / "processed" / "cleaned.parquet"


def test_snapshot_round_trip_and_metadata_preserve_all_training_records(inputs):
    values_path, labels_path, output = inputs
    hashes = [sha256_file(values_path), sha256_file(labels_path)]
    metadata = export_cleaned_dataset(values_path, labels_path, output)
    frame = pd.read_parquet(output).set_index("id")
    assert frame.index.tolist() == [1, 2, 5, 7, 8, 9]
    assert frame["status_group"].to_dict() == {
        1: "functional",
        2: "functional needs repair",
        5: "non functional",
        7: "functional",
        8: "functional",
        9: "functional",
    }
    assert "num_private" not in frame
    assert pd.isna(frame.loc[8, "population"])
    assert pd.isna(frame.loc[2, "construction_year"])
    assert frame.loc[1, "pump_age"] == 0
    assert frame.loc[[2, 5, 7], "pump_age"].isna().all()
    assert frame.loc[5, "construction_year"] == 2015
    assert frame.loc[2, "gps_height"] == -3
    assert frame.loc[8, "gps_height"] == frame.loc[8, "amount_tsh"] == 0
    assert frame.loc[8, "installer"] == "0"
    assert frame.loc[7, "installer"] == "unknown"
    assert str(frame["public_meeting"].dtype) == "boolean"
    assert pd.isna(frame.loc[5, "public_meeting"])
    assert frame.loc[2, "public_meeting"] == False
    assert metadata["quality"]["repeated_rows_excluding_id"] == 1
    assert metadata["quality"]["flag_counts"] == {
        "construction_year_missing": 1,
        "population_missing": 2,
        "pump_age_missing": 3,
        "pump_age_invalid": 2,
        "coordinate_placeholder": 2,
        "coordinate_screen_failed": 3,
    }
    # The broad audit envelope intentionally accepts latitude -0.95.
    assert not frame.loc[1, "coordinate_screen_failed"]
    assert metadata["output"]["rows"] == 6
    assert metadata["output"]["columns"] == len(frame.columns) + 1
    assert metadata["output"]["sha256"] == sha256_file(output)
    assert metadata["quality"]["round_trip_verified"]
    assert sum(metadata["quality"]["target_counts"].values()) == 6
    assert set(metadata["code"]["source_sha256"]) == {
        "src/team_wtm/cleaning.py",
        "src/team_wtm/export_cleaned.py",
    }
    assert [sha256_file(values_path), sha256_file(labels_path)] == hashes
    assert metadata["inputs"]["values"]["sha256"] == hashes[0]
    assert metadata["inputs"]["labels"]["sha256"] == hashes[1]
    assert json.loads(output.with_suffix(".metadata.json").read_text()) == metadata


def test_quality_features_are_additive_and_leave_source_values_unchanged(inputs):
    values_path, labels_path, _ = inputs
    values, labels = load_data(values_path, labels_path)
    cleaned = clean_data(merge_data(values, labels))
    original = cleaned.copy(deep=True)
    flagged = add_quality_features(cleaned)
    assert_frame_equal(cleaned, original)
    assert_frame_equal(flagged[cleaned.columns], cleaned)
    assert set(QUALITY_FLAGS).issubset(flagged.columns)


def test_regeneration_requires_explicit_overwrite(inputs):
    values_path, labels_path, output = inputs
    export_cleaned_dataset(values_path, labels_path, output)
    parquet_bytes = output.read_bytes()
    metadata_bytes = output.with_suffix(".metadata.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_cleaned_dataset(values_path, labels_path, output)
    assert output.read_bytes() == parquet_bytes
    assert output.with_suffix(".metadata.json").read_bytes() == metadata_bytes
    original = pd.read_parquet(output)
    metadata = export_cleaned_dataset(values_path, labels_path, output, overwrite=True)
    assert_frame_equal(pd.read_parquet(output), original)
    assert metadata["output"]["sha256"] == sha256_file(output)


@pytest.mark.parametrize("problem", ["unmatched_ids", "missing_target", "bad_target"])
def test_invalid_training_inputs_do_not_create_outputs(inputs, problem):
    values_path, labels_path, output = inputs
    labels = pd.read_csv(labels_path)
    if problem == "unmatched_ids":
        labels = labels.iloc[:-1]
    else:
        labels.loc[0, "status_group"] = (
            None if problem == "missing_target" else "unknown"
        )
    labels.to_csv(labels_path, index=False)
    with pytest.raises(ValueError):
        export_cleaned_dataset(values_path, labels_path, output)
    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()


def test_failed_parquet_write_keeps_existing_snapshot_and_metadata(inputs, monkeypatch):
    values_path, labels_path, output = inputs
    export_cleaned_dataset(values_path, labels_path, output)
    original = (output.read_bytes(), output.with_suffix(".metadata.json").read_bytes())

    def fail_write(*args, **kwargs):
        raise OSError("Simulated storage failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(OSError, match="storage failure"):
        export_cleaned_dataset(values_path, labels_path, output, overwrite=True)
    assert (
        output.read_bytes(),
        output.with_suffix(".metadata.json").read_bytes(),
    ) == original
    assert not list(output.parent.glob(".clean-export-*"))


def test_source_change_during_export_cannot_get_incorrect_provenance(
    inputs, monkeypatch
):
    import team_wtm.export_cleaned as module

    values_path, labels_path, output = inputs
    original_load = module.load_data

    def changed_load(*args):
        values, labels = original_load(*args)
        changed = values.copy()
        changed.loc[0, "population"] = 123
        changed.to_csv(values_path, index=False)
        return values, labels

    monkeypatch.setattr(module, "load_data", changed_load)
    with pytest.raises(RuntimeError, match="input file changed"):
        export_cleaned_dataset(values_path, labels_path, output)
    assert not output.exists()


def test_output_extension_is_checked_before_loading(inputs):
    values_path, labels_path, output = inputs
    with pytest.raises(ValueError, match="parquet extension"):
        export_cleaned_dataset(values_path, labels_path, output.with_suffix(".csv"))
