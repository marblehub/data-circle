"""Regression checks for the functions shared by the audit notebook and script."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from cleaning_code import (
    build_data_dictionary,
    check_data_quality,
    clean_data,
    inspect_columns,
    load_data,
    merge_data,
    run_pipeline,
)


@pytest.mark.parametrize("dataset", ["values", "labels"])
@pytest.mark.parametrize(
    ("invalid_ids", "message"),
    [
        (pd.DataFrame({"other": [1, 2]}), "must contain an 'id' column"),
        (pd.DataFrame({"id": [1, None]}), "missing IDs"),
        (pd.DataFrame({"id": [1, 1]}), "duplicate IDs"),
    ],
)
def test_invalid_ids_are_rejected_when_loading_and_merging(
    tmp_path, dataset, invalid_ids, message
):
    """Guard both CSV loading and direct DataFrame callers against ambiguous IDs."""
    frames = {
        "values": pd.DataFrame({"id": [1, 2], "population": [0, 15]}),
        "labels": pd.DataFrame({"id": [1, 2], "status_group": ["a", "b"]}),
    }
    frames[dataset] = invalid_ids
    paths = {name: tmp_path / f"{name}.csv" for name in frames}
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)

    with pytest.raises(ValueError, match=message):
        load_data(paths["values"], paths["labels"])
    with pytest.raises(ValueError, match=message):
        merge_data(frames["values"], frames["labels"])


def test_outer_merge_exposes_unmatched_ids_without_changing_inputs():
    values = pd.DataFrame({"id": [2, 1], "population": [10, 20]})
    labels = pd.DataFrame({"id": [2, 3], "status_group": ["a", "b"]})
    original_values, original_labels = values.copy(), labels.copy()

    merged = merge_data(values, labels, keep_merge_indicator=True)

    assert merged["id"].tolist() == [1, 2, 3]
    assert merged["_merge"].tolist() == ["left_only", "both", "right_only"]
    assert "_merge" not in merge_data(values, labels).columns
    assert_frame_equal(values, original_values)
    assert_frame_equal(labels, original_labels)


def test_cleaning_preserves_raw_data_rows_labels_and_valid_zeros():
    """Only the agreed placeholders and dates change; the target stays intact."""
    raw = pd.DataFrame(
        {
            "id": [2, 1, 3],
            "population": [0, 20, None],
            "construction_year": [2000, 0, None],
            "gps_height": [0, -10, 50],
            "amount_tsh": [0, 25, 0],
            "date_recorded": ["2013-01-01", "invalid", None],
            "num_private": [0, 1, 0],
            "status_group": ["functional", "non functional", None],
        }
    )
    original = raw.copy(deep=True)

    cleaned = clean_data(raw)

    expected = pd.DataFrame(
        {
            "id": [2, 1, 3],
            "population": [None, 20.0, None],
            "construction_year": [2000.0, None, None],
            "gps_height": [0, -10, 50],
            "amount_tsh": [0, 25, 0],
            "date_recorded": pd.to_datetime(["2013-01-01", None, None]),
            "status_group": ["functional", "non functional", None],
        }
    )
    assert_frame_equal(cleaned, expected)
    assert_frame_equal(raw, original)
    assert_frame_equal(clean_data(cleaned), cleaned)


def test_cleaning_accepts_optional_columns_and_rule_overrides():
    raw = pd.DataFrame({"id": [1], "population": [0], "num_private": [0]})

    assert_frame_equal(clean_data(raw, zero_to_missing=(), drop_columns=()), raw)
    assert_frame_equal(clean_data(raw[["id"]]), raw[["id"]])


def test_quality_reports_duplicates_and_missing_targets_without_printing(capsys):
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "population": [10, 10, None],
            "status_group": ["a", "a", None],
        }
    )
    original = raw.copy(deep=True)

    quality = check_data_quality(raw, verbose=False)

    assert quality["duplicate_rows"] == 1
    assert quality["missing_values"]["population"] == 1
    assert quality["target_distribution"]["a"] == 2
    assert quality["target_distribution"].sum() == 3
    assert quality["target_distribution"].index.isna().sum() == 1
    assert "target_distribution" not in check_data_quality(raw[["id"]], verbose=False)
    assert capsys.readouterr().out == ""
    assert_frame_equal(raw, original)


def test_dictionary_and_inspection_support_all_categorical_dtypes(capsys):
    raw = pd.DataFrame(
        {
            "object_field": pd.Series(["a", None, "a"], dtype=object),
            "string_field": pd.Series(["a", "b", None], dtype="string"),
            "category_field": pd.Series(["x", "x", None], dtype="category"),
            "date_recorded": ["2013-01-01"] * 3,
            "population": [0, 10, 20],
        }
    )
    original = raw.copy(deep=True)

    dictionary = build_data_dictionary(raw).set_index("Column")
    inspection = inspect_columns(raw, verbose=False)

    assert dictionary.index.tolist() == raw.columns.tolist()
    assert dictionary.loc["object_field", "Missing Values"] == 1
    assert dictionary.loc["string_field", "Unique Values"] == 2
    assert dictionary.loc["category_field", "Data Type"] == "category"
    assert inspection["categorical_columns"] == [
        "object_field",
        "string_field",
        "category_field",
    ]
    assert inspection["numeric_columns"] == ["population"]
    assert inspection["cardinality_summary"].loc["string_field", "Unique Values"] == 2
    assert (
        inspection["cardinality_summary"].loc["category_field", "Missing Values"] == 1
    )
    assert inspection["numeric_summary"].loc["mean", "population"] == 10
    assert capsys.readouterr().out == ""
    assert_frame_equal(raw, original)


def test_inspection_accepts_frames_without_numeric_or_categorical_columns():
    text_only = inspect_columns(pd.DataFrame({"name": ["a"]}), verbose=False)
    numeric_only = inspect_columns(pd.DataFrame({"id": [1]}), verbose=False)

    assert text_only["numeric_summary"].empty
    assert numeric_only["categorical_columns"] == []
    assert numeric_only["cardinality_summary"].empty


@pytest.mark.parametrize("run_inspection", [True, False])
def test_pipeline_matches_notebook_workflow(tmp_path, run_inspection):
    """Both entry points must yield identical data even when IDs do not match."""
    values = pd.DataFrame(
        {
            "id": [3, 1, 2],
            "population": [0, 10, 20],
            "construction_year": [0, 2000, 2005],
            "date_recorded": ["bad-date", "2013-01-01", "2013-02-01"],
            "num_private": [0, 1, 0],
        }
    )
    labels = pd.DataFrame({"id": [1, 2, 4], "status_group": ["a", "b", "a"]})
    values_path, labels_path = tmp_path / "values.csv", tmp_path / "labels.csv"
    values.to_csv(values_path, index=False)
    labels.to_csv(labels_path, index=False)

    loaded_values, loaded_labels = load_data(values_path, labels_path)
    raw = merge_data(loaded_values, loaded_labels, keep_merge_indicator=True)
    notebook_result = clean_data(raw.drop(columns="_merge"))
    pipeline_result = run_pipeline(
        values_path, labels_path, run_inspection=run_inspection
    )

    assert_frame_equal(pipeline_result, notebook_result)
    assert pipeline_result["id"].tolist() == [1, 2, 3, 4]
    assert pipeline_result["population"].isna().sum() == 2
    assert pipeline_result["date_recorded"].isna().sum() == 2
    assert "num_private" not in pipeline_result.columns
    assert_series_equal(pipeline_result["status_group"], raw["status_group"])
