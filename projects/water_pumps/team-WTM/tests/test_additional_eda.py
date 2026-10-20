"""Checks for EDA denominators, placeholder treatment, and confounding."""

import json
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from team_wtm.additional_eda import (
    AGE_GROUPS,
    MISSING_FEATURES,
    NUMERIC,
    STATUS,
    age_availability_summary,
    age_contrasts,
    age_rates,
    missingness_by,
    numeric_correlations,
    numeric_summary,
    prepare_eda,
    run_analysis,
    standardized_extremes,
)


def sample_raw(ages):
    count = len(ages)
    data = pd.DataFrame(
        {
            "id": range(count),
            "construction_year": [2013 - age for age in ages],
            "date_recorded": ["2013-06-01"] * count,
            "population": np.arange(count) + 1,
            "amount_tsh": np.arange(count),
            "gps_height": np.arange(count) - 2,
            "status_group": [STATUS[i % 3] for i in range(count)],
            "region": ["A"] * count,
            "extraction_type_class": ["gravity"] * count,
        }
    )
    for column in MISSING_FEATURES:
        if column not in data:
            data[column] = "recorded"
    return data


def test_age_boundaries_and_unknowns_do_not_mutate_source():
    raw = sample_raw([0, 5, 6, 10, 11, 20, 21, 30, 31, 53, 0, -1, 2])
    raw.loc[10, "construction_year"] = 0
    raw.loc[12, "date_recorded"] = "bad"
    raw.loc[0, "population"] = 0
    original = raw.copy(deep=True)
    data = prepare_eda(raw)
    assert_frame_equal(raw, original)
    assert data["age_group"].astype("string").iloc[:10].tolist() == [
        "0-5",
        "0-5",
        "6-10",
        "6-10",
        "11-20",
        "11-20",
        "21-30",
        "21-30",
        "31+",
        "31+",
    ]
    assert data["pump_age"].iloc[10:].isna().all()
    assert data.loc[0, "pump_age"] == 0
    assert pd.isna(data.loc[0, "population"])
    assert data.loc[0, "amount_tsh"] == 0
    assert data.loc[0, "gps_height"] == -2
    assert data.loc[2, "gps_height"] == 0
    assert data["age_availability"].value_counts().to_dict() == {
        "Usable age": 10,
        "Invalid year/date/age": 2,
        "Unknown construction year": 1,
    }


@pytest.mark.parametrize(
    "string_columns",
    [("population",), ("construction_year",), ("population", "construction_year")],
)
def test_mixed_csv_numeric_columns_keep_placeholder_and_age_rules(string_columns):
    control = sample_raw([1, 0, -1, 5, 10, 35])
    control["population"] = [0.0, 120.0, 150.0, np.nan, 250.0, 0.0]
    control["construction_year"] = [0.0, 2013.0, 2014.0, np.nan, 2003.0, 0.0]
    csv_input = control.copy(deep=True)
    for column in string_columns:
        csv_input[column] = csv_input[column].astype("string")
        csv_input.loc[0, column] = "0"
        csv_input.loc[3, column] = "not recorded"
        csv_input.loc[5, column] = " 0 "
    raw = pd.read_csv(StringIO(csv_input.to_csv(index=False)))
    original = raw.copy(deep=True)

    actual = prepare_eda(raw)
    expected = prepare_eda(control)

    assert_frame_equal(raw, original)
    assert_frame_equal(actual, expected)
    assert actual.loc[[0, 5], ["population", "construction_year"]].isna().all().all()
    assert actual.loc[0, "age_availability"] == "Unknown construction year"
    assert actual.loc[3, "age_availability"] == "Unknown construction year"
    assert actual.loc[1, "pump_age"] == 0
    assert actual.loc[2, "age_availability"] == "Invalid year/date/age"
    assert actual.loc[0, "amount_tsh"] == 0
    assert actual.loc[2, "gps_height"] == 0
    assert actual.loc[0, "gps_height"] == -2


@pytest.mark.parametrize("problem", ["missing_column", "missing_target", "bad_target"])
def test_analysis_rejects_incomplete_inputs(problem):
    raw = sample_raw([1, 2, 3])
    if problem == "missing_column":
        raw = raw.drop(columns="region")
    else:
        raw.loc[0, "status_group"] = None if problem == "missing_target" else "unknown"
    with pytest.raises(ValueError):
        prepare_eda(raw)


def test_correlation_excludes_codes_and_uses_pairwise_counts():
    data = pd.DataFrame(
        {
            "pump_age": [0, 1, 2, 3],
            "population": [1, 2, 4, 8],
            "amount_tsh": [0, 0, 0, 0],
            "gps_height": [-1, 0, 1, np.nan],
            "id": [1, 2, 3, 4],
            "region_code": [1, 2, 3, 4],
            "status_group": [STATUS[0]] * 4,
        }
    )
    original = data.copy(deep=True)
    result = numeric_correlations(data)
    sensitivity = numeric_correlations(data, mask_elevation_zero=True)
    assert result["spearman"].columns.tolist() == list(NUMERIC)
    assert result["spearman"].loc["pump_age", "population"] == pytest.approx(1)
    assert result["pearson"].loc["pump_age", "population"] < 1
    assert pd.isna(result["spearman"].loc["amount_tsh", "population"])
    assert result["pairwise_n"].loc["pump_age", "gps_height"] == 3
    assert sensitivity["pairwise_n"].loc["pump_age", "gps_height"] == 2
    assert_frame_equal(data, original)


def test_missingness_rates_condition_on_each_class_not_whole_dataset():
    raw = sample_raw([1] * 6)
    raw["status_group"] = [STATUS[0]] * 3 + [STATUS[2]] * 2 + [STATUS[1]]
    raw["population"] = [0, 0, 5, 0, 9, 1]
    raw.loc[0, "construction_year"] = 0
    data = prepare_eda(raw)
    table = missingness_by(data, "status_group").set_index(["group", "feature"])
    assert table.loc[(STATUS[0], "population"), "missing_pct"] == pytest.approx(200 / 3)
    assert table.loc[(STATUS[2], "population"), "missing_pct"] == 50
    assert table.loc[(STATUS[1], "population"), "missing_pct"] == 0
    assert table.loc[(STATUS[0], "population"), "n"] == 3
    summary = numeric_summary(data).set_index(["feature", "status"])
    assert summary.loc[("population", STATUS[0]), "valid_n"] == 1
    assert summary.loc[("population", STATUS[0]), "median"] == 5
    availability = age_availability_summary(data)
    assert availability.groupby("availability")["status_n"].sum().to_dict() == {
        "Unknown construction year": 1,
        "Usable age": 5,
    }


def test_age_tables_keep_regions_with_no_usable_ages_and_absent_classes():
    raw = sample_raw([1, 35, 10])
    raw["region"] = ["A", "A", "B"]
    raw["status_group"] = [STATUS[0], STATUS[2], STATUS[1]]
    raw.loc[2, "construction_year"] = 0
    rates = age_rates(prepare_eda(raw), "region")
    a = rates.loc[rates["region"].eq("A")].set_index("age_group")
    b = rates.loc[rates["region"].eq("B")]
    assert rates["n"].sum() == 2
    assert len(rates) == 2 * len(AGE_GROUPS)
    assert a.loc["0-5", "functional_pct"] == 100
    assert a.loc["0-5", "functional needs repair"] == 0
    assert b["n"].sum() == 0
    assert b["functional_pct"].isna().all()
    comparisons = age_contrasts(rates, "region", min_n=1).set_index("region")
    assert comparisons.loc["A", "eligible"]
    assert not comparisons.loc["B", "eligible"]
    assert comparisons.loc["B", "young_n"] == 0


def test_standardization_uses_same_strata_and_weights_for_both_ages():
    # A Simpson reversal: younger does better in both strata but worse when pooled.
    rates = pd.DataFrame(
        [
            {
                "region": "A",
                "age_group": "0-5",
                "n": 10,
                "functional": 9,
                "functional_pct": 90,
            },
            {
                "region": "A",
                "age_group": "31+",
                "n": 100,
                "functional": 80,
                "functional_pct": 80,
            },
            {
                "region": "B",
                "age_group": "0-5",
                "n": 100,
                "functional": 20,
                "functional_pct": 20,
            },
            {
                "region": "B",
                "age_group": "31+",
                "n": 10,
                "functional": 1,
                "functional_pct": 10,
            },
            {
                "region": "C",
                "age_group": "0-5",
                "n": 1000,
                "functional": 1000,
                "functional_pct": 100,
            },
        ]
    )
    contrasts = age_contrasts(rates, "region", min_n=10)
    summary = standardized_extremes(contrasts)
    assert summary["eligible_strata"] == 2
    assert summary["common_n"] == 220
    assert summary["young_n"] == summary["old_n"] == 110
    assert summary["young_standardized_pct"] == 55
    assert summary["old_standardized_pct"] == 45
    assert summary["standardized_gap_pp"] == 10
    assert summary["pooled_gap_pp"] == pytest.approx(100 * (29 - 81) / 110)
    assert not contrasts.set_index("region").loc["C", "eligible"]


def test_empty_common_support_and_invalid_threshold_are_explicit():
    rates = pd.DataFrame(
        [
            {
                "region": "A",
                "age_group": "0-5",
                "n": 29,
                "functional": 20,
                "functional_pct": 100 * 20 / 29,
            },
            {
                "region": "A",
                "age_group": "31+",
                "n": 30,
                "functional": 15,
                "functional_pct": 50,
            },
        ]
    )
    assert standardized_extremes(age_contrasts(rates, "region")) == {
        "eligible_strata": 0,
        "common_n": 0,
    }
    assert age_contrasts(rates, "region", 29).iloc[0]["eligible"]
    with pytest.raises(ValueError, match="positive"):
        age_contrasts(rates, "region", 0)


def test_analysis_writes_summary_and_figures_to_separate_directories(tmp_path):
    raw = sample_raw([0, 3, 8, 15, 25, 35] * 5)
    values_path = tmp_path / "values.csv"
    labels_path = tmp_path / "labels.csv"
    raw.drop(columns="status_group").to_csv(values_path, index=False)
    raw[["id", "status_group"]].to_csv(labels_path, index=False)
    output_dir = tmp_path / "reports" / "sprint1" / "additional_eda"
    figures_dir = tmp_path / "reports" / "figures" / "additional_eda"

    result = run_analysis(
        values_path, labels_path, output_dir, min_n=1, figures_dir=figures_dir
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["counts"] == result["counts"]
    assert summary["counts"]["all_rows"] == len(raw)
    assert {p.name for p in output_dir.iterdir()} == {"summary.json"}
    assert {p.name for p in figures_dir.iterdir()} == {
        "numeric_distributions.png",
        "numeric_boxplots.png",
        "numeric_spearman_heatmaps.png",
        "missingness_by_status.png",
        "missingness_by_region.png",
        "age_by_region.png",
        "age_by_extraction_type_class.png",
    }
    assert all(
        p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in figures_dir.iterdir()
    )
