"""Checks for the Issue #16 feature options and edge cases."""

import pandas as pd
import pytest

from team_wtm.cleaning import clean_data
from team_wtm.features import FEATURE_OPTIONS, add_feature_columns, select_features


@pytest.fixture
def featured():
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            # 1: built in recording year -> age 0 (valid)
            # 2: built after recording -> negative age (invalid)
            # 3: year 0 -> unknown
            # 4: normal age, but placeholder coordinates
            "construction_year": [2013, 2015, 0, 2000],
            "date_recorded": ["2013-03-01"] * 4,
            "population": [100, 50, 0, 20],
            "gps_height": [1200, 0, 500, -20],
            "longitude": [35.0, 35.0, 35.0, 0.0],
            "latitude": [-6.0, -6.0, -6.0, -2e-8],
        }
    )
    return add_feature_columns(clean_data(raw)).set_index("id")


def test_zero_age_is_valid(featured):
    assert featured.loc[1, "pump_age"] == 0
    assert not featured.loc[1, "pump_age_missing"]


def test_negative_age_is_missing_and_invalid(featured):
    assert pd.isna(featured.loc[2, "pump_age"])
    assert featured.loc[2, "pump_age_invalid"]


def test_unknown_year_is_missing_not_invalid(featured):
    assert featured.loc[3, "pump_age_missing"]
    assert not featured.loc[3, "pump_age_invalid"]
    assert featured.loc[3, "population_missing"]


def test_coordinate_placeholder_keeps_row(featured):
    assert len(featured) == 4
    assert featured.loc[4, "coordinate_unusable"]
    assert pd.isna(featured.loc[4, "longitude_screened"])
    assert featured.loc[4, "longitude"] == 0.0
    assert not featured.loc[1, "coordinate_unusable"]


def test_gps_height_zero_flag(featured):
    assert featured.loc[2, "gps_height_zero"]
    assert not featured.loc[4, "gps_height_zero"]  # negative is not zero


def test_quantity_columns_move_together():
    for name, columns in FEATURE_OPTIONS.items():
        assert ("quantity" in columns) == ("quantity_group" in columns), name


def test_no_target_or_id_in_any_option():
    for columns in FEATURE_OPTIONS.values():
        assert "status_group" not in columns
        assert "id" not in columns


def test_unknown_option_raises(featured):
    with pytest.raises(KeyError):
        select_features(featured, "does_not_exist")