"""Unit tests for quality features."""

from pandas.testing import assert_frame_equal

from team_wtm.cleaning import clean_data, load_data, merge_data
from team_wtm.export_cleaned import (
    QUALITY_FLAGS,
    add_quality_features,
)


def test_quality_features_are_additive_and_leave_source_values_unchanged(snapshot_inputs):
    values_path, labels_path, _ = snapshot_inputs
    values, labels = load_data(values_path, labels_path)
    cleaned = clean_data(merge_data(values, labels))
    original = cleaned.copy(deep=True)
    flagged = add_quality_features(cleaned)
    assert_frame_equal(cleaned, original)
    assert_frame_equal(flagged[cleaned.columns], cleaned)
    assert set(QUALITY_FLAGS).issubset(flagged.columns)
