"""Integration tests for cleaning workflow."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from team_wtm.cleaning import (
    clean_data,
    load_data,
    merge_data,
    run_pipeline,
)


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
