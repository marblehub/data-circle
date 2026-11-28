"""Integration tests for preprocessing pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from team_wtm.features import FEATURE_OPTIONS
from team_wtm.model_data_preparation import (
    build_pipeline,
    build_preprocessor,
    prepare_model_inputs,
)


def test_preprocessor_with_boolean_categories_can_be_saved_and_reloaded(feature_data, tmp_path):
    training = feature_data()
    training["permit"] = pd.Series([True, np.nan, False], dtype=object)
    X, _, _ = prepare_model_inputs(training, option="full")
    preprocessor = build_preprocessor(X).fit(X)

    path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, path)
    reloaded = joblib.load(path)

    np.testing.assert_allclose(reloaded.transform(X), preprocessor.transform(X))
    np.testing.assert_array_equal(
        reloaded.get_feature_names_out(), preprocessor.get_feature_names_out()
    )


@pytest.mark.parametrize("option", FEATURE_OPTIONS)
def test_training_and_prediction_preparation_match(training, option):
    from team_wtm.predict import prepare_features as prepare_prediction_features

    # Preserve a non-default index and row order through both entry points.
    records = training.iloc[::-1].drop(columns="status_group")
    before = records.copy(deep=True)
    expected, target, ids = prepare_model_inputs(
        records, option=option, require_target=False
    )
    actual = prepare_prediction_features(records, FEATURE_OPTIONS[option], option)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(records, before)
    assert target is None
    pd.testing.assert_series_equal(ids, records["id"])


@pytest.mark.parametrize("missing_value", [np.nan, None, pd.NA])
def test_pipeline_handles_missing_categories_and_reload(feature_data, tmp_path, missing_value):
    training = feature_data()
    training["permit"] = pd.Series([True, missing_value, False], dtype=object)
    columns, target, _ = prepare_model_inputs(training, option="full")
    supplied = feature_data(include_target=False)
    supplied.loc[0, "source_type"] = "new_source"
    supplied["permit"] = pd.Series([False, True, missing_value], dtype=object)
    validation, _, _ = prepare_model_inputs(
        supplied, option="full", require_target=False
    )

    pipeline = build_pipeline(columns, LogisticRegression(random_state=42))
    pipeline.fit(columns, target)
    predictions = pipeline.predict(validation)
    probabilities = pipeline.predict_proba(validation)
    assert len(predictions) == len(validation)
    assert np.isfinite(probabilities).all()
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(reloaded.predict(validation), predictions)
    np.testing.assert_allclose(reloaded.predict_proba(validation), probabilities)
