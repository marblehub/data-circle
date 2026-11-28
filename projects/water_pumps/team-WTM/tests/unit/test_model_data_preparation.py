"""Unit tests for model data preparation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from team_wtm.model_data_preparation import (
    build_preprocessor,
    merge_training_data,
    prepare_model_inputs,
)


def test_merge_rejects_duplicate_ids():
    values = pd.DataFrame({"id": [1, 1], "value": [10, 20]})
    labels = pd.DataFrame(
        {"id": [1, 2], "status_group": ["functional", "non functional"]}
    )
    with pytest.raises(ValueError, match="duplicate IDs"):
        merge_training_data(values, labels)


def test_merge_rejects_unmatched_ids_and_can_report_them():
    values = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    labels = pd.DataFrame({"id": [1], "status_group": ["functional"]})
    with pytest.raises(ValueError, match="unmatched IDs"):
        merge_training_data(values, labels)
    result = merge_training_data(values, labels, require_complete_match=False)
    assert len(result) == 2
    assert result["status_group"].isna().sum() == 1


def test_prepare_inputs_excludes_id_and_target_and_preserves_alignment(feature_data):
    data = feature_data()
    X, y, ids = prepare_model_inputs(data, option="full")
    assert "id" not in X.columns
    assert "status_group" not in X.columns
    assert X.index.equals(y.index) and X.index.equals(ids.index)
    assert ids.tolist() == [11, 12, 13]
    assert y.tolist() == data["status_group"].tolist()


def test_prepare_inputs_fails_when_required_feature_is_missing(feature_data):
    data = feature_data().drop(columns=["source_type"])
    with pytest.raises(KeyError, match="Columns missing for 'full'"):
        prepare_model_inputs(data, option="full")


def test_prepare_inputs_supports_unlabelled_test_data(feature_data):
    X, y, ids = prepare_model_inputs(
        feature_data(include_target=False), option="full", require_target=False
    )
    assert y is None
    assert ids.tolist() == [11, 12, 13]
    assert X.shape[0] == 3


def test_preprocessor_handles_missing_values_and_unseen_categories_consistently():
    X_train = pd.DataFrame(
        {"numeric": [1.0, np.nan, 3.0, 4.0], "category": ["a", "b", None, "a"]}
    )
    X_validation = pd.DataFrame(
        {"numeric": [np.nan, 5.0], "category": ["new", "b"]}
    )
    X_test = pd.DataFrame({"numeric": [2.0], "category": ["unseen_test"]})

    preprocessor = build_preprocessor(X_train)
    assert isinstance(preprocessor, ColumnTransformer)
    train_out = preprocessor.fit_transform(X_train)
    validation_out = preprocessor.transform(X_validation)
    test_out = preprocessor.transform(X_test)

    assert train_out.shape[1] == validation_out.shape[1] == test_out.shape[1]
    names = preprocessor.get_feature_names_out()
    assert len(names) == train_out.shape[1]
    assert np.isfinite(train_out).all()
    assert np.isfinite(validation_out).all()
    assert np.isfinite(test_out).all()

    numeric_imputer = preprocessor.named_transformers_["numerical"].named_steps["imputer"]
    assert numeric_imputer.statistics_[0] == pytest.approx(3.0)
    categories = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
        .categories_[0]
    )
    assert "__MISSING__" in categories
    assert "new" not in categories
    assert "unseen_test" not in categories


@pytest.mark.parametrize(
    "missing_value", [np.nan, None, pd.NA], ids=["nan", "none", "pd_na"]
)
def test_full_preparation_handles_missing_boolean_categories(feature_data, missing_value):
    training = feature_data()
    training["permit"] = pd.Series([True, missing_value, False], dtype=object)
    X_train, _, _ = prepare_model_inputs(training, option="full")

    supplied_test = feature_data(include_target=False)
    supplied_test["permit"] = pd.Series([False, True, missing_value], dtype=object)
    supplied_test.loc[0, "source_type"] = "unseen_source"
    X_test, target, ids = prepare_model_inputs(
        supplied_test, option="full", require_target=False
    )
    original = X_train.copy(deep=True)

    preprocessor = build_preprocessor(X_train)
    train_out = preprocessor.fit_transform(X_train)
    names = preprocessor.get_feature_names_out().copy()
    test_out = preprocessor.transform(X_test)

    assert target is None
    assert ids.tolist() == supplied_test["id"].tolist()
    assert X_train.columns.tolist() == X_test.columns.tolist()
    assert train_out.shape[1] == test_out.shape[1] == len(names)
    assert np.isfinite(train_out).all()
    assert np.isfinite(test_out).all()
    np.testing.assert_array_equal(preprocessor.get_feature_names_out(), names)
    pd.testing.assert_frame_equal(X_train, original)

    categorical = preprocessor.named_transformers_["categorical"]
    permit_position = categorical.feature_names_in_.tolist().index("permit")
    categories = categorical.named_steps["encoder"].categories_[permit_position]
    assert set(categories) == {"False", "True", "__MISSING__"}


def test_load_training_uses_complete_id_join_and_keeps_alignment(tmp_path):
    from team_wtm.model_data_preparation import load_training_data

    values = pd.DataFrame({"id": [3, 1, 2], "population": [30, 10, 20]})
    labels = pd.DataFrame(
        {
            "id": [2, 3, 1],
            "status_group": ["non functional", "functional", "functional needs repair"],
        }
    )
    values_path = tmp_path / "values.csv"
    labels_path = tmp_path / "labels.csv"
    values.to_csv(values_path, index=False)
    labels.to_csv(labels_path, index=False)

    result = load_training_data(values_path, labels_path)
    expected = merge_training_data(values, labels)
    pd.testing.assert_frame_equal(result, expected)
    assert result["id"].tolist() == [1, 2, 3]
    assert result["population"].tolist() == [10, 20, 30]
    assert result["status_group"].tolist() == [
        "functional needs repair", "non functional", "functional"
    ]


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([1, 1], "duplicate IDs"),
        ([1, None], "missing IDs"),
        ([1, 3], "unmatched IDs"),
    ],
)
def test_load_training_rejects_invalid_id_relationships(tmp_path, ids, message):
    from team_wtm.model_data_preparation import load_training_data

    pd.DataFrame({"id": ids}).to_csv(tmp_path / "values.csv", index=False)
    labels = pd.DataFrame(
        {"id": [1, 2], "status_group": ["functional", "non functional"]}
    )
    labels.to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_training_data(tmp_path / "values.csv", tmp_path / "labels.csv")
