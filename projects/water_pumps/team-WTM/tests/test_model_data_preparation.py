import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from team_wtm.features import FEATURE_OPTIONS
from team_wtm.model_data_preparation import (
    build_preprocessor,
    merge_training_data,
    prepare_model_inputs,
)


def _feature_data(include_target=True):
    data = {
        "id": [11, 12, 13],
        "extraction_type_class": ["gravity", "handpump", "gravity"],
        "waterpoint_type_group": ["communal", "communal", "other"],
        "source_type": ["spring", "borehole", "spring"],
        "management_group": ["user-group", "commercial", "user-group"],
        "payment_type": ["never pay", "pay monthly", "never pay"],
        "scheme_management": ["VWC", "Water Board", "VWC"],
        "permit": [True, False, True],
        "public_meeting": [True, None, False],
        "water_quality": ["soft", "salty", "soft"],
        "population": [100, 0, 50],
        "gps_height": [120, None, 140],
        "construction_year": [2000, 0, 1990],
        "date_recorded": ["2010-01-01", "2012-01-01", "2010-01-01"],
        "longitude": [35.0, 0.0, 36.0],
        "latitude": [-6.0, 0.0, -7.0],
        "quantity": ["enough", "dry", "enough"],
        "quantity_group": ["enough", "dry", "enough"],
        "pump_age": [10, None, 20],
        "pump_age_missing": [False, True, False],
        "population_missing": [False, True, False],
        "gps_height_zero": [False, False, False],
        "coordinate_unusable": [False, False, False],
        "region": ["A", "B", "A"],
        "basin": ["X", "Y", "X"],
        "longitude_screened": [35.0, None, 36.0],
        "latitude_screened": [-6.0, None, -7.0],
    }
    if include_target:
        data["status_group"] = [
            "functional",
            "non functional",
            "functional needs repair",
        ]
    return pd.DataFrame(data)


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


def test_prepare_inputs_excludes_id_and_target_and_preserves_alignment():
    data = _feature_data()
    X, y, ids = prepare_model_inputs(data, option="full")
    assert "id" not in X.columns
    assert "status_group" not in X.columns
    assert X.index.equals(y.index) and X.index.equals(ids.index)
    assert ids.tolist() == [11, 12, 13]
    assert y.tolist() == data["status_group"].tolist()


def test_prepare_inputs_fails_when_required_feature_is_missing():
    data = _feature_data().drop(columns=["source_type"])
    with pytest.raises(KeyError, match="Columns missing for 'full'"):
        prepare_model_inputs(data, option="full")


def test_prepare_inputs_supports_unlabelled_test_data():
    X, y, ids = prepare_model_inputs(
        _feature_data(include_target=False), option="full", require_target=False
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
def test_full_preparation_handles_missing_boolean_categories(missing_value):
    training = _feature_data()
    training["permit"] = pd.Series([True, missing_value, False], dtype=object)
    X_train, _, _ = prepare_model_inputs(training, option="full")

    supplied_test = _feature_data(include_target=False)
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


def test_preprocessor_with_boolean_categories_can_be_saved_and_reloaded(tmp_path):
    training = _feature_data()
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
