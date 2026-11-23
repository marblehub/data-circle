import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

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
    data = pd.DataFrame(
        {"id": [1], "status_group": ["functional"], "extraction_type_class": ["gravity"]}
    )
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
    assert numeric_imputer.statistics_[0] == pytest.approx(8 / 3)
    categories = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
        .categories_[0]
    )
    assert "new" not in categories
    assert "unseen_test" not in categories
