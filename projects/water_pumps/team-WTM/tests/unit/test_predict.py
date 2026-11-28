"""Unit tests for predict."""

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

import team_wtm.modeling.predict as pr
import team_wtm.modeling.training as fm
from team_wtm.modeling.features import FEATURE_OPTIONS


def test_source_has_no_fit_or_holdout():
    source = Path(pr.__file__).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "score_holdout" not in source
    assert "SPLIT_PATH" not in source


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r.assign(status_group="functional"), "target column"),
        (lambda r: r.iloc[0:0], "empty"),
        (lambda r: r.drop(columns="id"), "'id' column"),
        (lambda r: r.assign(id=[None] + r["id"].tolist()[1:]), "missing ids"),
        (
            lambda r: r.assign(id=[r["id"].iloc[1]] + r["id"].tolist()[1:]),
            "duplicate ids",
        ),
    ],
)
def test_bad_records_are_rejected(records, model, change, message):
    with pytest.raises(ValueError, match=message):
        pr.predict_records(change(records), model)


@pytest.mark.parametrize("bad", [None, [1, 2], {"id": [1]}, "records.csv"])
def test_non_dataframe_is_rejected(model, bad):
    with pytest.raises(TypeError, match="DataFrame"):
        pr.predict_records(bad, model)


def test_records_missing_a_needed_column_are_rejected(records, model):
    with pytest.raises((KeyError, ValueError)):
        pr.predict_records(records.drop(columns="quantity"), model)


def test_features_follow_the_model_order_even_if_it_differs(records):
    reversed_order = FEATURE_OPTIONS[fm.FINAL_FEATURES][::-1]
    features = pr.prepare_features(records, reversed_order)
    assert list(features.columns) == reversed_order
    assert len(features) == len(records)


def test_model_with_other_features_is_refused(records):
    with pytest.raises(ValueError, match="does not use"):
        pr.prepare_features(records, FEATURE_OPTIONS["base"])


def test_wrong_order_and_missing_columns_are_refused():
    expected = ["a", "b", "c"]
    with pytest.raises(ValueError, match="wrong order"):
        pr.check_feature_columns(pd.DataFrame(columns=["b", "a", "c"]), expected)
    with pytest.raises(ValueError, match="Missing: \\['c'\\]"):
        pr.check_feature_columns(pd.DataFrame(columns=["a", "b"]), expected)
    with pytest.raises(ValueError, match="Unexpected: \\['d'\\]"):
        pr.check_feature_columns(pd.DataFrame(columns=["a", "b", "c", "d"]), expected)
    pr.check_feature_columns(pd.DataFrame(columns=expected), expected)


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="final_model"):
        pr.load_model(tmp_path / "absent.joblib", report_path=None)


def test_matching_versions_give_no_warning(tmp_path):
    report = tmp_path / "final_model.json"
    report.write_text(json.dumps({"versions": fm.versions()}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pr.check_versions(report) == []


def test_different_versions_give_a_warning(tmp_path):
    old = dict(fm.versions(), scikit_learn="0.0.1", pandas="0.0.1")
    report = tmp_path / "final_model.json"
    report.write_text(json.dumps({"versions": old}))
    with pytest.warns(UserWarning, match="version differs") as caught:
        messages = pr.check_versions(report)
    assert len(messages) == 2
    assert len(caught) == 2
    assert any("scikit_learn" in str(warning.message) for warning in caught)


@pytest.mark.parametrize("name", pr.RECORD_FILE_NAMES)
def test_every_known_test_file_name_is_found(tmp_path, name):
    (tmp_path / name).write_text("id\n1\n")
    assert pr.find_records_path(tmp_path) == tmp_path / name


def test_no_test_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No test records"):
        pr.find_records_path(tmp_path)
