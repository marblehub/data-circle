"""Check the prediction workflow (predict.py) on made-up records.

The made-up `training` data comes from tests/conftest.py. A small final
pipeline is fitted here, in the test only; predict.py itself never fits.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest

import team_wtm.final_model as fm
import team_wtm.predict as pr
from team_wtm.features import FEATURE_OPTIONS, select_features
from team_wtm.model_comparison import add_model_columns


@pytest.fixture
def model(training):
    """The fixed final setup, fitted on the made-up training data."""
    features = select_features(add_model_columns(training), fm.FINAL_FEATURES)
    return fm.fit_final_pipeline(features, training["status_group"])


@pytest.fixture
def records(training):
    """Made-up records without the target, in a shuffled (unsorted) order."""
    shuffled = training.sample(frac=1, random_state=1)
    return shuffled.drop(columns="status_group")


@pytest.fixture
def model_file(model, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return path


class NoFitModel:
    """Wraps a fitted model; fails the test if anything tries to fit it."""

    def __init__(self, model):
        self.model = model
        self.feature_names_in_ = model.feature_names_in_
        self.seen_columns = None

    def fit(self, *args, **kwargs):
        raise AssertionError("predict.py must never fit the model")

    def predict(self, features):
        self.seen_columns = list(features.columns)
        return self.model.predict(features)


# --- Predictions -----------------------------------------------------------


def test_one_prediction_per_record_with_ids_in_original_order(records, model):
    predictions = pr.predict_records(records, model)
    assert list(predictions.columns) == ["id", "status_group"]
    assert len(predictions) == len(records)
    assert predictions["id"].tolist() == records["id"].tolist()


def test_labels_are_allowed_and_not_missing(records, model):
    predictions = pr.predict_records(records, model)
    assert set(predictions["status_group"]) <= {
        "functional",
        "functional needs repair",
        "non functional",
    }
    assert not predictions.isna().any().any()


def test_records_are_not_changed(records, model):
    before = records.copy()
    pr.predict_records(records, model)
    pd.testing.assert_frame_equal(records, before)


def test_predict_never_fits_and_uses_trained_column_order(records, model):
    wrapped = NoFitModel(model)
    pr.predict_records(records, wrapped)
    assert wrapped.seen_columns == list(model.feature_names_in_)


def test_source_has_no_fit_or_holdout():
    source = Path(pr.__file__).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "score_holdout" not in source
    assert "SPLIT_PATH" not in source


# --- Rejected records ------------------------------------------------------


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


# --- Column order ----------------------------------------------------------


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


# --- Loading the model and versions ----------------------------------------


def test_reloaded_model_gives_identical_predictions(records, model, model_file):
    reloaded = pr.load_model(model_file, report_path=None)
    pd.testing.assert_frame_equal(
        pr.predict_records(records, model), pr.predict_records(records, reloaded)
    )


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


# --- Files and command line ------------------------------------------------


@pytest.mark.parametrize("name", pr.RECORD_FILE_NAMES)
def test_every_known_test_file_name_is_found(tmp_path, name):
    (tmp_path / name).write_text("id\n1\n")
    assert pr.find_records_path(tmp_path) == tmp_path / name


def test_no_test_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No test records"):
        pr.find_records_path(tmp_path)


def test_save_predictions_as_csv_and_json(records, model, tmp_path):
    predictions = pr.predict_records(records, model)
    csv_path = pr.save_predictions(predictions, tmp_path / "out" / "p.csv", "csv")
    json_path = pr.save_predictions(predictions, tmp_path / "out" / "p.json", "json")
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), predictions)
    assert pd.read_json(json_path)["id"].tolist() == predictions["id"].tolist()
    with pytest.raises(ValueError, match="Unknown format"):
        pr.save_predictions(predictions, tmp_path / "p.txt", "txt")


@pytest.mark.parametrize("project_cli", [False, True])
def test_command_line_with_records(records, model_file, tmp_path, project_cli):
    from team_wtm import cli

    records_path = tmp_path / "records.csv"
    records.to_csv(records_path, index=False)
    output = tmp_path / "predictions.json"
    args = [
        "--records",
        str(records_path),
        "--model",
        str(model_file),
        "--output",
        str(output),
        "--format",
        "json",
    ]
    if project_cli:
        cli.main(["predict", *args])
    else:
        pr.main(args)
    saved = pd.read_json(output)
    assert list(saved.columns) == ["id", "status_group"]
    assert saved["id"].tolist() == records["id"].tolist()


def test_completion_check_report_has_totals_only(records, model_file, tmp_path):
    records_path = tmp_path / "records.csv"
    records.to_csv(records_path, index=False)
    submission_path = tmp_path / "SubmissionFormat.csv"
    records[["id"]].assign(status_group="functional").to_csv(
        submission_path, index=False
    )
    report_path = tmp_path / "check.json"

    report = pr.run_completion_check(
        records_path,
        model_file,
        tmp_path / "p.csv",
        "csv",
        submission_path,
        report_path,
    )

    assert report["all_checks_passed"]
    assert report["prediction_rows"] == len(records)
    assert sum(report["label_counts"].values()) == len(records)
    # Only totals: no list of ids or labels is stored in the report.
    saved = json.loads(report_path.read_text())
    assert not any(
        isinstance(value, list) and len(value) == len(records)
        for value in saved.values()
    )


def test_completion_check_fails_when_ids_differ(
    records, model_file, tmp_path, monkeypatch, capsys
):
    from functools import partial

    from team_wtm import cli

    records_path = tmp_path / "records.csv"
    records.to_csv(records_path, index=False)
    submission_path = tmp_path / "SubmissionFormat.csv"
    # Same ids, but sorted: the order no longer matches.
    records[["id"]].sort_values("id").assign(status_group="functional").to_csv(
        submission_path, index=False
    )

    monkeypatch.setattr(pr, "find_records_path", lambda: records_path)
    monkeypatch.setattr(
        pr,
        "run_completion_check",
        partial(
            pr.run_completion_check,
            submission_path=submission_path,
            report_path=tmp_path / "check.json",
        ),
    )
    output = tmp_path / "p.csv"
    report = pr.run_prediction(model_path=model_file, output_path=output)
    assert not report["checks"]["ids_match_in_order"]
    assert not report["all_checks_passed"]
    assert capsys.readouterr().out == ""

    with pytest.raises(SystemExit, match="Completion check failed"):
        cli.main(["predict", "--model", str(model_file), "--output", str(output)])
    assert "ids_match_in_order: FAIL" in capsys.readouterr().out


def test_predict_module_does_not_import_streamlit():
    code = "import sys, team_wtm.predict; print('streamlit' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
