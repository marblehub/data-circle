"""Integration tests for prediction workflow."""

import json
import subprocess
import sys

import pandas as pd
import pytest

import team_wtm.predict as pr


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


def test_reloaded_model_gives_identical_predictions(records, model, model_file):
    reloaded = pr.load_model(model_file, report_path=None)
    pd.testing.assert_frame_equal(
        pr.predict_records(records, model), pr.predict_records(records, reloaded)
    )


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
    assert "ids_match_in_order: FAIL" in capsys.readouterr().err


def test_predict_module_does_not_import_streamlit():
    code = "import sys, team_wtm.predict; print('streamlit' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
