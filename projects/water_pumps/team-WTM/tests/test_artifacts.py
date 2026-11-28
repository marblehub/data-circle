"""Check model identity, linked dashboard evidence, and one-time holdout reuse."""

import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from team_wtm import predict
from team_wtm.artifacts import (
    read_json,
    read_prediction_check,
    report_ref,
    sha256_file,
    verified_report,
    write_json,
)
from team_wtm.config import DEFAULT_CONFIG
from team_wtm.features import select_features
from team_wtm.model_comparison import MODELS, add_model_columns, build_pipeline


@pytest.mark.parametrize("name", ["scores.csv", "comparisons/scores.csv"])
def test_verified_report_rejects_drift_and_accepts_windows_line_endings(tmp_path, name):
    table = tmp_path / name
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(b"score\n0.5\n")
    manifest = tmp_path / "model.json"
    write_json(manifest, {"reports": {"scores": report_ref(table, relative_to=tmp_path)}})
    table.write_bytes(b"score\r\n0.5\r\n")
    assert verified_report(manifest, "scores") == table.read_bytes()
    table.write_text("score\n0.7\n")
    with pytest.raises(ValueError, match="checksum"):
        verified_report(manifest, "scores")


def test_custom_model_identity_and_corrupt_manifest(training, tmp_path):
    features = select_features(add_model_columns(training), "base")
    estimator = MODELS["logistic_regression"]()
    pipeline = build_pipeline(features, estimator).fit(
        features, training["status_group"]
    )
    binary = tmp_path / "alternative.joblib"
    joblib.dump(pipeline, binary)
    records = training.drop(columns="status_group")
    records.to_csv(tmp_path / "records.csv", index=False)
    submission = records[["id"]].assign(status_group="functional")
    submission.to_csv(tmp_path / "format.csv", index=False)

    def check():
        return predict.run_completion_check(
            tmp_path / "records.csv",
            binary,
            tmp_path / "predictions.csv",
            "csv",
            tmp_path / "format.csv",
        )

    standalone = check()
    assert standalone["run_id"] is None and standalone["model"] == "LogisticRegression"
    assert standalone["features"] == "base"
    assert (tmp_path / "predictions.csv.check.json").is_file()
    sidecar = binary.with_suffix(".json")
    metadata = {
        "run_id": "alternative-run",
        "model_sha256": sha256_file(binary),
        "versions": predict.versions(),
        "settings": {
            "model": "logistic_regression",
            "features": "base",
            "feature_columns": features.columns.tolist(),
            "model_settings": estimator.get_params(),
        },
    }
    write_json(sidecar, metadata)
    assert check()["run_id"] == "alternative-run"
    assert check()["model"] == "logistic_regression"
    from team_wtm.cli import main

    explicit = tmp_path / "selected-model.json"
    sidecar.rename(explicit)
    main(
        [
            "predict",
            "--model",
            str(binary),
            "--model-report",
            str(explicit),
            "--records",
            str(tmp_path / "records.csv"),
            "--output",
            str(tmp_path / "cli.csv"),
        ]
    )
    assert (tmp_path / "cli.csv").read_bytes() == (
        tmp_path / "predictions.csv"
    ).read_bytes()
    explicit.rename(sidecar)
    metadata["settings"]["feature_columns"] = features.columns.tolist()[::-1]
    write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="columns"):
        predict.load_model(binary)
    metadata["settings"]["feature_columns"] = features.columns.tolist()
    metadata["settings"]["model"] = "logistic_regression_balanced"
    write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="estimator"):
        predict.load_model(binary)
    metadata["model_sha256"] = "0" * 64
    write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="checksum"):
        predict.load_model(binary)


def test_final_run_archive_and_holdout_guards(training, tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    config = root / "configs/project.toml"
    config.write_text(
        DEFAULT_CONFIG.read_text().replace("forest_trees = 300", "forest_trees = 2")
    )
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    training.drop(columns="status_group").to_csv(
        raw / "Training_set_values.csv", index=False
    )
    labels = raw / "Training_set_labels.csv"
    training[["id", "status_group"]].to_csv(labels, index=False)
    env = {**os.environ, "TEAM_WTM_CONFIG": str(config)}

    def run(module, *, check=True):
        return subprocess.run(
            [sys.executable, "-m", f"team_wtm.{module}"],
            cwd=root,
            env=env,
            check=check,
            text=True,
            capture_output=True,
        )

    run("baseline")
    run("final_model")
    reports = root / "reports/sprint2"
    public = reports / "final_model.json"
    holdout = reports / "final/holdout_result.json"
    original_holdout = holdout.read_bytes()
    first = read_json(public)
    archived = root / "artifacts/runs" / first["run_id"]
    assert sha256_file(archived / "model.joblib") == first["model_sha256"]
    assert verified_report(archived / "model.json", "holdout") == original_holdout
    assert read_json(archived / "config_snapshot.json")["forest_trees"] == 2
    assert (
        predict.load_model(archived / "model.joblib").feature_names_in_.tolist()
        == first["settings"]["feature_columns"]
    )

    run("final_model")
    second = read_json(public)
    assert second["run_id"] != first["run_id"]
    assert holdout.read_bytes() == original_holdout
    assert (
        "reused" in (root / "artifacts/runs" / second["run_id"] / "run.log").read_text()
    )
    check = tmp_path / "prediction.json"
    write_json(
        check, {"run_id": first["run_id"], "model_sha256": first["model_sha256"]}
    )
    with pytest.raises(ValueError, match="run_id"):
        read_prediction_check(public, check)

    original_config = config.read_text()
    config.write_text(original_config.replace("forest_trees = 2", "forest_trees = 3"))
    assert "holdout settings differ" in run("final_model", check=False).stderr
    config.write_text(original_config)
    values = pd.read_csv(labels)
    values.loc[0, "status_group"] = "non functional"
    values.to_csv(labels, index=False)
    assert "holdout inputs differ" in run("final_model", check=False).stderr
    assert holdout.read_bytes() == original_holdout
    assert read_json(public) == second
    assert sha256_file(root / second["model_file"]) == second["model_sha256"]


@pytest.mark.parametrize("name", ["../outside.csv", "/tmp/outside.csv"])
def test_verified_report_rejects_paths_outside_manifest(tmp_path, name):
    manifest = tmp_path / "model.json"
    write_json(manifest, {"reports": {"scores": {"file": name, "sha256": "unused"}}})
    with pytest.raises(ValueError, match="within the manifest directory"):
        verified_report(manifest, "scores")
