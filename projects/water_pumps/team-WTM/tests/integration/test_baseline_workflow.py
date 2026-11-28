"""Integration tests for baseline workflow."""

import pandas as pd


def test_baseline_command_saves_results_without_git_and_keeps_existing_template(
    evaluation_training, tmp_path, monkeypatch
):
    import json
    import subprocess

    from team_wtm import baseline

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    template = report_dir / "comparisons/model_comparison_template.csv"
    template.parent.mkdir()
    template.write_text("team_note\nKeep this template edit\n")
    split_path = tmp_path / "data" / "splits.csv"
    from dataclasses import replace

    values_path = tmp_path / "values.csv"
    labels_path = tmp_path / "labels.csv"
    evaluation_training.drop(columns="status_group").to_csv(values_path, index=False)
    evaluation_training[["id", "status_group"]].to_csv(labels_path, index=False)
    monkeypatch.setattr(baseline, "SETTINGS", replace(
        baseline.SETTINGS, training_values=values_path, training_labels=labels_path,
        workflow_runs_dir=tmp_path / "workflows",
    ))
    monkeypatch.setattr(baseline, "load_training_data", lambda: evaluation_training.copy())
    monkeypatch.setattr(baseline, "REPORT_DIR", report_dir)
    monkeypatch.setattr(baseline, "SPLIT_PATH", split_path)

    def reject_external_command(*args, **kwargs):
        raise AssertionError("The baseline must not need Git or another command.")

    monkeypatch.setattr(subprocess, "run", reject_external_command)
    baseline.main()

    summary = json.loads((report_dir / "baseline/baseline_summary.json").read_text())
    scores = pd.read_csv(report_dir / "baseline/baseline_cv_scores.csv")
    assert len(scores) == 5
    assert summary["rows"]["all"] == len(evaluation_training)
    assert summary["holdout_scored"] is False
    assert summary["supplied_test_file_loaded"] is False
    assert "git_commit" not in summary
    assert "code_sha256" not in summary
    assert set(pd.read_csv(split_path)["id"]) == set(evaluation_training["id"])
    assert template.read_text() == "team_note\nKeep this template edit\n"
