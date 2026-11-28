"""Integration tests for logging workflows."""

import logging
from dataclasses import replace

import pandas as pd

from team_wtm import cleaning
from team_wtm.config import SETTINGS
from team_wtm.logging_config import configure_logging
from team_wtm.runtime import track_workflow


def test_module_logs_reach_console_project_and_current_run_once(tmp_path, capsys):
    config = replace(
        SETTINGS,
        log_file=tmp_path / "project.log",
        workflow_runs_dir=tmp_path / "runs",
    )
    configure_logging(config)
    configure_logging(config)
    logger = logging.getLogger("team_wtm.preparation")
    logger.info("Before workflow")
    with track_workflow("preparation", config=config) as run:
        logger.info("Processing pumps")
        logger.debug("Detailed steps")
    logger.info("After workflow")
    project = config.log_file.read_text()
    current = (run.directory / "run.log").read_text()
    assert project.count("Processing pumps") == 1
    assert current.count("Processing pumps") == 1
    assert "Before workflow" in project
    assert "After workflow" in project
    assert "Before workflow" not in current
    assert "After workflow" not in current
    assert "Detailed steps" not in project
    assert "Detailed steps" not in current
    assert "Z INFO team_wtm.preparation" in current
    assert f"workflow=preparation run={run.run_id}" in current
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Processing pumps") == 1


def test_log_level_applies_to_console_and_files(tmp_path, capsys):
    config = replace(
        SETTINGS,
        log_file=tmp_path / "project.log",
        workflow_runs_dir=tmp_path / "runs",
        log_level="WARNING",
    )
    configure_logging(config)
    logger = logging.getLogger("team_wtm.preparation")
    with track_workflow("severity", config=config) as run:
        logger.info("Quiet detail")
        logger.warning("Review input")
    for text in (config.log_file.read_text(), (run.directory / "run.log").read_text()):
        assert "Review input" in text
        assert "Quiet detail" not in text
    assert "Review input" in capsys.readouterr().err


def test_audit_logs_aggregate_diagnostics_without_record_values(training, tmp_path):
    config = replace(
        SETTINGS,
        log_file=tmp_path / "project.log",
        workflow_runs_dir=tmp_path / "runs",
        log_level="DEBUG",
    )
    configure_logging(config, console=False)
    records = training.assign(pump_name="private-record-value")
    before = records.copy(deep=True)
    with track_workflow("audit", config=config) as run:
        quality = cleaning.check_data_quality(records)
        inspected = cleaning.inspect_columns(records)
    pd.testing.assert_frame_equal(records, before)
    assert inspected["cardinality_summary"].loc["pump_name", "Unique Values"] == 1
    assert "missing_values" in quality
    for text in (config.log_file.read_text(), (run.directory / "run.log").read_text()):
        assert "Missing cells" in text
        assert "Categorical cardinality" in text
        assert "private-record-value" not in text


def test_cleaning_workflow_logs_process_milestones(training, tmp_path, monkeypatch):
    values = tmp_path / "values.csv"
    labels = tmp_path / "labels.csv"
    training.drop(columns="status_group").to_csv(values, index=False)
    training[["id", "status_group"]].to_csv(labels, index=False)
    config = replace(
        SETTINGS,
        log_file=tmp_path / "project.log",
        workflow_runs_dir=tmp_path / "runs",
        training_values=values,
        training_labels=labels,
    )
    monkeypatch.setattr(cleaning, "SETTINGS", config)
    configure_logging(config, console=False)
    result = cleaning.run_pipeline(values, labels)
    expected = cleaning.clean_data(
        cleaning.merge_data(*cleaning.load_data(values, labels))
    )
    pd.testing.assert_frame_equal(result, expected)
    record = next((tmp_path / "runs").glob("*/run.log"))
    text = record.read_text()
    assert "team_wtm.cleaning" in text
    assert "Loading training values and labels" in text
    assert "Cleaning complete" in text
