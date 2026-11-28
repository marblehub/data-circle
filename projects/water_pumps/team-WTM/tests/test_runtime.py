"""Check durable workflow records, failure recovery and concurrent logging."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from team_wtm.artifacts import read_json, sha256_file
from team_wtm.config import SETTINGS
from team_wtm.runtime import track_workflow


def test_successful_and_failed_runs_preserve_their_own_reports(tmp_path, capsys):
    config = replace(SETTINGS, workflow_runs_dir=tmp_path / "runs")
    source = tmp_path / "source.csv"
    source.write_text("score\n0.5\n")
    with track_workflow("comparison", config=config) as first:
        first.record_inputs({"scores": source})
        first.archive([source], tmp_path)
        first.progress("First comparison")
    old = read_json(first.directory / "run.json")
    assert old["status"] == "completed"
    assert old["schema_version"] == 1
    assert old["inputs"]["scores"]["path"] == str(source.resolve())
    assert old["inputs"]["scores"]["sha256"] == sha256_file(source)
    assert (first.directory / "reports/source.csv").read_text() == "score\n0.5\n"
    assert read_json(first.directory / "config_snapshot.json")["forest_trees"] == 300

    with (
        pytest.raises(ValueError, match="bad setup"),
        track_workflow("comparison", config=config) as failed,
    ):
        failed.progress("Second comparison")
        raise ValueError("bad setup")
    failure = read_json(failed.directory / "run.json")
    assert failure["status"] == "failed"
    assert failure["error"]["type"] == "ValueError"
    assert "Traceback" in (failed.directory / "run.log").read_text()
    assert "bad setup" in (failed.directory / "run.log").read_text()
    assert first.run_id != failed.run_id
    assert read_json(first.directory / "run.json") == old
    assert capsys.readouterr().out == ""


def test_concurrent_logs_stay_with_their_run(tmp_path):
    config = replace(SETTINGS, workflow_runs_dir=tmp_path / "runs")
    barrier = Barrier(2)

    def execute(name):
        with track_workflow(name, config=config) as run:
            barrier.wait(timeout=10)
            run.progress(name)
            barrier.wait(timeout=10)
        return run.directory

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(execute, ["marker_one", "marker_two"]))
    for own, other, path in zip(
        ["marker_one", "marker_two"], ["marker_two", "marker_one"], paths
    ):
        text = (path / "run.log").read_text()
        assert own in text
        assert other not in text


def test_shared_report_publication_lock_is_released_after_failure(tmp_path):
    config = replace(SETTINGS, workflow_runs_dir=tmp_path / "runs")
    reports = tmp_path / "reports"
    with track_workflow("first", config=config, report_dir=reports):
        with (
            pytest.raises(RuntimeError, match="Another workflow"),
            track_workflow("second", config=config, report_dir=reports),
        ):
            pytest.fail("An overlapping publisher acquired the lock")
        assert (reports / ".workflow.lock").is_dir()
    assert not (reports / ".workflow.lock").exists()
    with track_workflow("third", config=config, report_dir=reports):
        pass
    assert not (reports / ".workflow.lock").exists()


def test_changed_input_is_recorded_as_failed(tmp_path):
    config = replace(SETTINGS, workflow_runs_dir=tmp_path / "runs")
    source = tmp_path / "input.csv"
    source.write_text("before")
    with (
        pytest.raises(RuntimeError, match="Input changed"),
        track_workflow("comparison", config=config) as run,
    ):
        run.record_inputs({"input": source})
        source.write_text("after")
    assert read_json(run.directory / "run.json")["status"] == "failed"


def test_progress_callback_failure_is_recorded(tmp_path):
    config = replace(SETTINGS, workflow_runs_dir=tmp_path / "runs")

    def broken_callback(message):
        raise ValueError("display failed")

    with (
        pytest.raises(ValueError, match="display failed"),
        track_workflow("callback", config=config, on_progress=broken_callback),
    ):
        pass
    record = next((tmp_path / "runs").glob("*/run.json"))
    assert read_json(record)["status"] == "failed"
