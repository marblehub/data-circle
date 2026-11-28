"""Unit tests for artifacts."""

import pytest

from team_wtm.artifacts import (
    report_ref,
    verified_report,
    write_json,
)


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


@pytest.mark.parametrize("name", ["../outside.csv", "/tmp/outside.csv"])
def test_verified_report_rejects_paths_outside_manifest(tmp_path, name):
    manifest = tmp_path / "model.json"
    write_json(manifest, {"reports": {"scores": {"file": name, "sha256": "unused"}}})
    with pytest.raises(ValueError, match="within the manifest directory"):
        verified_report(manifest, "scores")
