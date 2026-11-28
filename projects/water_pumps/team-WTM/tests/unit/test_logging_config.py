"""Unit tests for logging config."""

import logging
from dataclasses import replace

from team_wtm.config import SETTINGS
from team_wtm.logging_config import configure_logging, shutdown_logging


def test_project_log_rotation_keeps_configured_backups(tmp_path):
    config = replace(
        SETTINGS,
        log_file=tmp_path / "project.log",
        log_max_bytes=256,
        log_backup_count=2,
    )
    configure_logging(config, console=False)
    logger = logging.getLogger("team_wtm.rotation")
    for index in range(30):
        logger.info("marker-%02d %s", index, "x" * 40)
    shutdown_logging()
    assert {path.name for path in tmp_path.glob("project.log*")} == {
        "project.log",
        "project.log.1",
        "project.log.2",
    }
    assert "marker-29" in config.log_file.read_text()
