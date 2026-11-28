"""Unit tests for config."""

import pytest

from team_wtm.config import DEFAULT_CONFIG, load_settings


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("folds = 5", "folds = 1", "folds"),
        ("forest_jobs = -1", "forest_jobs = 0", "forest_jobs"),
        ("random_seed = 42", "random_seed = true", "random_seed"),
        ("forest_trees = 300", "forest_trees = -1", "forest_trees"),
        ("region_folds = 5", "region_folds = 1", "region_folds"),
        ("max_bytes = 5000000", "max_bytes = 0", "max_bytes"),
        ("backup_count = 3", "backup_count = 0", "backup_count"),
        ('log_file = "artifacts/logs/project.log"', 'log_file = ""', "log_file"),
        ("min_repairs = 30", "min_repairs = 0", "min_repairs"),
        (
            "minimum_category_frequency = 0.01",
            "minimum_category_frequency = true",
            "minimum_category_frequency",
        ),
        ('level = "INFO"', 'level = "INVALID"', "logging level"),
        (
            'workflow_runs_dir = "artifacts/workflows"',
            'workflow_runs_dir = ""',
            "workflow_runs_dir",
        ),
    ],
)
def test_invalid_config_names_the_setting(tmp_path, old, new, field):
    path = tmp_path / "project.toml"
    path.write_text(DEFAULT_CONFIG.read_text().replace(old, new))
    with pytest.raises(ValueError, match=field):
        load_settings(path)
