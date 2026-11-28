"""Exercise the same settings through the legacy commands and the project CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from team_wtm.config import DEFAULT_CONFIG, load_settings


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("folds = 5", "folds = 1", "folds"),
        ("forest_jobs = -1", "forest_jobs = 0", "forest_jobs"),
        ("random_seed = 42", "random_seed = true", "random_seed"),
        ("forest_trees = 300", "forest_trees = -1", "forest_trees"),
    ],
)
def test_invalid_config_names_the_setting(tmp_path, old, new, field):
    path = tmp_path / "project.toml"
    path.write_text(DEFAULT_CONFIG.read_text().replace(old, new))
    with pytest.raises(ValueError, match=field):
        load_settings(path)


def test_configured_comparison_from_outside_the_project(training, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "project.toml"
    config = DEFAULT_CONFIG.read_text()
    config = config.replace("data/raw", "inputs").replace("reports/sprint2", "summary")
    config = config.replace("folds = 5", "folds = 3").replace(
        "random_seed = 42", "random_seed = 19"
    )
    config = config.replace("forest_trees = 300", "forest_trees = 2")
    start = config.index("candidates = ")
    end = config.index("\n", start)
    config = config[:start] + 'candidates = ["random_forest"]' + config[end:]
    path.write_text(config)
    settings = load_settings(path)
    assert settings.project_root == tmp_path
    assert settings.forest_trees == 2
    settings.raw_dir.mkdir()
    training.drop(columns="status_group").to_csv(settings.training_values, index=False)
    training[["id", "status_group"]].to_csv(settings.training_labels, index=False)

    env = {**os.environ, "TEAM_WTM_CONFIG": str(path)}
    python = Path(sys.executable)
    # The console script is installed beside the interpreter by uv.
    cli = python.with_name("team-wtm.exe" if os.name == "nt" else "team-wtm")

    def run(*args):
        return subprocess.run(
            args, cwd=tmp_path, env=env, check=True, capture_output=True, text=True
        )

    run(python, "-m", "team_wtm.baseline")
    output = run(cli, "compare").stdout
    assert "Training random_forest" in output
    assert "macro_f1_mean" in output
    scores_path = settings.report_dir / "comparisons/model_cv_scores.csv"
    scores = pd.read_csv(scores_path)
    assert set(scores["model"]) == {"majority_baseline", "random_forest"}
    assert set(scores["fold"]) == {0, 1, 2}
    assert set(scores["random_seed"]) == {19}
    summary = json.loads(
        (settings.report_dir / "comparisons/model_comparison_summary.json").read_text()
    )
    assert summary["holdout_scored"] is False

    run(python, "-m", "team_wtm.model_comparison", "models")
    pd.testing.assert_frame_equal(scores, pd.read_csv(scores_path))

    direct = run(
        python,
        "-c",
        "from team_wtm.model_comparison import run_comparison; "
        "run_comparison().to_csv('returned_table.csv')",
    )
    assert direct.stdout == ""
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "returned_table.csv"),
        pd.read_csv(settings.report_dir / "comparisons/model_comparison_table.csv"),
    )
