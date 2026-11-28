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
        ("region_folds = 5", "region_folds = 1", "region_folds"),
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


def test_configured_comparison_from_outside_the_project(training, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "project.toml"
    config = DEFAULT_CONFIG.read_text()
    config = config.replace("data/raw", "inputs").replace("reports/sprint2", "summary")
    config = config.replace("\nfolds = 5", "\nfolds = 3").replace(
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


def test_all_workflows_share_config_and_record_runs(training, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "project.toml"
    config = DEFAULT_CONFIG.read_text()
    config = config.replace("forest_trees = 300", "forest_trees = 2")
    config = config.replace("\nfolds = 5", "\nfolds = 3")
    config = config.replace("region_folds = 5", "region_folds = 2")
    config = config.replace("min_repairs = 30", "min_repairs = 2")
    config = config.replace("reports/figures", "charts")
    config_path.write_text(config)
    settings = load_settings(config_path)
    settings.raw_dir.mkdir(parents=True)
    training.drop(columns="status_group").to_csv(settings.training_values, index=False)
    training[["id", "status_group"]].to_csv(settings.training_labels, index=False)
    training.drop(columns="status_group").to_csv(
        settings.raw_dir / "Test_set_values.csv", index=False
    )
    training[["id", "status_group"]].to_csv(settings.submission_format, index=False)
    env = {**os.environ, "TEAM_WTM_CONFIG": str(config_path)}
    cli = Path(sys.executable).with_name("team-wtm")
    for command in (
        ["baseline"],
        ["compare"],
        ["compare", "quantity"],
        ["compare", "location"],
        ["scores"],
        ["features", "ladder"],
        ["features", "h6"],
        ["features", "groups"],
        ["features", "regions"],
        ["features", "charts"],
        ["export-cleaned"],
        ["final-model"],
        ["predict"],
    ):
        subprocess.run(
            [str(cli), *command],
            cwd=tmp_path,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    assert (settings.figures_dir / "sprint2/pm3_feature_options.png").is_file()
    assert settings.out_of_fold_predictions.is_file()
    assert settings.region_predictions.is_file()
    assert settings.cleaned_snapshot.is_file()
    assert (settings.report_dir / "final/prediction_check.json").is_file()
    records = [
        json.loads(path.read_text())
        for path in settings.workflow_runs_dir.glob("*/run.json")
    ]
    assert len(records) == 13
    assert all(record["status"] == "completed" for record in records)
    assert len({record["run_id"] for record in records}) == 13
    compare = next(
        record for record in records if record["workflow"] == "compare-models"
    )
    archived = settings.workflow_runs_dir / compare["run_id"]
    assert (archived / "reports/comparisons/model_cv_scores.csv").is_file()
    old_scores = (archived / "reports/comparisons/model_cv_scores.csv").read_bytes()
    subprocess.run(
        [str(cli), "compare"], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    assert (
        archived / "reports/comparisons/model_cv_scores.csv"
    ).read_bytes() == old_scores
    assert len(list(settings.workflow_runs_dir.glob("*/run.json"))) == 14
    # A changed preprocessing setup must not reuse the earlier reserved evaluation.
    config_path.write_text(
        config.replace(
            "minimum_category_frequency = 0.01", "minimum_category_frequency = 0.02"
        )
    )
    refused = subprocess.run(
        [str(cli), "final-model"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "holdout settings differ" in refused.stderr
    failed = [
        json.loads(path.read_text())
        for path in settings.workflow_runs_dir.glob("*/run.json")
        if json.loads(path.read_text())["status"] == "failed"
    ]
    assert len(failed) == 1

    config_path.write_text(config)
    quiet = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from team_wtm.baseline import run_baseline_workflow; run_baseline_workflow(); "
                "from team_wtm.final_model import run_final_model; run_final_model()"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert quiet.stdout == ""
    eda = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from team_wtm.target_categorical_eda import load_data; "
                "from team_wtm.model_data_preparation import load_training_data; "
                "import pandas as pd; a=load_data(); b=load_training_data(); "
                "pd.testing.assert_frame_equal(a[b.columns], b)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert eda.stdout == ""
