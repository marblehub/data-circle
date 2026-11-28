"""Check source-package APIs, legacy entry points and dashboard imports."""

import importlib
import os
import subprocess
import sys

import pytest

from team_wtm.config import DEFAULT_CONFIG

MODULES = [
    ("cleaning", "data.cleaning"),
    ("export_cleaned", "data.export"),
    ("features", "modeling.features"),
    ("final_model", "modeling.training"),
    ("predict", "modeling.predict"),
    ("baseline", "evaluation.baseline"),
    ("model_comparison", "evaluation.comparison"),
    ("per_class_scores", "evaluation.per_condition"),
    ("additional_eda", "analysis.numeric"),
    ("target_categorical_eda", "analysis.categorical"),
    ("feature_checks", "analysis.feature_checks"),
    ("feature_check_charts", "analysis.charts"),
]
COMMANDS = [
    pair
    for pair in MODULES
    if pair[0]
    not in {"cleaning", "features", "target_categorical_eda", "feature_check_charts"}
]


@pytest.fixture
def deployment(tmp_path):
    config = tmp_path / "configs/project.toml"
    config.parent.mkdir()
    config.write_text(DEFAULT_CONFIG.read_text())
    return {**os.environ, "TEAM_WTM_CONFIG": str(config)}


@pytest.mark.parametrize(("legacy", "canonical"), MODULES)
def test_legacy_modules_share_the_canonical_implementation(legacy, canonical):
    assert importlib.import_module("team_wtm." + legacy) is importlib.import_module(
        "team_wtm." + canonical
    )


def test_preparation_compatibility_includes_saved_pipeline_functions():
    from team_wtm import model_data_preparation as legacy
    from team_wtm.data import preparation
    from team_wtm.modeling import preprocessing

    assert legacy.load_training_data is preparation.load_training_data
    assert legacy.prepare_model_inputs is preparation.prepare_model_inputs
    assert legacy.build_pipeline is preprocessing.build_pipeline
    assert (
        legacy._categorical_values_as_text is preprocessing._categorical_values_as_text
    )
    comparison = importlib.import_module("team_wtm.model_comparison")
    assert comparison.categories_as_text is preprocessing.categories_as_text


def test_legacy_evaluation_exports_support_split_and_scoring_callers():
    from team_wtm import evaluation
    from team_wtm.evaluation import splits

    assert evaluation.make_split_plan is splits.make_split_plan
    assert evaluation.development_folds is splits.development_folds
    assert evaluation.score_predictions is splits.score_predictions


@pytest.mark.parametrize(("legacy", "canonical"), COMMANDS)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_module_help_works_without_datasets_or_models(
    tmp_path, deployment, legacy, canonical, use_legacy
):
    module = legacy if use_legacy else canonical
    result = subprocess.run(
        [sys.executable, "-m", "team_wtm." + module, "--help"],
        cwd=tmp_path,
        env=deployment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "usage:" in result.stdout.lower()
    assert "RuntimeWarning" not in result.stderr


def test_prediction_import_is_independent_of_training_and_analysis(
    tmp_path, deployment
):
    code = (
        "import sys; from team_wtm.modeling.predict import load_model, predict_records; "
        "assert 'team_wtm.modeling.training' not in sys.modules; "
        "assert 'team_wtm.evaluation.comparison' not in sys.modules; "
        "assert 'team_wtm.analysis.numeric' not in sys.modules; "
        "assert 'streamlit' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=deployment,
        text=True,
        capture_output=True,
        check=True,
    )


def test_report_reader_import_is_independent_of_runtime_configuration(tmp_path):
    code = (
        "import sys; from team_wtm.artifacts import verified_report; "
        "assert 'team_wtm.config' not in sys.modules; "
        "assert 'team_wtm.modeling.predict' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
