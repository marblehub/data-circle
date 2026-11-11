"""Check fold isolation, ID alignment and majority-baseline scoring."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from team_wtm.baseline import run_baseline
from team_wtm.evaluation import (
    TARGET_CLASSES,
    development_folds,
    make_split_plan,
    score_predictions,
)


@pytest.fixture
def training():
    # Unique feature values give enough independent groups for each class.
    rows = []
    for class_number, label in enumerate(TARGET_CLASSES):
        for number in range(30):
            rows.append(
                {
                    "id": len(rows),
                    "feature": class_number * 100 + number,
                    "status_group": label,
                }
            )
    # Matching features with different labels must still stay together.
    rows.extend(
        [
            {"id": 90, "feature": 0, "status_group": "non functional"},
            {"id": 91, "feature": 0, "status_group": "functional"},
        ]
    )
    return pd.DataFrame(rows)


def test_groups_stay_together_and_each_development_row_is_validated_once(training):
    plan = make_split_plan(training)
    assert plan["id"].is_unique
    assert set(plan["id"]) == set(training["id"])
    duplicate_rows = plan[plan["id"].isin([0, 90, 91])]
    assert duplicate_rows["group"].nunique() == 1
    assert duplicate_rows["split"].nunique() == 1
    assert duplicate_rows["cv_fold"].nunique() == 1
    seen = []
    holdout_ids = set(plan.loc[plan["split"].eq("holdout"), "id"])
    for _, fit_rows, validation_rows in development_folds(training, plan):
        fit_ids = set(training.iloc[fit_rows]["id"])
        validation_ids = set(training.iloc[validation_rows]["id"])
        assert not fit_ids & validation_ids
        assert not (fit_ids | validation_ids) & holdout_ids
        fit_groups = set(plan.loc[plan["id"].isin(fit_ids), "group"])
        validation_groups = set(plan.loc[plan["id"].isin(validation_ids), "group"])
        assert not fit_groups & validation_groups
        seen.extend(validation_ids)
    assert len(seen) == len(set(seen))
    assert set(seen) == set(training["id"]) - holdout_ids


def test_split_plan_does_not_change_when_input_rows_are_shuffled(training):
    original = training.copy(deep=True)
    assert_frame_equal(
        make_split_plan(training),
        make_split_plan(training.sample(frac=1, random_state=7)),
    )
    assert_frame_equal(training, original)


def test_baseline_matches_plan_by_id_and_never_uses_holdout_labels(training):
    plan = make_split_plan(training)
    expected, details = run_baseline(training, plan)
    changed = training.sample(frac=1, random_state=3).copy()
    holdout_ids = plan.loc[plan["split"].eq("holdout"), "id"]
    changed.loc[changed["id"].isin(holdout_ids), "status_group"] = "non functional"
    actual, actual_details = run_baseline(changed, plan)
    assert_frame_equal(actual, expected)
    assert actual_details == details


def test_majority_is_learned_from_each_fit_fold(training):
    plan = make_split_plan(training)
    changed = training.copy()
    first_fold, fit_rows, validation_rows = development_folds(changed, plan)[0]
    changed.loc[changed.index[fit_rows], "status_group"] = "functional needs repair"
    changed.loc[changed.index[validation_rows], "status_group"] = "non functional"
    results, details = run_baseline(changed, plan)
    assert details[0]["predicted_class"] == "functional needs repair"
    first_result = results.loc[results["fold"].eq(first_fold)].iloc[0]
    assert first_result["accuracy"] == 0
    assert first_result["false_repair_alarms"] == len(validation_rows)


def test_scores_keep_unpredicted_classes_and_count_repair_errors():
    actual = ["functional", "functional needs repair", "non functional"]
    predicted = ["functional needs repair", "functional", "non functional"]
    scores = score_predictions(actual, predicted)
    assert scores["accuracy"] == pytest.approx(1 / 3)
    assert scores["macro_f1"] == pytest.approx(1 / 3)
    assert scores["missed_repairs"] == scores["false_repair_alarms"] == 1
    assert scores["confusion_matrix"] == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    majority = score_predictions(actual, ["functional"] * 3)
    assert majority["repair_recall"] == majority["repair_precision"] == 0
    assert majority["repair_support"] == 1


def test_invalid_ids_labels_and_plan_coverage_are_rejected(training):
    with pytest.raises(ValueError, match="present and unique"):
        make_split_plan(pd.concat([training, training.iloc[[0]]]))
    invalid = training.copy()
    invalid.loc[0, "status_group"] = "predicted label"
    with pytest.raises(ValueError, match="known labels"):
        make_split_plan(invalid)
    plan = make_split_plan(training)
    with pytest.raises(ValueError, match="match training IDs"):
        list(development_folds(training, plan.iloc[1:]))
    with pytest.raises(ValueError, match="known label names"):
        score_predictions(["functional"], ["predicted label"])


def test_modified_plan_cannot_put_matching_features_in_different_folds(training):
    plan = make_split_plan(training)
    matching = plan["id"].isin([0, 90, 91])
    plan.loc[matching, "split"] = "development"
    plan.loc[matching, "cv_fold"] = [0, 1, 1]
    with pytest.raises(ValueError, match="crossed development folds"):
        list(development_folds(training, plan))


def test_baseline_command_saves_results_without_git_and_keeps_existing_template(
    training, tmp_path, monkeypatch
):
    import json
    import subprocess

    from team_wtm import baseline

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    template = report_dir / "model_comparison_template.csv"
    template.write_text("team_note\nKeep this template edit\n")
    split_path = tmp_path / "data" / "splits.csv"
    values = training.drop(columns="status_group")
    labels = training[["id", "status_group"]]
    monkeypatch.setattr(baseline, "load_data", lambda: (values, labels))
    monkeypatch.setattr(baseline, "REPORT_DIR", report_dir)
    monkeypatch.setattr(baseline, "SPLIT_PATH", split_path)

    def reject_external_command(*args, **kwargs):
        raise AssertionError("The baseline must not need Git or another command.")

    monkeypatch.setattr(subprocess, "run", reject_external_command)
    baseline.main()

    summary = json.loads((report_dir / "baseline_summary.json").read_text())
    scores = pd.read_csv(report_dir / "baseline_cv_scores.csv")
    assert len(scores) == 5
    assert summary["rows"]["all"] == len(training)
    assert summary["holdout_scored"] is False
    assert summary["supplied_test_file_loaded"] is False
    assert "git_commit" not in summary
    assert "code_sha256" not in summary
    assert set(pd.read_csv(split_path)["id"]) == set(training["id"])
    assert template.read_text() == "team_note\nKeep this template edit\n"
