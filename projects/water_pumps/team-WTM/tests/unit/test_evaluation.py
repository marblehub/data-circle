"""Unit tests for evaluation."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from team_wtm.evaluation.baseline import run_baseline
from team_wtm.evaluation.splits import (
    development_folds,
    make_split_plan,
    score_predictions,
)


def test_groups_stay_together_and_each_development_row_is_validated_once(evaluation_training):
    plan = make_split_plan(evaluation_training)
    assert plan["id"].is_unique
    assert set(plan["id"]) == set(evaluation_training["id"])
    duplicate_rows = plan[plan["id"].isin([0, 90, 91])]
    assert duplicate_rows["group"].nunique() == 1
    assert duplicate_rows["split"].nunique() == 1
    assert duplicate_rows["cv_fold"].nunique() == 1
    seen = []
    holdout_ids = set(plan.loc[plan["split"].eq("holdout"), "id"])
    for _, fit_rows, validation_rows in development_folds(evaluation_training, plan):
        fit_ids = set(evaluation_training.iloc[fit_rows]["id"])
        validation_ids = set(evaluation_training.iloc[validation_rows]["id"])
        assert not fit_ids & validation_ids
        assert not (fit_ids | validation_ids) & holdout_ids
        fit_groups = set(plan.loc[plan["id"].isin(fit_ids), "group"])
        validation_groups = set(plan.loc[plan["id"].isin(validation_ids), "group"])
        assert not fit_groups & validation_groups
        seen.extend(validation_ids)
    assert len(seen) == len(set(seen))
    assert set(seen) == set(evaluation_training["id"]) - holdout_ids


def test_split_plan_does_not_change_when_input_rows_are_shuffled(evaluation_training):
    original = evaluation_training.copy(deep=True)
    assert_frame_equal(
        make_split_plan(evaluation_training),
        make_split_plan(evaluation_training.sample(frac=1, random_state=7)),
    )
    assert_frame_equal(evaluation_training, original)


def test_baseline_matches_plan_by_id_and_never_uses_holdout_labels(evaluation_training):
    plan = make_split_plan(evaluation_training)
    expected, details = run_baseline(evaluation_training, plan)
    changed = evaluation_training.sample(frac=1, random_state=3).copy()
    holdout_ids = plan.loc[plan["split"].eq("holdout"), "id"]
    changed.loc[changed["id"].isin(holdout_ids), "status_group"] = "non functional"
    actual, actual_details = run_baseline(changed, plan)
    assert_frame_equal(actual, expected)
    assert actual_details == details


def test_majority_is_learned_from_each_fit_fold(evaluation_training):
    plan = make_split_plan(evaluation_training)
    changed = evaluation_training.copy()
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


def test_invalid_ids_labels_and_plan_coverage_are_rejected(evaluation_training):
    with pytest.raises(ValueError, match="present and unique"):
        make_split_plan(pd.concat([evaluation_training, evaluation_training.iloc[[0]]]))
    invalid = evaluation_training.copy()
    invalid.loc[0, "status_group"] = "predicted label"
    with pytest.raises(ValueError, match="known labels"):
        make_split_plan(invalid)
    plan = make_split_plan(evaluation_training)
    with pytest.raises(ValueError, match="match training IDs"):
        list(development_folds(evaluation_training, plan.iloc[1:]))
    with pytest.raises(ValueError, match="known label names"):
        score_predictions(["functional"], ["predicted label"])


def test_modified_plan_cannot_put_matching_features_in_different_folds(evaluation_training):
    plan = make_split_plan(evaluation_training)
    matching = plan["id"].isin([0, 90, 91])
    plan.loc[matching, "split"] = "development"
    plan.loc[matching, "cv_fold"] = [0, 1, 1]
    with pytest.raises(ValueError, match="crossed development folds"):
        list(development_folds(evaluation_training, plan))
