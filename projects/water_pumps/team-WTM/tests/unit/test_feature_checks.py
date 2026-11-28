"""Checks for the Issue #16 group results and region folds (no model training)."""

import pandas as pd
import pytest

from team_wtm.analysis.feature_checks import group_results, region_folds

REPAIR = "functional needs repair"


@pytest.fixture
def predictions():
    # Region A: 3 real repairs, 1 found. Region B: no repairs at all.
    return pd.DataFrame({
        "actual": [REPAIR, REPAIR, REPAIR, "functional", "functional", "non functional"],
        "predicted": [REPAIR, "functional", "non functional", REPAIR, "functional", "non functional"],
        "region": ["A", "A", "A", "A", "B", "B"],
        "age": ["known", "known", "unknown", "known", "unknown", "unknown"],
    })


def test_group_counts_and_missed_repairs(predictions):
    table = group_results(predictions, "region", min_repairs=2).set_index("region")
    assert table.loc["A", "records"] == 4
    assert table.loc["A", "repair_records"] == 3
    assert table.loc["A", "missed_repairs"] == 2
    assert table.loc["A", "false_repair_alarms"] == 1
    assert table.loc["A", "repair_recall"] == pytest.approx(1 / 3)


def test_small_groups_are_flagged_not_dropped(predictions):
    table = group_results(predictions, "region", min_repairs=2).set_index("region")
    assert not table.loc["A", "too_small"]
    assert table.loc["B", "too_small"]  # 0 repairs < 2
    assert pd.isna(table.loc["B", "repair_recall"])  # no repairs: recall undefined
    assert table["records"].sum() == len(predictions)


def test_two_key_groups(predictions):
    table = group_results(predictions, ["region", "age"], min_repairs=1)
    assert len(table) == 3  # A/known, A/unknown, B/unknown
    assert table["records"].sum() == len(predictions)


def test_region_folds_never_share_a_region_and_skip_holdout():
    regions = [r for r in "ABCDEFGHIJ" for _ in range(4)]
    training = pd.DataFrame({"id": range(40), "region": regions})
    plan = pd.DataFrame({
        "id": range(40),
        "split": ["holdout" if i % 10 == 0 else "development" for i in range(40)],
    })
    folds = region_folds(training, plan, n_folds=5)
    holdout = set(plan.index[plan["split"].eq("holdout")])
    validated = []
    for _, fit_rows, validation_rows, fold_regions in folds:
        fit_regions = set(training["region"].iloc[fit_rows])
        assert not fit_regions & set(fold_regions)
        assert not holdout & (set(fit_rows) | set(validation_rows))
        validated.extend(validation_rows)
    # Every development record is validated exactly once.
    assert sorted(validated) == sorted(set(range(40)) - holdout)
