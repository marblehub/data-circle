"""Check the Issue #17 training loop on small made-up data."""

import numpy as np
import pandas as pd
import pytest

import team_wtm.model_comparison as mc
from team_wtm.evaluation import RESULT_COLUMNS, make_split_plan

# The made-up `training` data comes from tests/conftest.py.


def test_categories_as_text_keeps_missing_values():
    data = pd.DataFrame({"permit": [True, False, None], "population": [1.0, 2.0, 3.0],
                         "pump_age_missing": [True, False, True]})
    output = mc.categories_as_text(data)
    assert output["permit"].tolist()[:2] == ["True", "False"]
    # Pure True/False flag columns become text too.
    assert output["pump_age_missing"].tolist() == ["True", "False", "True"]
    assert pd.isna(output.loc[2, "permit"])
    # Numbers are left unchanged.
    assert output["population"].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("model_name", list(mc.MODELS))
def test_each_model_gives_one_scored_row_per_fold(training, model_name):
    plan = make_split_plan(training)
    rows, matrices = mc.run_cross_validation(training, plan, model_name)

    assert rows.columns.tolist() == RESULT_COLUMNS
    assert rows["fold"].tolist() == [0, 1, 2, 3, 4]
    development = int(plan["split"].eq("development").sum())
    assert (rows["train_rows"] + rows["validation_rows"]).eq(development).all()
    assert rows["validation_rows"].sum() == development
    assert rows["model"].eq(model_name).all()
    assert len(matrices) == 5
    # Every validation row appears once in its fold's confusion matrix.
    for row, matrix in zip(rows.itertuples(), matrices):
        assert np.array(matrix["confusion_matrix"]).sum() == row.validation_rows


def test_fit_rows_never_include_validation_or_holdout_rows(training, monkeypatch):
    plan = make_split_plan(training)
    holdout_rows = set(training.index[plan.set_index("id").loc[training["id"], "split"]
                                      .eq("holdout").to_numpy()])
    seen = []

    class RecordingPipeline:
        """Stands in for the real pipeline and records which rows it sees."""

        def fit(self, features, target):
            self.fit_rows = set(features.index)
            return self

        def predict(self, features):
            seen.append((self.fit_rows, set(features.index)))
            return np.array(["functional"] * len(features))

    monkeypatch.setattr(mc, "build_pipeline", lambda fit_features, model: RecordingPipeline())
    mc.run_cross_validation(training, plan, "logistic_regression")

    assert len(seen) == 5
    for fit_rows, validation_rows in seen:
        assert fit_rows.isdisjoint(validation_rows)
        assert fit_rows.isdisjoint(holdout_rows)
        assert validation_rows.isdisjoint(holdout_rows)


def made_up_results():
    """Two models with three folds of easy-to-check scores."""
    rows = []
    for model, macro_f1 in [("small", [0.1, 0.2, 0.3]), ("big", [0.5, 0.7, 0.9])]:
        for fold, score in enumerate(macro_f1):
            row = {column: 0.0 for column in mc.SUMMARY_METRICS}
            row.update({"model": model, "features": "base", "fold": fold,
                        "macro_f1": score, "missed_repairs": fold * 10})
            rows.append(row)
    return pd.DataFrame(rows)


def test_summarize_folds_uses_mean_and_sample_std():
    summary = mc.summarize_folds(made_up_results())
    big = summary.loc[("big", "base")]
    assert big["folds"] == 3
    assert big["macro_f1_mean"] == pytest.approx(0.7)
    # Sample std (ddof=1) of 0.5, 0.7, 0.9 is 0.2. Population std would be 0.163.
    assert big["macro_f1_std"] == pytest.approx(0.2)
    assert big["missed_repairs_mean"] == pytest.approx(10)
    assert big["missed_repairs_std"] == pytest.approx(10)
    # Each mean sits next to its std.
    assert summary.columns.tolist()[:3] == ["folds", "macro_f1_mean", "macro_f1_std"]


def test_comparison_table_puts_best_macro_f1_first():
    table = mc.comparison_table(made_up_results())
    assert table.index.get_level_values("model").tolist() == ["big", "small"]


def test_holdout_labels_do_not_change_development_scores(training):
    plan = make_split_plan(training)
    before, _ = mc.run_cross_validation(training, plan, "logistic_regression")

    # Flip every holdout label. Development scores must stay exactly the same.
    changed = training.copy()
    is_holdout = plan.set_index("id").loc[changed["id"], "split"].eq("holdout").to_numpy()
    changed.loc[is_holdout, "status_group"] = "non functional"
    after, _ = mc.run_cross_validation(changed, plan, "logistic_regression")

    pd.testing.assert_frame_equal(before, after)
