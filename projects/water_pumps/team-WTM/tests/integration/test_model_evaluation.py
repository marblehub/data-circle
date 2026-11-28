"""Integration tests for model evaluation."""

import numpy as np
import pandas as pd
import pytest

import team_wtm.model_comparison as mc
from team_wtm.evaluation import RESULT_COLUMNS, make_split_plan


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


def test_holdout_labels_do_not_change_development_scores(training):
    plan = make_split_plan(training)
    before, _ = mc.run_cross_validation(training, plan, "logistic_regression")

    # Flip every holdout label. Development scores must stay exactly the same.
    changed = training.copy()
    is_holdout = plan.set_index("id").loc[changed["id"], "split"].eq("holdout").to_numpy()
    changed.loc[is_holdout, "status_group"] = "non functional"
    after, _ = mc.run_cross_validation(changed, plan, "logistic_regression")

    pd.testing.assert_frame_equal(before, after)
