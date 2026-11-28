"""Unit tests for final model."""

import numpy as np

import team_wtm.final_model as fm
from team_wtm.evaluation import make_split_plan


def test_split_positions_cover_every_row_once(training):
    plan = make_split_plan(training)
    development, holdout = fm.split_positions(training, plan)
    assert len(development) + len(holdout) == len(training)
    assert set(development).isdisjoint(holdout)
    holdout_ids = set(plan.loc[plan["split"] == "holdout", "id"])
    assert set(training["id"].iloc[holdout]) == holdout_ids


def test_holdout_model_is_fitted_on_development_rows_only(full_features, training, monkeypatch):
    plan = make_split_plan(training)
    development, holdout = fm.split_positions(training, plan)
    seen = {}

    class RecordingPipeline:
        def fit(self, features, target):
            seen["fit"] = set(features.index)
            return self

        def predict(self, features):
            seen["predict"] = set(features.index)
            return np.array(["functional"] * len(features))

    monkeypatch.setattr(fm, "build_pipeline", lambda features, model: RecordingPipeline())
    scores = fm.score_holdout(full_features, training["status_group"],
                              (development, holdout))

    assert seen["fit"] == set(development)
    assert seen["predict"] == set(holdout)
    assert scores["fit_rows"] == len(development)
    assert scores["holdout_rows"] == len(holdout)
