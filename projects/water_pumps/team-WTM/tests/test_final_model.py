"""Check the one-time holdout scoring and the save/reload step (Issue #17)."""

import numpy as np

import team_wtm.final_model as fm
from team_wtm.evaluation import make_split_plan
from team_wtm.features import select_features
from team_wtm.model_comparison import add_model_columns

# The made-up `training` data comes from tests/conftest.py.


def full_features(training):
    return select_features(add_model_columns(training), fm.FINAL_FEATURES)


def test_split_positions_cover_every_row_once(training):
    plan = make_split_plan(training)
    development, holdout = fm.split_positions(training, plan)
    assert len(development) + len(holdout) == len(training)
    assert set(development).isdisjoint(holdout)
    holdout_ids = set(plan.loc[plan["split"] == "holdout", "id"])
    assert set(training["id"].iloc[holdout]) == holdout_ids


def test_holdout_model_is_fitted_on_development_rows_only(training, monkeypatch):
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
    scores = fm.score_holdout(full_features(training), training["status_group"],
                              (development, holdout))

    assert seen["fit"] == set(development)
    assert seen["predict"] == set(holdout)
    assert scores["fit_rows"] == len(development)
    assert scores["holdout_rows"] == len(holdout)


def test_saved_pipeline_gives_the_same_predictions(training, tmp_path):
    features = full_features(training)
    pipeline = fm.fit_final_pipeline(features, training["status_group"])
    path = tmp_path / "pipeline.joblib"

    reloaded, check = fm.save_and_check_reload(pipeline, features, path)

    assert path.exists()
    assert check == {"same_predictions": True, "same_probabilities": True,
                     "rows_checked": len(features)}
    # The reloaded pipeline still contains every step, not just the model.
    assert list(reloaded.named_steps) == ["categories_as_text", "preprocessor", "model"]
