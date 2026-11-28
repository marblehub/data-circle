"""Integration tests for model reload."""

import team_wtm.final_model as fm


def test_saved_pipeline_gives_the_same_predictions(full_features, training, tmp_path):
    features = full_features
    pipeline = fm.fit_final_pipeline(features, training["status_group"])
    path = tmp_path / "pipeline.joblib"

    reloaded, check = fm.save_and_check_reload(pipeline, features, path)

    assert path.exists()
    assert check == {"same_predictions": True, "same_probabilities": True,
                     "rows_checked": len(features)}
    # The reloaded pipeline still contains every step, not just the model.
    assert list(reloaded.named_steps) == ["preprocessor", "model"]
