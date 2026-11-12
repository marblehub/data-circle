"""Score the fixed final setup once on the holdout, then refit and save it.

Run with: uv run python -m team_wtm.final_model

Fixed setup (chosen from development cross-validation, Issue #17):
Random Forest with class_weight="balanced" on PM-3's 'full' feature option.

Safety rule: the holdout is scored only once. If the holdout result file
already exists, it is read back and the holdout is not scored again.
"""

import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from team_wtm.cleaning import PROJECT_DIR
from team_wtm.evaluation import RANDOM_SEED, TARGET_CLASSES, score_predictions
from team_wtm.features import FEATURE_OPTIONS, select_features
from team_wtm.model_comparison import (
    MODELS,
    REPORT_DIR,
    SPLIT_PATH,
    add_model_columns,
    build_pipeline,
    load_training,
)

FINAL_MODEL = "random_forest_balanced"
FINAL_FEATURES = "full"
HOLDOUT_PATH = REPORT_DIR / "holdout_result.json"
FINAL_MODEL_REPORT = REPORT_DIR / "final_model.json"
# models/**/*.joblib is ignored by Git, so the model file is never committed.
MODEL_PATH = PROJECT_DIR / "models" / "final_pipeline.joblib"


def split_positions(training, plan):
    """Return row positions of development and holdout records."""
    if set(training["id"]) != set(plan["id"]) or plan["id"].duplicated().any():
        raise ValueError("The split plan must match training IDs exactly.")
    split = plan.set_index("id").loc[training["id"], "split"].to_numpy()
    development = np.flatnonzero(split == "development")
    holdout = np.flatnonzero(split == "holdout")
    if len(development) + len(holdout) != len(training):
        raise ValueError("Unknown split name.")
    return development, holdout


def fit_final_pipeline(features, target):
    """Fit a fresh copy of the fixed setup on the given rows."""
    pipeline = build_pipeline(features, MODELS[FINAL_MODEL]())
    return pipeline.fit(features, target)


def score_holdout(features, target, plan_split):
    """Fit on all development rows, then score the holdout rows once."""
    development, holdout = plan_split
    pipeline = fit_final_pipeline(features.iloc[development], target.iloc[development])
    predictions = pipeline.predict(features.iloc[holdout])
    scores = score_predictions(target.iloc[holdout], predictions)
    scores["fit_rows"] = len(development)
    scores["holdout_rows"] = len(holdout)
    return scores


def save_and_check_reload(pipeline, features, path):
    """Save with joblib, load it back and compare predictions row by row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    reloaded = joblib.load(path)
    same_labels = bool(
        np.array_equal(pipeline.predict(features), reloaded.predict(features))
    )
    same_probabilities = bool(
        np.allclose(pipeline.predict_proba(features), reloaded.predict_proba(features))
    )
    return reloaded, {
        "same_predictions": same_labels,
        "same_probabilities": same_probabilities,
        "rows_checked": len(features),
    }


def versions():
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def main():
    training = load_training()
    plan = pd.read_csv(SPLIT_PATH)
    features = select_features(add_model_columns(training), FINAL_FEATURES)
    target = training["status_group"]
    settings = {
        "model": FINAL_MODEL,
        "features": FINAL_FEATURES,
        "feature_columns": FEATURE_OPTIONS[FINAL_FEATURES],
        "random_seed": RANDOM_SEED,
        "model_settings": MODELS[FINAL_MODEL]().get_params(),
    }

    # 1. Holdout: scored once only. Saved before anything else happens.
    if HOLDOUT_PATH.exists():
        print(f"Holdout already scored; not scoring again. See {HOLDOUT_PATH}")
        holdout = json.loads(HOLDOUT_PATH.read_text())
    else:
        print("Scoring the fixed setup on the labelled holdout (once) ...")
        scores = score_holdout(features, target, split_positions(training, plan))
        holdout = {
            "scope": "LABELLED FINAL HOLDOUT - scored once, after model selection.",
            "settings": settings,
            "class_order": list(TARGET_CLASSES),
            "confusion_matrix_rows": "actual",
            "confusion_matrix_columns": "predicted",
            "scores": scores,
            "versions": versions(),
        }
        HOLDOUT_PATH.write_text(json.dumps(holdout, indent=2) + "\n")

    # 2. Refit the same setup on all labelled records.
    print(f"Refitting on all {len(training)} labelled records ...")
    pipeline = fit_final_pipeline(features, target)

    # 3. Save, reload and check that predictions are identical.
    _, reload_check = save_and_check_reload(pipeline, features, MODEL_PATH)
    report = {
        "settings": settings,
        "fit_rows": len(training),
        "model_file": MODEL_PATH.relative_to(PROJECT_DIR).as_posix(),
        "model_file_committed": False,
        "reload_check": reload_check,
        "versions": versions(),
    }
    FINAL_MODEL_REPORT.write_text(json.dumps(report, indent=2) + "\n")

    scores = holdout["scores"]
    print(f"Holdout macro F1: {scores['macro_f1']:.4f}  accuracy: {scores['accuracy']:.4f}")
    print(f"Holdout repair recall: {scores['repair_recall']:.4f}  "
          f"precision: {scores['repair_precision']:.4f}")
    print(f"Reload check: {reload_check}")


if __name__ == "__main__":
    main()
