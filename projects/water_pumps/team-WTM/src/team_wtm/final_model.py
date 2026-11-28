"""Score the fixed final setup once on the holdout, then refit and save it.

Run with: uv run python -m team_wtm.final_model

Fixed setup (chosen from development cross-validation, Issue #17):
Random Forest with class_weight="balanced" on PM-3's 'full' feature option.

Safety rule: the holdout is scored only once. If the holdout result file
already exists, it is read back and the holdout is not scored again.
"""

import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn

from team_wtm.artifacts import (
    archive_run,
    evaluation_inputs,
    model_manifest,
    read_json,
    report_ref,
    verified_report,
    write_json,
)
from team_wtm.config import SETTINGS
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
HOLDOUT_PATH = REPORT_DIR / "final/holdout_result.json"
FINAL_MODEL_REPORT = REPORT_DIR / "final_model.json"
# models/**/*.joblib is ignored by Git, so the model file is never committed.
MODEL_PATH = SETTINGS.model_file


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
    with TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / path.name
        joblib.dump(pipeline, candidate, compress=3)
        reloaded = joblib.load(candidate)
        same_labels = np.array_equal(
            pipeline.predict(features), reloaded.predict(features)
        )
        same_probabilities = np.allclose(
            pipeline.predict_proba(features), reloaded.predict_proba(features)
        )
        if not (same_labels and same_probabilities):
            raise ValueError("Reload check failed; cannot publish this model.")
        candidate.replace(path)
    return reloaded, {
        "same_predictions": bool(same_labels),
        "same_probabilities": bool(same_probabilities),
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock = REPORT_DIR / ".final_model.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Final-model publication already in progress: {lock}"
        ) from None
    try:
        _publish_final_model()
    finally:
        lock.rmdir()


def _publish_final_model():
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

    inputs = evaluation_inputs(
        SETTINGS.training_values, SETTINGS.training_labels, SPLIT_PATH
    )
    previous = read_json(FINAL_MODEL_REPORT) if FINAL_MODEL_REPORT.is_file() else {}
    if previous and not HOLDOUT_PATH.is_file():
        raise ValueError(
            "The previous final-model report exists but its reserved holdout result is missing."
        )
    prior_holdout = previous.get("reports", {}).get("holdout")
    if prior_holdout is not None and prior_holdout != report_ref(HOLDOUT_PATH, relative_to=REPORT_DIR):
        raise ValueError(
            "Reserved holdout file differs from the previous model manifest."
        )

    # 1. Holdout: scored once only. Saved before anything else happens.
    reused_holdout = HOLDOUT_PATH.exists()
    if reused_holdout:
        holdout = read_json(HOLDOUT_PATH)
        if holdout.get("settings") != settings:
            raise ValueError(
                "Saved holdout settings differ; the score cannot be reused."
            )
        saved_inputs = holdout.get("evaluation_inputs")
        if saved_inputs is None and previous.get("reports", {}).get(
            "holdout"
        ) == report_ref(HOLDOUT_PATH, relative_to=REPORT_DIR):
            # The original Sprint 2 holdout predates input fingerprints.
            saved_inputs = previous.get("evaluation_inputs")
        if saved_inputs != inputs:
            raise ValueError(
                "Saved holdout inputs differ or are unverified; keep the original result."
            )
        print(f"Reusing the matching holdout result at {HOLDOUT_PATH}")
    else:
        print("Scoring the fixed setup on the labelled holdout (once) ...")
        scores = score_holdout(features, target, split_positions(training, plan))
        holdout = {
            "scope": "LABELLED FINAL HOLDOUT - scored once, after model selection.",
            "settings": settings,
            "evaluation_inputs": inputs,
            "class_order": list(TARGET_CLASSES),
            "confusion_matrix_rows": "actual",
            "confusion_matrix_columns": "predicted",
            "scores": scores,
            "versions": versions(),
        }
        write_json(HOLDOUT_PATH, holdout)

    reports = {"holdout": report_ref(HOLDOUT_PATH, relative_to=REPORT_DIR)}
    if (
        previous.get("evaluation_inputs") == inputs
        and previous.get("settings") == settings
    ):
        for name, entry in previous.get("reports", {}).items():
            verified_report(FINAL_MODEL_REPORT, name)
            reports[name] = entry

    # 2. Refit the same setup on all labelled records.
    print(f"Refitting on all {len(training)} labelled records ...")
    pipeline = fit_final_pipeline(features, target)

    # 3. Save, reload and check that predictions are identical.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as tmp:
        candidate = Path(tmp) / MODEL_PATH.name
        _, reload_check = save_and_check_reload(pipeline, features, candidate)
        report = {
            "settings": settings,
            "fit_rows": len(training),
            "model_file": (
                MODEL_PATH.relative_to(SETTINGS.project_root)
                if MODEL_PATH.is_relative_to(SETTINGS.project_root)
                else MODEL_PATH
            ).as_posix(),
            "model_file_committed": False,
            "reload_check": reload_check,
            "versions": versions(),
        }
        manifest = model_manifest(report, candidate, inputs, reports)
        archive_run(
            manifest, candidate, REPORT_DIR, SETTINGS, reused_holdout=reused_holdout
        )
        candidate.replace(MODEL_PATH)
    write_json(FINAL_MODEL_REPORT, manifest)

    scores = holdout["scores"]
    print(
        f"Holdout macro F1: {scores['macro_f1']:.4f}  accuracy: {scores['accuracy']:.4f}"
    )
    print(
        f"Holdout repair recall: {scores['repair_recall']:.4f}  "
        f"precision: {scores['repair_precision']:.4f}"
    )
    print(f"Reload check: {reload_check}")


if __name__ == "__main__":
    main()
