"""Precision, recall and F1 for each pump condition, from saved fold scores.

Run with:
    uv run python -m team_wtm.per_class_scores

Reads the fold results already saved by model_comparison.py. No model is
fitted and the labelled holdout is not used.
"""

import sys

import pandas as pd

from team_wtm.config import SETTINGS
from team_wtm.evaluation import CLASS_NAMES
from team_wtm.runtime import track_workflow

REPORT_DIR = SETTINGS.report_dir / "comparisons"
SCORE_FILES = [
    "model_cv_scores.csv",
    "quantity_cv_scores.csv",
    "location_cv_scores.csv",
]
OUTPUT_PATH = REPORT_DIR / "per_class_scores.csv"

KEY_COLUMNS = ["model", "features", "fold"]
METRICS = ["precision", "recall", "f1"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
FINAL_MODEL = SETTINGS.final_model
FINAL_FEATURES = SETTINGS.final_features


def load_fold_scores(report_dir=REPORT_DIR, file_names=SCORE_FILES):
    """Join the saved fold scores and keep each (model, features, fold) once.

    Some setups are stored in more than one file. Stop if two copies differ.
    """
    tables = [pd.read_csv(report_dir / name) for name in file_names]
    scores = pd.concat(tables, ignore_index=True)
    score_columns = [
        f"{short_name}_{metric}"
        for short_name in CLASS_NAMES.values()
        for metric in METRICS
    ]
    copies = scores.drop_duplicates(KEY_COLUMNS + score_columns)
    if copies.duplicated(KEY_COLUMNS).any():
        raise ValueError("Copies of the same fold have different scores.")
    return scores.drop_duplicates(KEY_COLUMNS).reset_index(drop=True)


def per_class_summary(scores):
    """Mean and sample std (ddof=1) over folds for each condition and metric.

    Returns one row per model, features, condition and metric.
    """
    rows = []
    for label, short_name in CLASS_NAMES.items():
        for metric in METRICS:
            grouped = scores.groupby(["model", "features"], sort=False)
            values = grouped[f"{short_name}_{metric}"]
            part = pd.DataFrame(
                {
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "folds": values.size(),
                }
            ).reset_index()
            part.insert(2, "condition", label)
            part.insert(3, "metric", metric)
            rows.append(part)
    summary = pd.concat(rows, ignore_index=True)
    return summary.sort_values(["model", "features"], kind="stable").reset_index(
        drop=True
    )


def markdown_table(summary, model=FINAL_MODEL, features=FINAL_FEATURES, digits=3):
    """Markdown table for one setup: one row per condition, mean ± std."""
    setup = summary[summary["model"].eq(model) & summary["features"].eq(features)]
    if setup.empty:
        raise ValueError(f"No scores for model {model!r} with features {features!r}.")

    header = "| Condition | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    lines = [header, "| --- |" + " ---: |" * len(METRICS)]
    for label in CLASS_NAMES:
        cells = []
        for metric in METRICS:
            row = setup[setup["condition"].eq(label) & setup["metric"].eq(metric)]
            mean, std = row["mean"].iloc[0], row["std"].iloc[0]
            cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_per_class_report(*, on_progress=None):
    """Summarize the saved fold scores without fitting a model."""
    with track_workflow(
        "per-condition-scores", config=SETTINGS,
        on_progress=on_progress, report_dir=SETTINGS.report_dir,
    ) as run:
        run.record_inputs({name: REPORT_DIR / name for name in SCORE_FILES})
        scores = load_fold_scores()
        summary = per_class_summary(scores)
        selected = scores[
            scores["model"].eq(FINAL_MODEL) & scores["features"].eq(FINAL_FEATURES)
        ]["macro_f1"]
        summary.attrs["macro_f1"] = {
            "mean": float(selected.mean()),
            "std": float(selected.std(ddof=1)),
            "folds": len(selected),
        }
        summary.to_csv(OUTPUT_PATH, index=False)
        run.archive([OUTPUT_PATH], SETTINGS.report_dir)
        return summary


def main(argv=None):
    from team_wtm.cli import main as run_cli

    run_cli(["scores", *([] if argv is None else argv)])


if __name__ == "__main__":
    main(sys.argv[1:])
