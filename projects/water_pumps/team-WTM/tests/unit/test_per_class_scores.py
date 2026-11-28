"""Check the per-condition summary on small made-up fold scores."""

import numpy as np
import pandas as pd
import pytest

import team_wtm.evaluation.per_condition as pcs
from team_wtm.evaluation.splits import CLASS_NAMES


def make_scores(model="random_forest_balanced", features="full", folds=3):
    """Made-up fold scores: each column gets a different, known set of values."""
    rows = []
    for fold in range(folds):
        row = {
            "model": model,
            "features": features,
            "fold": fold,
            "macro_f1": 0.5 + fold / 100,
        }
        for position, short_name in enumerate(CLASS_NAMES.values()):
            for offset, metric in enumerate(pcs.METRICS):
                row[f"{short_name}_{metric}"] = (
                    0.1 * position + 0.01 * offset + 0.1 * fold
                )
        rows.append(row)
    return pd.DataFrame(rows)


def test_summary_has_every_condition_and_metric():
    summary = pcs.per_class_summary(make_scores())
    assert len(summary) == len(CLASS_NAMES) * len(pcs.METRICS)
    assert set(summary["condition"]) == set(CLASS_NAMES)
    assert set(summary["metric"]) == set(pcs.METRICS)
    assert summary["folds"].eq(3).all()


def test_summary_uses_mean_and_sample_std():
    summary = pcs.per_class_summary(make_scores())
    row = summary[
        summary["condition"].eq("functional needs repair")
        & summary["metric"].eq("recall")
    ].iloc[0]
    # repair recall over folds: 0.11, 0.21, 0.31.
    values = np.array([0.11, 0.21, 0.31])
    assert row["mean"] == pytest.approx(values.mean())
    assert row["std"] == pytest.approx(values.std(ddof=1))
    assert row["std"] == pytest.approx(0.1)
    assert row["std"] != pytest.approx(values.std(ddof=0))


def test_load_counts_a_fold_stored_in_two_files_once(tmp_path):
    scores = make_scores()
    scores.to_csv(tmp_path / "a.csv", index=False)
    scores.to_csv(tmp_path / "b.csv", index=False)
    loaded = pcs.load_fold_scores(tmp_path, ["a.csv", "b.csv"])
    assert len(loaded) == 3
    summary = pcs.per_class_summary(loaded)
    assert summary["folds"].eq(3).all()


def test_load_stops_if_copies_of_a_fold_differ(tmp_path):
    scores = make_scores()
    changed = scores.copy()
    changed.loc[0, "repair_f1"] = 0.99
    scores.to_csv(tmp_path / "a.csv", index=False)
    changed.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        pcs.load_fold_scores(tmp_path, ["a.csv", "b.csv"])


def test_markdown_table_has_one_row_per_condition():
    summary = pcs.per_class_summary(make_scores())
    table = pcs.markdown_table(summary).splitlines()
    assert len(table) == 2 + 3
    assert table[0] == "| Condition | Precision | Recall | F1 |"
    for label, line in zip(CLASS_NAMES, table[2:]):
        assert line.startswith(f"| {label} |")
    # repair recall: mean 0.21, std 0.1.
    assert "0.210 ± 0.100" in table[3]


def test_markdown_table_rejects_unknown_setup():
    summary = pcs.per_class_summary(make_scores())
    with pytest.raises(ValueError):
        pcs.markdown_table(summary, model="random_forest_balanced", features="base")
    with pytest.raises(ValueError):
        pcs.markdown_table(summary, model="unknown_model")
