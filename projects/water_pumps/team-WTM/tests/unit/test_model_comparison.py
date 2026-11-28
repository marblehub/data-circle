"""Unit tests for model comparison."""

import pandas as pd
import pytest

import team_wtm.evaluation.comparison as mc


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
