"""Integration tests for EDA reports."""

import json

from team_wtm.additional_eda import (
    NUMERIC,
    run_analysis,
)


def test_analysis_writes_summary_and_figures_to_separate_directories(sample_raw, tmp_path):
    raw = sample_raw([0, 3, 8, 15, 25, 35] * 5)
    values_path = tmp_path / "values.csv"
    labels_path = tmp_path / "labels.csv"
    raw.drop(columns="status_group").to_csv(values_path, index=False)
    raw[["id", "status_group"]].to_csv(labels_path, index=False)
    output_dir = tmp_path / "reports" / "sprint1" / "additional_eda"
    figures_dir = tmp_path / "reports" / "figures" / "additional_eda"

    result = run_analysis(
        values_path, labels_path, output_dir, min_n=1, figures_dir=figures_dir
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["numeric_outliers"] == result["numeric_outliers"]
    assert len(summary["numeric_outliers"]) == len(NUMERIC)
    assert summary["counts"] == result["counts"]
    assert summary["counts"]["all_rows"] == len(raw)
    assert {p.name for p in output_dir.iterdir()} == {"summary.json"}
    assert {p.name for p in figures_dir.iterdir()} == {
        "numeric_distributions.png",
        "numeric_boxplots.png",
        "numeric_outlier_distributions.png",
        "numeric_spearman_heatmaps.png",
        "missingness_by_status.png",
        "missingness_by_region.png",
        "age_by_region.png",
        "age_by_extraction_type_class.png",
    }
    assert all(
        p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in figures_dir.iterdir()
    )
