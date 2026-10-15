"""Numeric EDA, missingness checks, and stratified age comparisons.

Run python -m team_wtm.additional_eda in the installed project environment.
Only aggregate summaries and figures are written; source records stay local.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from team_wtm.cleaning import (
    DEFAULT_LABELS_PATH,
    DEFAULT_VALUES_PATH,
    PROJECT_DIR,
    clean_data,
    load_data,
    merge_data,
)

STATUS = ("functional", "functional needs repair", "non functional")
AGE_GROUPS = ("0-5", "6-10", "11-20", "21-30", "31+")
NUMERIC = ("pump_age", "population", "amount_tsh", "gps_height")
MISSING_FEATURES = (
    "construction_year",
    "population",
    "scheme_name",
    "scheme_management",
    "installer",
    "funder",
    "public_meeting",
    "permit",
)
LABELS = {
    "pump_age": "Age at recording (years)",
    "population": "Population",
    "amount_tsh": "amount_tsh (source units)",
    "gps_height": "Elevation (m)",
    "construction_year": "Construction year",
    "scheme_name": "Scheme name",
    "scheme_management": "Scheme management",
    "installer": "Installer",
    "funder": "Funder",
    "public_meeting": "Public meeting",
    "permit": "Permit",
}
COLORS = ("#008F78", "#E69F00", "#C6283E")
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "additional_eda"


def prepare_eda(raw: pd.DataFrame) -> pd.DataFrame:
    """Reuse shared cleaning, then derive age without changing source records.

    Nonpositive/unknown construction years, missing dates, and negative ages
    cannot supply usable ages. Elevation and amount zeros remain unchanged.
    """
    required = (
        set(NUMERIC[1:])
        | set(MISSING_FEATURES)
        | {
            "date_recorded",
            "status_group",
            "region",
            "extraction_type_class",
        }
    )
    absent = required - set(raw.columns)
    if absent:
        raise ValueError(f"Missing analysis columns: {sorted(absent)}")
    if raw["status_group"].isna().any() or not raw["status_group"].isin(STATUS).all():
        raise ValueError("All analysis records need one of the three target labels.")
    data = clean_data(raw)
    for column in ("population", "amount_tsh", "gps_height", "construction_year"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    age = data["date_recorded"].dt.year - data["construction_year"]
    data["pump_age"] = age.where(data["construction_year"].gt(0) & age.ge(0))
    data["age_group"] = pd.cut(
        data["pump_age"],
        bins=[-1, 5, 10, 20, 30, np.inf],
        labels=AGE_GROUPS,
    )
    data["age_availability"] = np.select(
        [data["construction_year"].isna(), data["pump_age"].isna()],
        ["Unknown construction year", "Invalid year/date/age"],
        default="Usable age",
    )
    return data


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Report class denominators, valid counts, and the complete numeric range."""
    rows = []
    for feature in NUMERIC:
        for status in STATUS:
            group = data.loc[data["status_group"].eq(status), feature]
            valid = group.dropna()
            quantiles = valid.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            rows.append(
                {
                    "feature": feature,
                    "status": status,
                    "class_n": len(group),
                    "valid_n": len(valid),
                    "missing_n": int(group.isna().sum()),
                    "zero_n": int(valid.eq(0).sum()),
                    "negative_n": int(valid.lt(0).sum()),
                    "min": valid.min(),
                    "p01": quantiles.loc[0.01],
                    "p25": quantiles.loc[0.25],
                    "median": quantiles.loc[0.5],
                    "p75": quantiles.loc[0.75],
                    "p99": quantiles.loc[0.99],
                    "max": valid.max(),
                }
            )
    return pd.DataFrame(rows)


def numeric_correlations(data: pd.DataFrame, *, mask_elevation_zero=False):
    """Return Spearman/Pearson coefficients and their pairwise valid counts.

    A fixed allowlist excludes identifiers, category codes, and the target.
    The optional elevation mask is a sensitivity analysis, not a cleaning rule.
    """
    numeric = data.loc[:, list(NUMERIC)].copy()
    if mask_elevation_zero:
        numeric["gps_height"] = numeric["gps_height"].mask(numeric["gps_height"].eq(0))
    valid = numeric.notna().astype("int64")
    return {
        "spearman": numeric.corr(method="spearman", min_periods=3),
        "pearson": numeric.corr(method="pearson", min_periods=3),
        "pairwise_n": valid.T @ valid,
    }


def missingness_by(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Conditional missing rates use each class/region's full denominator."""
    rows = []
    for value, subset in data.groupby(group, dropna=False, observed=True, sort=True):
        for feature in MISSING_FEATURES:
            missing = int(subset[feature].isna().sum())
            rows.append(
                {
                    "group": value,
                    "feature": feature,
                    "n": len(subset),
                    "missing_n": missing,
                    "missing_pct": 100 * missing / len(subset),
                }
            )
    return pd.DataFrame(rows)


def age_availability_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Show how target composition differs between included and excluded ages."""
    rows = []
    for availability, group in data.groupby("age_availability", sort=True):
        for status in STATUS:
            count = int(group["status_group"].eq(status).sum())
            rows.append(
                {
                    "availability": availability,
                    "status": status,
                    "n": len(group),
                    "status_n": count,
                    "status_pct": 100 * count / len(group),
                }
            )
    return pd.DataFrame(rows)


def age_rates(data: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Return all observed stratum/age cells, including small-cell counts."""
    usable = data.loc[data["age_group"].notna()].copy()
    usable[stratum] = usable[stratum].fillna("<missing>")
    full_index = pd.MultiIndex.from_product(
        [sorted(data[stratum].fillna("<missing>").unique()), AGE_GROUPS],
        names=[stratum, "age_group"],
    )
    counts = (
        usable.groupby([stratum, "age_group", "status_group"], observed=True)
        .size()
        .unstack("status_group", fill_value=0)
        .reindex(index=full_index, columns=STATUS, fill_value=0)
    )
    result = counts.reset_index()
    result["n"] = result[list(STATUS)].sum(axis=1)
    for status in STATUS:
        result[status + "_pct"] = 100 * result[status] / result["n"]
    result["age_group"] = result["age_group"].astype(str)
    return result


def age_contrasts(rates: pd.DataFrame, stratum: str, min_n: int = 30):
    """Compare youngest with oldest only when both denominators meet min_n."""
    if min_n < 1:
        raise ValueError("min_n must be positive.")
    rows = []
    for value, group in rates.groupby(stratum, sort=True):
        cells = group.set_index("age_group")
        row = {stratum: value}
        for label, key in (("young", AGE_GROUPS[0]), ("old", AGE_GROUPS[-1])):
            row[label + "_n"] = int(cells.loc[key, "n"]) if key in cells.index else 0
            row[label + "_functional"] = (
                int(cells.loc[key, "functional"]) if key in cells.index else 0
            )
            row[label + "_pct"] = (
                float(cells.loc[key, "functional_pct"])
                if key in cells.index
                else np.nan
            )
        row["eligible"] = row["young_n"] >= min_n and row["old_n"] >= min_n
        row["young_minus_old_pp"] = row["young_pct"] - row["old_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def standardized_extremes(contrasts: pd.DataFrame) -> dict:
    """Use one pooled extreme-age stratum mix for both age groups.

    Crude and standardized results use the SAME eligible strata. This is
    descriptive one-factor standardization, not a causal adjusted estimate.
    """
    eligible = contrasts.loc[contrasts["eligible"]].copy()
    if eligible.empty:
        return {"eligible_strata": 0, "common_n": 0}
    pooled = eligible["young_n"] + eligible["old_n"]
    weights = pooled / pooled.sum()
    result = {"eligible_strata": len(eligible), "common_n": int(pooled.sum())}
    for label in ("young", "old"):
        result[label + "_n"] = int(eligible[label + "_n"].sum())
        result[label + "_pooled_pct"] = (
            100 * eligible[label + "_functional"].sum() / result[label + "_n"]
        )
        result[label + "_standardized_pct"] = float(
            (weights * eligible[label + "_pct"]).sum()
        )
    result["pooled_gap_pp"] = result["young_pooled_pct"] - result["old_pooled_pct"]
    result["standardized_gap_pp"] = (
        result["young_standardized_pct"] - result["old_standardized_pct"]
    )
    return result


def _save(fig, output: Path, name: str):
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_numeric(data, output):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig2, boxes = plt.subplots(2, 2, figsize=(13, 9))
    for feature, ax, box in zip(NUMERIC, axes.flat, boxes.flat, strict=True):
        values = [
            data.loc[data["status_group"].eq(s), feature].dropna() for s in STATUS
        ]
        transformed = feature in ("population", "amount_tsh")
        all_values = data[feature].dropna()
        all_plot = (
            np.sign(all_values) * np.log1p(np.abs(all_values))
            if transformed
            else all_values
        )
        edges = np.histogram_bin_edges(all_plot, bins=35)
        for status, color, series in zip(STATUS, COLORS, values, strict=True):
            plotted = (
                np.sign(series) * np.log1p(np.abs(series)) if transformed else series
            )
            if len(plotted):
                ax.hist(
                    plotted,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    label=status,
                    color=color,
                )
        ax.set_title(LABELS[feature], loc="left", fontweight="bold")
        ax.set_xlabel(
            "Signed log1p scale (all observed values)"
            if transformed
            else LABELS[feature]
        )
        ax.set_ylabel("Within-class density")
        result = box.boxplot(
            values, tick_labels=STATUS, showfliers=False, patch_artist=True
        )
        for patch, color in zip(result["boxes"], COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        if transformed:
            box.set_yscale("symlog", linthresh=1)
        box.set_title(LABELS[feature], loc="left", fontweight="bold")
        box.tick_params(axis="x", labelrotation=15, labelsize=9)
        box.set_ylabel(
            "Original units (symlog axis)" if transformed else LABELS[feature]
        )
    axes.flat[0].legend(fontsize=9)
    fig.suptitle(
        "Numeric distributions by recorded status", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    fig2.suptitle(
        "Numeric distributions by class: medians and IQRs\n"
        "Whiskers: 1.5 x IQR; individual outlier markers omitted, no records deleted",
        fontsize=14,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, output, "numeric_distributions.png")
    _save(fig2, output, "numeric_boxplots.png")


def plot_correlations(base, sensitivity, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    names = ["Age", "Population", "amount_tsh", "Elevation"]
    for result, ax, title in zip(
        (base, sensitivity),
        axes,
        ("Elevation zeros retained", "Sensitivity: elevation zeros excluded"),
        strict=True,
    ):
        corr = result["spearman"]
        annotations = np.array(
            [
                [
                    f"{corr.iloc[i, j]:.2f}\nn={result['pairwise_n'].iloc[i, j]:,}"
                    for j in range(len(NUMERIC))
                ]
                for i in range(len(NUMERIC))
            ]
        )
        sns.heatmap(
            corr,
            annot=annotations,
            fmt="",
            cmap="vlag",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            xticklabels=names,
            yticklabels=names,
            annot_kws={"fontsize": 9},
            cbar=ax is axes[-1],
        )
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Spearman correlation and pairwise sample sizes", fontsize=15)
    fig.tight_layout()
    _save(fig, output, "numeric_spearman_heatmaps.png")


def plot_missingness(by_status, by_region, output):
    for table, name, title, features, size in (
        (
            by_status,
            "missingness_by_status.png",
            "Missing or unknown values within each class",
            MISSING_FEATURES,
            (12, 6),
        ),
        (
            by_region,
            "missingness_by_region.png",
            "Missing or unknown values within each region",
            MISSING_FEATURES[:3],
            (9, 12),
        ),
    ):
        pivot = table.pivot(index="group", columns="feature", values="missing_pct")
        pivot = pivot.reindex(columns=list(features))
        if name == "missingness_by_status.png":
            pivot = pivot.reindex(STATUS).T
        else:
            pivot = pivot.sort_values("construction_year", ascending=False)
        fig, ax = plt.subplots(figsize=size)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrBr",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "% missing/unknown"},
        )
        if name == "missingness_by_status.png":
            ax.set_yticklabels([LABELS[x] for x in pivot.index], rotation=0)
        else:
            ax.set_xticklabels([LABELS[x] for x in pivot.columns], rotation=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            title + "\nEach percentage uses that class/region's full count", pad=16
        )
        fig.tight_layout()
        _save(fig, output, name)


def plot_age_heatmap(rates, stratum, output, min_n):
    values = rates.pivot(index=stratum, columns="age_group", values="functional_pct")
    counts = rates.pivot(index=stratum, columns="age_group", values="n")
    values = values.reindex(columns=AGE_GROUPS).sort_index()
    counts = counts.reindex(index=values.index, columns=AGE_GROUPS).fillna(0)
    annotations = np.array(
        [
            [
                f"{values.iloc[i, j]:.1f}%\nn={int(counts.iloc[i, j]):,}"
                if counts.iloc[i, j] >= min_n
                else ""
                for j in range(len(AGE_GROUPS))
            ]
            for i in range(len(values))
        ]
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(values) * 0.58 + 2)))
    ax.set_facecolor("#E5E5E5")
    sns.heatmap(
        values,
        mask=counts.lt(min_n),
        annot=annotations,
        fmt="",
        cmap="YlGn",
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "% functional"},
    )
    for i in range(len(values)):
        for j in range(len(AGE_GROUPS)):
            if counts.iloc[i, j] < min_n:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"n={int(counts.iloc[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#555555",
                )
    ax.set_title(
        f"Functional share within {stratum.replace('_', ' ')} and age group\n"
        f"Grey cells: fewer than {min_n} records; rates not displayed",
        pad=16,
    )
    ax.set_xlabel("Age at recording (years)")
    ax.set_ylabel(stratum.replace("_", " ").title())
    fig.tight_layout()
    _save(fig, output, f"age_by_{stratum}.png")


def _records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=10))


def run_analysis(
    values_path=DEFAULT_VALUES_PATH,
    labels_path=DEFAULT_LABELS_PATH,
    output_dir=DEFAULT_OUTPUT,
    min_n=30,
):
    """Regenerate aggregate outputs without saving individual observations."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if min_n < 1:
        raise ValueError("min_n must be positive.")
    values, labels = load_data(values_path, labels_path)
    raw = merge_data(values, labels)
    data = prepare_eda(raw)
    base = numeric_correlations(data)
    sensitivity = numeric_correlations(data, mask_elevation_zero=True)
    by_status = missingness_by(data, "status_group")
    by_region = missingness_by(data, "region")
    result = {
        "method": {
            "min_cell_n": min_n,
            "age_groups": AGE_GROUPS,
            "numeric_features": NUMERIC,
            "target_order": STATUS,
        },
        "inputs": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (values_path, labels_path)
        },
        "counts": {
            "all_rows": len(data),
            "usable_age": int(data["pump_age"].notna().sum()),
            "unknown_year": int(data["construction_year"].isna().sum()),
            "invalid_known_age": int(
                data["age_availability"].eq("Invalid year/date/age").sum()
            ),
            "elevation_zero": int(data["gps_height"].eq(0).sum()),
        },
        "numeric_summary": _records(numeric_summary(data)),
        "correlations": {
            "features": NUMERIC,
            **{
                label: {k: v.to_numpy().tolist() for k, v in item.items()}
                for label, item in (
                    ("base", base),
                    ("elevation_sensitivity", sensitivity),
                )
            },
        },
        "missingness_by_status": _records(by_status),
        "missingness_by_region": _records(by_region),
        "age_availability": _records(age_availability_summary(data)),
    }
    for stratum in ("region", "extraction_type_class"):
        rates = age_rates(data, stratum)
        contrasts = age_contrasts(rates, stratum, min_n)
        result[stratum] = {
            "age_rates": _records(rates),
            "contrasts": _records(contrasts),
            "standardized": standardized_extremes(contrasts),
        }
        plot_age_heatmap(rates, stratum, output, min_n)
    plot_numeric(data, output)
    plot_correlations(base, sensitivity, output)
    plot_missingness(by_status, by_region, output)
    serializable = json.loads(pd.Series(result).to_json(default_handler=str))
    (output / "summary.json").write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-cell-n", type=int, default=30)
    args = parser.parse_args()
    result = run_analysis(args.values, args.labels, args.output_dir, args.min_cell_n)
    print(json.dumps({"output": str(args.output_dir), **result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
