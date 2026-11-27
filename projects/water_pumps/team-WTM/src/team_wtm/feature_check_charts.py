"""Aggregate charts for the Issue #16 checks (reads only saved summary tables).

Run with:
    uv run python -m team_wtm.feature_checks charts
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from team_wtm.feature_checks import FIGURE_DIR, MIN_REPAIRS, REPORT_DIR

# Reference palette (validated slots 1 and 2), recessive ink and grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

OPTION_LABELS = {
    "single_waterpoint_type": "One field: waterpoint type",
    "single_quantity": "Quantity only (quantity and quantity_group)",
    "base": "base (13 columns)",
    "base_age": "+ age",
    "base_age_missing": "+ missing-value flags",
    "full_no_quantity": "full without quantity",
    "full": "full (22 columns, final model)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _feature_table():
    """Macro F1 per option for the final model, from all saved comparisons."""
    frames = [
        pd.read_csv(REPORT_DIR / f"{name}_comparison_table.csv")
        for name in ("pm3_h6", "pm3_ladder", "quantity")
    ]
    table = pd.concat(frames)
    table = table[table["model"].eq("random_forest_balanced")]
    table = table.drop_duplicates("features").set_index("features")
    order = [o for o in OPTION_LABELS if o in table.index]
    return table.loc[order]


def chart_feature_options(path):
    table = _feature_table()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    y = range(len(table))
    ax.barh(
        y,
        table["macro_f1_mean"],
        xerr=table["macro_f1_std"],
        height=0.55,
        color=BLUE,
        error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3},
    )
    for i, (mean, std) in enumerate(zip(table["macro_f1_mean"], table["macro_f1_std"])):
        ax.text(
            mean + std + 0.008, i, f"{mean:.3f}", va="center", color=INK, fontsize=9
        )
    ax.axvline(0.235, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.24, -0.75, "majority baseline 0.235", color=MUTED, fontsize=8)
    ax.set_yticks(list(y), [OPTION_LABELS[o] for o in table.index], color=INK)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Macro F1 (mean ± std over 5 shared folds)", color=MUTED)
    ax.set_title(
        "Feature options, Random Forest (balanced), same folds",
        loc="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_region_transfer(path):
    data = pd.read_csv(REPORT_DIR / "pm3_region_transfer_by_region.csv")
    data = data.sort_values("repair_recall_main")
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    _style(ax)
    y = range(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        ax.plot(
            [row["repair_recall_unseen"], row["repair_recall_main"]],
            [i, i],
            color=GRID,
            linewidth=2,
            zorder=1,
        )
    ax.scatter(
        data["repair_recall_main"],
        y,
        s=60,
        color=BLUE,
        zorder=2,
        edgecolors=SURFACE,
        linewidths=2,
        label="Main folds (region seen in fitting)",
    )
    ax.scatter(
        data["repair_recall_unseen"],
        y,
        s=60,
        color=ORANGE,
        zorder=2,
        marker="s",
        edgecolors=SURFACE,
        linewidths=2,
        label="Region folds (region unseen)",
    )
    labels = [
        f"{r}  (n repair = {n}{', too small' if s else ''})"
        for r, n, s in zip(data["region"], data["repair_records"], data["too_small"])
    ]
    ax.set_yticks(list(y), labels, color=INK, fontsize=8.5)
    ax.set_xlim(-0.02, 0.8)
    ax.set_xlabel(
        "Repair recall (share of real 'needs repair' pumps found)", color=MUTED
    )
    ax.set_title(
        "Repair recall by region: seen vs unseen region",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(
        0.01,
        0.01,
        f"Regions with fewer than {MIN_REPAIRS} real repairs are marked "
        "'too small'. Development records only.",
        color=MUTED,
        fontsize=8,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_charts():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart_feature_options(FIGURE_DIR / "pm3_feature_options.png")
    chart_region_transfer(FIGURE_DIR / "pm3_region_transfer.png")
    print(f"Saved charts to {FIGURE_DIR}")
