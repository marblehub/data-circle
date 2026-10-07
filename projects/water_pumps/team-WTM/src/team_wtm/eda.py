"""Helpers for target-class and categorical EDA (Issue #5, PM-3).

These functions only read and describe the data. They do not change raw files.
Data cleaning (Issue #4) and geospatial / engineered features (Issue #6) live elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../team-WTM
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# File names as downloaded from DrivenData
VALUES_FILE = "Training_set_values.csv"
LABELS_FILE = "Training_set_labels.csv"

# --------------------------------------------------------------------------
# Status colours: fixed for every chart, so red always means "non functional"
# --------------------------------------------------------------------------
STATUS_ORDER = ["functional", "functional needs repair", "non functional"]
STATUS_COLORS = {
    "functional": "#0ca30c",
    "functional needs repair": "#fab219",
    "non functional": "#d03b3b",
}


def load_data(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Read values + labels, join on `id`, add 0/1 flags for each status."""
    raw_dir = Path(raw_dir)
    values = pd.read_csv(raw_dir / VALUES_FILE)
    labels = pd.read_csv(raw_dir / LABELS_FILE)
    df = values.merge(labels, on="id", how="inner", validate="one_to_one")
    df["is_functional"] = (df["status_group"] == "functional").astype(int)
    df["needs_repair"] = (df["status_group"] == "functional needs repair").astype(int)
    df["non_functional"] = (df["status_group"] == "non functional").astype(int)
    return df


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def status_share(df: pd.DataFrame, col: str, min_count: int = 0) -> pd.DataFrame:
    """Share of each status inside each category of `col` (rows sum to 1), plus `n`."""
    s = df[col].astype("object").fillna("missing")
    tab = pd.crosstab(s, df["status_group"], normalize="index")[STATUS_ORDER]
    tab["n"] = s.value_counts()
    return tab[tab["n"] >= min_count].sort_values("non functional")


def clean_text(series: pd.Series) -> pd.Series:
    """Lower-case and strip text; treat '0', '-' and '' as missing."""
    s = series.astype("object").fillna("missing").astype(str).str.strip().str.lower()
    return s.replace({"0": "missing", "-": "missing", "": "missing", "nan": "missing"})


def lump_rare(series: pd.Series, top: int = 15, other: str = "other") -> pd.Series:
    """Keep the `top` most common values (after clean_text), put the rest into 'other'."""
    s = clean_text(series)
    keep = s.value_counts().head(top).index
    return s.where(s.isin(keep), other)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V between two categorical series (0 = no association, 1 = perfect)."""
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    r, k = t.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_status_bars(tab: pd.DataFrame, title: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal 100% stacked bars, one per category, with n on the right."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(tab) + 1.4))
    left = np.zeros(len(tab))
    labels = [str(i) for i in tab.index]
    for status in STATUS_ORDER:
        vals = tab[status].to_numpy()
        ax.barh(labels, vals, left=left, color=STATUS_COLORS[status], label=status,
                edgecolor="white", linewidth=1.5, height=0.7)
        left += vals
    for y, n in enumerate(tab["n"]):
        ax.text(1.01, y, f"n={n:,}", va="center", fontsize=8, color="#555555")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of water points")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=22)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8,
              borderaxespad=0.2, handlelength=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return ax
