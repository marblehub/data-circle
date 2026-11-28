"""Read the settings shared by the Sprint 2 commands and Python modules."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR.parents[1] / "configs" / "project.toml"


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_dir: Path
    training_values: Path
    training_labels: Path
    test_records: tuple[str, ...]
    submission_format: Path
    split_plan: Path
    report_dir: Path
    runs_dir: Path
    model_file: Path
    predictions_file: Path
    random_seed: int
    folds: int
    candidates: tuple[str, ...]
    logistic_max_iter: int
    forest_trees: int
    forest_jobs: int
    workflow_runs_dir: Path
    processed_dir: Path
    figures_dir: Path
    cleaned_snapshot: Path
    out_of_fold_predictions: Path
    region_predictions: Path
    eda_report_dir: Path
    min_repairs: int
    region_folds: int
    final_model: str
    final_features: str
    minimum_category_frequency: float
    log_level: str
    log_file: Path
    log_max_bytes: int
    log_backup_count: int
    eda_min_cell_records: int


def _text(table, key):
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _names(table, key):
    values = table[key]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a nonempty list")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{key} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{key} contains duplicate names")
    return tuple(values)


def _integer(table, key, minimum=None):
    value = table[key]
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid integer for {key}: {value!r}")
    return value


def load_settings(path=None):
    """Load one complete TOML file. TEAM_WTM_CONFIG can select another file."""
    path = path or os.environ.get("TEAM_WTM_CONFIG")
    if path is None:
        path = DEFAULT_CONFIG
        if not path.is_file():
            # A wheel installation can run against a separate project checkout.
            cwd = Path.cwd()
            for folder in (cwd, *cwd.parents):
                candidate = folder / "configs" / "project.toml"
                if candidate.is_file():
                    path = candidate
                    break
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Project config not found at {path}. Set TEAM_WTM_CONFIG to your project.toml."
        )
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    try:
        paths = data["paths"]
        evaluation = data["evaluation"]
        comparison = data["comparison"]
        root = (path.parent / _text(paths, "project_root")).resolve()
        raw = root / _text(paths, "raw_dir")
        seed = _integer(evaluation, "random_seed", 0)
        if seed >= 2**32:
            raise ValueError("random_seed must be smaller than 2**32")
        jobs = _integer(comparison, "forest_jobs")
        if jobs == 0:
            raise ValueError("forest_jobs must be nonzero")
        checks = {
            "min_repairs": 30,
            "region_folds": 5,
            **data.get("feature_checks", {}),
        }
        final = {
            "model": "random_forest_balanced",
            "features": "full",
            **data.get("final_model", {}),
        }
        frequency = data.get("preprocessing", {}).get(
            "minimum_category_frequency", 0.01
        )
        if type(frequency) not in (int, float) or not 0 < frequency <= 1:
            raise ValueError("minimum_category_frequency must be between 0 and 1")
        level = data.get("logging", {}).get("level", "INFO")
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                "logging level must be DEBUG, INFO, WARNING, ERROR or CRITICAL"
            )
        processed = root / _text(
            {"processed_dir": "data/processed", **paths}, "processed_dir"
        )
        figures = root / _text(
            {"figures_dir": "reports/figures", **paths}, "figures_dir"
        )
        return Settings(
            project_root=root,
            raw_dir=raw,
            training_values=raw / _text(paths, "training_values"),
            training_labels=raw / _text(paths, "training_labels"),
            test_records=_names(paths, "test_records"),
            submission_format=raw / _text(paths, "submission_format"),
            split_plan=root / _text(paths, "split_plan"),
            report_dir=root / _text(paths, "report_dir"),
            runs_dir=root / _text({"runs_dir": "artifacts/runs", **paths}, "runs_dir"),
            model_file=root / _text(paths, "model_file"),
            predictions_file=root / _text(paths, "predictions_file"),
            random_seed=seed,
            folds=_integer(evaluation, "folds", 2),
            candidates=_names(comparison, "candidates"),
            logistic_max_iter=_integer(comparison, "logistic_max_iter", 1),
            forest_trees=_integer(comparison, "forest_trees", 1),
            forest_jobs=jobs,
            workflow_runs_dir=root
            / _text(
                {"workflow_runs_dir": "artifacts/workflows", **paths},
                "workflow_runs_dir",
            ),
            processed_dir=processed,
            figures_dir=figures,
            cleaned_snapshot=root
            / _text(
                {
                    "cleaned_snapshot": str(processed / "training_cleaned.parquet"),
                    **paths,
                },
                "cleaned_snapshot",
            ),
            out_of_fold_predictions=root
            / _text(
                {
                    "out_of_fold_predictions": str(
                        processed / "pm3_out_of_fold_predictions.csv"
                    ),
                    **paths,
                },
                "out_of_fold_predictions",
            ),
            region_predictions=root
            / _text(
                {
                    "region_predictions": str(
                        processed / "pm3_region_fold_predictions.csv"
                    ),
                    **paths,
                },
                "region_predictions",
            ),
            eda_report_dir=root
            / _text(
                {"eda_report_dir": "reports/sprint1/additional_eda", **paths},
                "eda_report_dir",
            ),
            min_repairs=_integer(checks, "min_repairs", 1),
            region_folds=_integer(checks, "region_folds", 2),
            final_model=_text(final, "model"),
            final_features=_text(final, "features"),
            minimum_category_frequency=float(frequency),
            log_level=level,
            log_file=root
            / _text({"log_file": "artifacts/logs/project.log", **paths}, "log_file"),
            log_max_bytes=_integer(
                {"max_bytes": 5000000, **data.get("logging", {})}, "max_bytes", 1
            ),
            log_backup_count=_integer(
                {"backup_count": 3, **data.get("logging", {})}, "backup_count", 1
            ),
            eda_min_cell_records=_integer(
                {"min_cell_records": 30, **data.get("eda", {})}, "min_cell_records", 1
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid settings in {path}: {error}") from error


SETTINGS = load_settings()
