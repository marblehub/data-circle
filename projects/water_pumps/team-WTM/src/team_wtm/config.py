"""Read the settings shared by the Sprint 2 commands and Python modules."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "project.toml"


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
    model_file: Path
    predictions_file: Path
    random_seed: int
    folds: int
    candidates: tuple[str, ...]
    logistic_max_iter: int
    forest_trees: int
    forest_jobs: int


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
        return Settings(
            project_root=root,
            raw_dir=raw,
            training_values=raw / _text(paths, "training_values"),
            training_labels=raw / _text(paths, "training_labels"),
            test_records=_names(paths, "test_records"),
            submission_format=raw / _text(paths, "submission_format"),
            split_plan=root / _text(paths, "split_plan"),
            report_dir=root / _text(paths, "report_dir"),
            model_file=root / _text(paths, "model_file"),
            predictions_file=root / _text(paths, "predictions_file"),
            random_seed=seed,
            folds=_integer(evaluation, "folds", 2),
            candidates=_names(comparison, "candidates"),
            logistic_max_iter=_integer(comparison, "logistic_max_iter", 1),
            forest_trees=_integer(comparison, "forest_trees", 1),
            forest_jobs=jobs,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid settings in {path}: {error}") from error


SETTINGS = load_settings()
