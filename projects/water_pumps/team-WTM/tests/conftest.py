"""Shared made-up data for the Issue #17 model tests."""

import joblib
import numpy as np
import pandas as pd
import pytest

from team_wtm.evaluation.splits import TARGET_CLASSES


@pytest.fixture
def training():
    """120 made-up pumps with every column the 'base' and 'full' options need."""
    rng = np.random.default_rng(0)
    rows = []
    for number in range(120):
        label = TARGET_CLASSES[number % 3]
        rows.append(
            {
                "id": number,
                "extraction_type_class": rng.choice(["gravity", "handpump"]),
                "waterpoint_type_group": rng.choice(["communal standpipe", "hand pump"]),
                "source_type": rng.choice(["spring", "borehole"]),
                "management_group": "user-group",
                "payment_type": rng.choice(["never pay", "monthly"]),
                "scheme_management": rng.choice(["VWC", None]),
                "permit": rng.choice([True, False, None]),
                "public_meeting": rng.choice([True, False, None]),
                "water_quality": "soft",
                # The label is easy to learn from quantity, so models can do well.
                "quantity": {"functional": "enough",
                             "functional needs repair": "seasonal",
                             "non functional": "dry"}[label],
                "quantity_group": "enough",
                "population": int(rng.integers(0, 500)),
                "gps_height": number,  # unique, so every row is its own group
                "construction_year": int(rng.choice([0, 1995, 2005])),
                "date_recorded": "2013-02-01",
                "region": rng.choice(["Iringa", "Mara"]),
                "basin": rng.choice(["Pangani", "Rufiji"]),
                # One placeholder coordinate, so the location screen is exercised.
                "longitude": 0.0 if number == 5 else 35.0 + number / 1000,
                "latitude": -2e-8 if number == 5 else -6.0,
                "status_group": label,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def isolate_default_workflow_history(tmp_path, monkeypatch):
    """Keep default test runs in temporary experiment directories."""
    from dataclasses import replace

    from team_wtm import logging_config, runtime

    original = runtime.WorkflowRun
    default_directory = runtime.SETTINGS.workflow_runs_dir

    def create_run(name, config, on_progress):
        if config.workflow_runs_dir == default_directory:
            config = replace(config, workflow_runs_dir=tmp_path / "workflows")
        return original(name, config, on_progress)

    original_configure = logging_config.configure_logging
    default_log = runtime.SETTINGS.log_file

    def configure_test_logging(config=None, *, console=True):
        config = config or runtime.SETTINGS
        if config.log_file == default_log:
            config = replace(config, log_file=tmp_path / "project.log")
        return original_configure(config, console=console)

    monkeypatch.setattr(runtime, "WorkflowRun", create_run)
    monkeypatch.setattr(logging_config, "configure_logging", configure_test_logging)
    try:
        yield
    finally:
        logging_config.shutdown_logging()


@pytest.fixture
def evaluation_training():
    # Unique feature values give enough independent groups for each class.
    rows = []
    for class_number, label in enumerate(TARGET_CLASSES):
        for number in range(30):
            rows.append(
                {
                    "id": len(rows),
                    "feature": class_number * 100 + number,
                    "status_group": label,
                }
            )
    # Matching features with different labels must still stay together.
    rows.extend(
        [
            {"id": 90, "feature": 0, "status_group": "non functional"},
            {"id": 91, "feature": 0, "status_group": "functional"},
        ]
    )
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot_inputs(tmp_path):
    values = pd.DataFrame(
        {
            "id": [8, 2, 5, 1, 7, 9],
            "construction_year": [2000, 0, 2015, 2013, 2001, 2000],
            "date_recorded": ["2013-01-01"] * 4 + ["bad-date", "2013-01-01"],
            "population": [0, 5, 10, 1, 20, 0],
            "gps_height": [0, -3, 100, 50, 60, 0],
            "amount_tsh": [0, 0, 25, 0, 50, 0],
            "longitude": [0, 35, 35, 35, 35, 0],
            "latitude": [-2e-8, -6, -6, -0.95, None, -2e-8],
            "installer": ["0", "DWE", "DWE", "DWE", "unknown", "0"],
            "public_meeting": [True, False, None, True, False, True],
            "permit": [None, True, False, True, True, None],
            "num_private": [0] * 6,
        }
    )
    labels = pd.DataFrame(
        {
            "id": [1, 2, 5, 7, 8, 9],
            "status_group": [
                "functional",
                "functional needs repair",
                "non functional",
                "functional",
                "functional",
                "functional",
            ],
        }
    )
    values_path, labels_path = tmp_path / "values.csv", tmp_path / "labels.csv"
    values.to_csv(values_path, index=False)
    labels.to_csv(labels_path, index=False)
    return values_path, labels_path, tmp_path / "processed" / "cleaned.parquet"


@pytest.fixture
def sample_raw():
    """Return an EDA table factory with controlled synthetic pump ages."""
    from team_wtm.analysis.numeric import MISSING_FEATURES, STATUS

    def make_raw(ages):
        count = len(ages)
        data = pd.DataFrame(
            {
                "id": range(count),
                "construction_year": [2013 - age for age in ages],
                "date_recorded": ["2013-06-01"] * count,
                "population": np.arange(count) + 1,
                "amount_tsh": np.arange(count),
                "gps_height": np.arange(count) - 2,
                "status_group": [STATUS[i % 3] for i in range(count)],
                "region": ["A"] * count,
                "extraction_type_class": ["gravity"] * count,
            }
        )
        for column in MISSING_FEATURES:
            if column not in data:
                data[column] = "recorded"
        return data
    return make_raw


@pytest.fixture
def feature_data():
    """Return a preparation table factory with an optional target."""
    def make_feature_data(include_target=True):
        data = {
            "id": [11, 12, 13],
            "extraction_type_class": ["gravity", "handpump", "gravity"],
            "waterpoint_type_group": ["communal", "communal", "other"],
            "source_type": ["spring", "borehole", "spring"],
            "management_group": ["user-group", "commercial", "user-group"],
            "payment_type": ["never pay", "pay monthly", "never pay"],
            "scheme_management": ["VWC", "Water Board", "VWC"],
            "permit": [True, False, True],
            "public_meeting": [True, None, False],
            "water_quality": ["soft", "salty", "soft"],
            "population": [100, 0, 50],
            "gps_height": [120, None, 140],
            "construction_year": [2000, 0, 1990],
            "date_recorded": ["2010-01-01", "2012-01-01", "2010-01-01"],
            "longitude": [35.0, 0.0, 36.0],
            "latitude": [-6.0, 0.0, -7.0],
            "quantity": ["enough", "dry", "enough"],
            "quantity_group": ["enough", "dry", "enough"],
            "pump_age": [10, None, 20],
            "pump_age_missing": [False, True, False],
            "population_missing": [False, True, False],
            "gps_height_zero": [False, False, False],
            "coordinate_unusable": [False, False, False],
            "region": ["A", "B", "A"],
            "basin": ["X", "Y", "X"],
            "longitude_screened": [35.0, None, 36.0],
            "latitude_screened": [-6.0, None, -7.0],
        }
        if include_target:
            data["status_group"] = [
                "functional",
                "non functional",
                "functional needs repair",
            ]
        return pd.DataFrame(data)
    return make_feature_data

@pytest.fixture
def full_features(training):
    """Prepare the configured final-model columns from synthetic records."""
    from team_wtm.evaluation.comparison import add_model_columns
    from team_wtm.modeling.features import select_features
    from team_wtm.modeling.training import FINAL_FEATURES

    return select_features(add_model_columns(training), FINAL_FEATURES)


@pytest.fixture
def model(training, full_features):
    """Fit the configured final model on synthetic records for prediction checks."""
    from team_wtm.modeling.training import fit_final_pipeline

    return fit_final_pipeline(full_features, training["status_group"])


@pytest.fixture
def records(training):
    """Made-up records without the target, in a shuffled (unsorted) order."""
    shuffled = training.sample(frac=1, random_state=1)
    return shuffled.drop(columns="status_group")


@pytest.fixture
def model_file(model, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return path
