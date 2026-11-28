"""Construct fresh candidate estimators from shared configuration."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from team_wtm.config import SETTINGS

RANDOM_SEED = SETTINGS.random_seed

MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=SETTINGS.logistic_max_iter, random_state=RANDOM_SEED
    ),
    "logistic_regression_balanced": lambda: LogisticRegression(
        max_iter=SETTINGS.logistic_max_iter,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=SETTINGS.forest_trees,
        random_state=RANDOM_SEED,
        n_jobs=SETTINGS.forest_jobs,
    ),
    "random_forest_balanced": lambda: RandomForestClassifier(
        n_estimators=SETTINGS.forest_trees,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=SETTINGS.forest_jobs,
    ),
}
