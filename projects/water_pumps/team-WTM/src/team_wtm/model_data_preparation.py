"""Compatibility imports for shared record preparation and model preprocessing."""

from team_wtm.data.preparation import (
    ID_COLUMN,
    TARGET_COLUMN,
    load_training_data,
    merge_training_data,
    prepare_features,
    prepare_model_inputs,
)
from team_wtm.modeling.preprocessing import (
    _categorical_values_as_text,  # noqa: F401
    build_pipeline,
    build_preprocessor,
    categories_as_text,
)

__all__ = [
    "ID_COLUMN",
    "TARGET_COLUMN",
    "build_pipeline",
    "build_preprocessor",
    "categories_as_text",
    "load_training_data",
    "merge_training_data",
    "prepare_features",
    "prepare_model_inputs",
]
