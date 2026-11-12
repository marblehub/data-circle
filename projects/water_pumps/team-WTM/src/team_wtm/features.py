"""Feature options for Sprint 2 comparisons (Issue #16).

Reuses the shared Sprint 1 rules (pump age and quality flags) from
export_cleaned.add_quality_features. Nothing here re-implements them.
"""

import pandas as pd

from team_wtm.export_cleaned import add_quality_features

# Groups of related columns. Each group is switched on or off as one unit.
FEATURE_GROUPS = {
    "technical": ["extraction_type_class", "waterpoint_type_group", "source_type"],
    "management": ["management_group", "payment_type", "scheme_management",
                   "permit", "public_meeting"],
    "water_quality": ["water_quality"],
    "numeric": ["population", "gps_height"],
    # D09: quantity == quantity_group, so they are always kept or removed together.
    "quantity": ["quantity", "quantity_group"],
    "age": ["pump_age", "pump_age_missing"],
    "missing": ["population_missing", "gps_height_zero", "coordinate_unusable"],
    "location": ["region", "basin", "longitude_screened", "latitude_screened"],
}


def _combine(*groups: str) -> list[str]:
    return [column for group in groups for column in FEATURE_GROUPS[group]]


BASE_GROUPS = ("technical", "management", "water_quality", "numeric", "quantity")

FEATURE_OPTIONS = {
    # Main ladder: add one group at a time.
    "base": _combine(*BASE_GROUPS),
    "base_age": _combine(*BASE_GROUPS, "age"),
    "base_age_missing": _combine(*BASE_GROUPS, "age", "missing"),
    "full": _combine(*BASE_GROUPS, "age", "missing", "location"),
    # Quantity test (D26): same as full, without the quantity group.
    "full_no_quantity": [c for c in _combine(*BASE_GROUPS, "age", "missing", "location")
                         if c not in FEATURE_GROUPS["quantity"]],
    # H6 single-field setups (strongest Cramer's V fields in Issue #5).
    "single_quantity": _combine("quantity"),
    "single_waterpoint_type": ["waterpoint_type_group"],
}


def add_feature_columns(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add shared quality flags plus the extra Issue #16 columns.

    Rows are never removed. Unusable coordinates become missing only in the
    *_screened columns; the original longitude/latitude stay unchanged.
    """
    output = add_quality_features(cleaned)
    height = pd.to_numeric(output["gps_height"], errors="coerce")
    output["gps_height_zero"] = height.eq(0)
    unusable = output["coordinate_placeholder"] | output["coordinate_screen_failed"]
    output["coordinate_unusable"] = unusable
    output["longitude_screened"] = output["longitude"].where(~unusable)
    output["latitude_screened"] = output["latitude"].where(~unusable)
    return output


def select_features(data: pd.DataFrame, option: str) -> pd.DataFrame:
    """Return only the columns of one named feature option."""
    if option not in FEATURE_OPTIONS:
        raise KeyError(f"Unknown feature option: {option}")
    columns = FEATURE_OPTIONS[option]
    absent = [column for column in columns if column not in data.columns]
    if absent:
        raise KeyError(f"Columns missing for '{option}': {absent}")
    return data[columns]