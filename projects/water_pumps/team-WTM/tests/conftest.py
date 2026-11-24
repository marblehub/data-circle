"""Shared made-up data for the Issue #17 model tests."""

import numpy as np
import pandas as pd
import pytest

from team_wtm.evaluation import TARGET_CLASSES


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
