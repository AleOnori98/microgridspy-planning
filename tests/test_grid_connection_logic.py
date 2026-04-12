from __future__ import annotations

import numpy as np
import pytest

from core.io.utils import simulate_grid_availability_dynamic
from core.multi_year_model.data import (
    InputValidationError,
    _first_connection_year_to_ordinal,
)


def test_first_connection_year_maps_numeric_calendar_labels() -> None:
    ordinal = _first_connection_year_to_ordinal(2027, [2025, 2026, 2027])
    assert ordinal == 3


def test_first_connection_year_maps_exact_non_numeric_label() -> None:
    ordinal = _first_connection_year_to_ordinal("Y2", ["Y1", "Y2", "Y3"])
    assert ordinal == 2


def test_first_connection_year_rejects_ambiguous_numeric_value_for_non_numeric_labels() -> None:
    with pytest.raises(InputValidationError, match="does not match any sets.year label"):
        _first_connection_year_to_ordinal(2, ["Y1", "Y2", "Y3"])


def test_first_connection_year_rejects_missing_calendar_year_inside_horizon() -> None:
    with pytest.raises(InputValidationError, match="does not match any modeled calendar year"):
        _first_connection_year_to_ordinal(2026, [2025, 2027, 2028])


def test_dynamic_grid_availability_zeroes_pre_connection_years() -> None:
    availability = simulate_grid_availability_dynamic(
        avg_outages_per_year=0.0,
        avg_outage_duration_min=0.0,
        years=4,
        periods_per_year=5,
        first_year_connection=3,
        rng=np.random.default_rng(0),
    )

    assert availability.shape == (5, 4)
    assert np.all(availability[:, :2] == 0.0)
    assert np.all(availability[:, 2:] == 1.0)
