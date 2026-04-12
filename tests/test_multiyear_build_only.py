from __future__ import annotations

import numpy as np

import pytest

from core.multi_year_model.constraints import InputValidationError
from core.multi_year_model.model import MultiYearModel


def test_multiyear_build_only_smoke() -> None:
    m = MultiYearModel("test_multiyear")

    m._initialize_sets()
    m._initialize_data()
    m._initialize_vars()
    m._initialize_constraints()

    assert {"period", "year", "scenario", "inv_step"}.issubset(set(m.data.dims))
    assert isinstance(m.vars, dict)
    assert len(m.vars) > 0
    assert m.model is not None
    assert hasattr(m.model, "constraints")
    assert len(m.model.constraints) > 0


def test_multiyear_zero_land_limit_adds_constraint() -> None:
    m_nan = MultiYearModel("test_multiyear")
    m_nan._initialize_sets()
    m_nan._initialize_data()
    m_nan.data["land_availability_m2"] = m_nan.data["land_availability_m2"] * np.nan
    m_nan._initialize_vars()
    m_nan._initialize_constraints()
    constraints_without_land = len(m_nan.model.constraints)

    m_zero = MultiYearModel("test_multiyear")
    m_zero._initialize_sets()
    m_zero._initialize_data()
    m_zero.data["land_availability_m2"] = m_zero.data["land_availability_m2"] * 0.0
    m_zero._initialize_vars()
    m_zero._initialize_constraints()
    constraints_with_zero_land = len(m_zero.model.constraints)

    assert constraints_with_zero_land == constraints_without_land + 1


def test_multiyear_negative_land_limit_is_rejected() -> None:
    m = MultiYearModel("test_multiyear")
    m._initialize_sets()
    m._initialize_data()
    m.data["land_availability_m2"] = m.data["land_availability_m2"] * 0.0 - 1.0
    m._initialize_vars()

    with pytest.raises(InputValidationError, match="land_availability_m2 must be >= 0"):
        m._initialize_constraints()

