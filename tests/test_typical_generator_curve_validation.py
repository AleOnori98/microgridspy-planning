from __future__ import annotations

from pathlib import Path

import numpy as np

from core.data_pipeline.battery_loss_model import resolve_efficiency_curve_values
from core.data_pipeline.typical_year_parsing import _validate_generator_partial_load_curve


def test_generator_partial_load_curve_is_anchored_at_zero() -> None:
    rel_full, eff_full, fuel_raw_full, fuel_surrogate_full = _validate_generator_partial_load_curve(
        rel=np.array([0.5, 1.0], dtype=float),
        eff=np.array([0.30, 0.30], dtype=float),
        path=Path("generator_efficiency_curve.csv"),
    )

    assert np.allclose(rel_full, np.array([0.0, 0.5, 1.0]))
    assert eff_full.shape == rel_full.shape
    assert fuel_raw_full.shape == rel_full.shape
    assert fuel_surrogate_full.shape == rel_full.shape
    assert np.isclose(eff_full[0], 0.0)
    assert np.allclose(fuel_raw_full, fuel_surrogate_full)


def test_generator_partial_load_curve_builds_conservative_surrogate_for_nonconvex_fuel_curve() -> None:
    rel_full, eff_full, fuel_raw_full, fuel_surrogate_full = _validate_generator_partial_load_curve(
        rel=np.array([0.5, 1.0], dtype=float),
        eff=np.array([0.20, 1.00], dtype=float),
        path=Path("generator_efficiency_curve.csv"),
    )

    assert np.allclose(rel_full, np.array([0.0, 0.5, 1.0]))
    assert np.isclose(eff_full[-1], 1.0)
    assert np.all(fuel_surrogate_full >= fuel_raw_full - 1e-10)
    slopes = np.diff(fuel_surrogate_full) / np.diff(rel_full)
    assert np.all(np.diff(slopes) >= -1e-10)


def test_generator_partial_load_curve_accepts_normalized_multiplier_with_zero_anchor() -> None:
    eff_abs, eff_mult, mode = resolve_efficiency_curve_values(
        np.array([0.0, 0.67, 0.87, 0.97, 1.0], dtype=float),
        base_efficiency=0.30,
        path=Path("generator_efficiency_curve.csv"),
        column_name="Efficiency [-]",
        allow_zero=True,
    )

    rel_full, eff_full, fuel_raw_full, fuel_surrogate_full = _validate_generator_partial_load_curve(
        rel=np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
        eff=eff_abs,
        path=Path("generator_efficiency_curve.csv"),
    )

    assert mode == "normalized_multiplier"
    assert np.isclose(eff_mult[-1], 1.0)
    assert np.isclose(eff_full[-1], 0.30)
    assert np.allclose(rel_full, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(fuel_raw_full[1:], rel_full[1:] / eff_full[1:])
    assert np.all(fuel_surrogate_full >= fuel_raw_full - 1e-10)
