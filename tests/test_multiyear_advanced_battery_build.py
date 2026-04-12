from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from core.multi_year_model.model import MultiYearModel


def test_multiyear_advanced_battery_build_smoke() -> None:
    m = MultiYearModel("test_multiyear_constrained")
    m._initialize_sets()
    m._initialize_data()
    m._initialize_vars()
    m._initialize_constraints()

    assert "battery_charge_dc" in m.vars
    assert "battery_discharge_loss" in m.vars
    assert "battery_cycle_fade" in m.vars
    assert "battery_average_soc" in m.vars
    assert "battery_calendar_fade" in m.vars
    assert "battery_effective_energy_capacity" in m.vars
    assert "battery_charge_loss_slope" in m.data
    assert "battery_discharge_loss_intercept" in m.data
    assert "battery_calendar_fade_slope" in m.data
    assert "battery_charge_dc_limit" in m.model.constraints
    assert "battery_discharge_dc_limit" in m.model.constraints
    assert "battery_charge_ac_dc_coupling" in m.model.constraints
    assert "battery_discharge_ac_dc_coupling" in m.model.constraints
    assert "battery_cycle_fade_definition" in m.model.constraints
    assert "battery_calendar_fade_epigraph" in m.model.constraints
    assert "battery_effective_energy_capacity_upper_available" in m.model.constraints
    assert "battery_effective_energy_capacity_initial" in m.model.constraints
    assert "battery_average_soc_definition" in m.model.constraints
    assert "battery_max_installable_capacity" in m.model.constraints
    assert "soc_initial" in m.model.constraints
    assert any(str(name).startswith("soc_year_link_") for name in m.model.constraints)
    assert any(str(name).startswith("battery_effective_energy_capacity_year_link_") for name in m.model.constraints)
    assert "soc_cyclic" not in m.model.constraints
    assert m.data["battery_charge_efficiency"].dims == ()
    assert m.data["battery_initial_soc"].dims == ()
    assert m.data["battery_charge_loss_slope"].dims == ("battery_loss_segment",)
    assert m.data["battery_calendar_fade_slope"].dims == ("battery_calendar_segment",)
    assert m.data["generator_nominal_efficiency_full_load"].dims == ()
    assert m.data["fuel_lhv_kwh_per_unit_fuel"].dims == ()
    assert m.data["res_fixed_om_share_per_year"].dims == ("inv_step", "resource")
    assert m.data["res_capacity_degradation_rate_per_year"].dims == ("resource",)


def test_multiyear_constant_efficiency_keeps_legacy_public_interface() -> None:
    m = MultiYearModel("test_multiyear")
    m._initialize_sets()
    m._initialize_data()
    m._initialize_vars()
    m._initialize_constraints()

    assert "battery_charge_dc" not in m.vars
    assert "battery_discharge_loss" not in m.vars
    assert "battery_cycle_fade" not in m.vars
    assert "battery_calendar_fade" not in m.vars
    assert "battery_average_soc" not in m.vars
    assert "battery_effective_energy_capacity" not in m.vars
    assert "battery_charge_ac_dc_coupling" not in m.model.constraints
    assert "battery_discharge_loss_epigraph" not in m.model.constraints
    assert "soc_initial" in m.model.constraints
    assert any(str(name).startswith("soc_year_link_") for name in m.model.constraints)
    assert "soc_cyclic" not in m.model.constraints


def test_multiyear_degradation_builds_without_max_installable_capacity() -> None:
    m = MultiYearModel("test_multiyear_constrained")
    m._initialize_sets()
    m._initialize_data()
    m.data["battery_max_installable_capacity_kwh"] = xr.DataArray(np.nan)
    m._initialize_vars()
    m._initialize_constraints()
    assert "battery_effective_energy_capacity_upper_available" in m.model.constraints


def test_multiyear_battery_negative_max_installable_capacity_is_rejected() -> None:
    m = MultiYearModel("test_multiyear_constrained")
    m._initialize_sets()
    m._initialize_data()
    m.data["battery_max_installable_capacity_kwh"] = xr.DataArray(-1.0)
    m._initialize_vars()

    with pytest.raises(RuntimeError, match="battery_max_installable_capacity_kwh"):
        m._initialize_constraints()


def test_multiyear_endogenous_battery_degradation_ignores_legacy_exogenous_capacity_fade() -> None:
    m = MultiYearModel("test_multiyear_constrained")
    m._initialize_sets()
    m._initialize_data()
    m.data["battery_capacity_degradation_rate_per_year"] = xr.full_like(
        m.data["battery_capacity_degradation_rate_per_year"],
        0.25,
        dtype=float,
    )
    m._initialize_vars()
    m._initialize_constraints()

    assert "battery_cycle_fade_definition" in m.model.constraints
    assert "battery_effective_energy_capacity_upper_available" in m.model.constraints


def test_multiyear_cycle_fade_only_keeps_exogenous_capacity_fade() -> None:
    m = MultiYearModel("test_multiyear_constrained")
    m._initialize_sets()
    m._initialize_data()
    settings = (m.data.attrs or {}).get("settings", {})
    battery_model = (settings.get("battery_model", {}) or {})
    degradation_model = (battery_model.get("degradation_model", {}) or {})
    degradation_model["calendar_fade_enabled"] = False
    degradation_model["cycle_fade_enabled"] = True
    battery_model["degradation_model"] = degradation_model
    settings["battery_model"] = battery_model
    m.data.attrs["settings"] = settings
    m.data["battery_capacity_degradation_rate_per_year"] = xr.full_like(
        m.data["battery_capacity_degradation_rate_per_year"],
        0.25,
        dtype=float,
    )
    m._initialize_vars()
    m._initialize_constraints()

    assert "battery_cycle_fade_definition" in m.model.constraints
