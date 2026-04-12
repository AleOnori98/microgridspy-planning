from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import linopy as lp

from core.data_pipeline.battery_loss_model import (
    CONVEX_LOSS_EPIGRAPH,
    CONSTANT_EFFICIENCY,
    get_battery_loss_model_from_formulation,
    load_battery_loss_curve_dataset,
)
from core.data_pipeline.battery_degradation_model import (
    get_battery_degradation_settings,
    suppress_exogenous_battery_capacity_degradation_when_endogenous,
)
from core.data_pipeline.battery_calendar_fade_model import load_battery_calendar_fade_curve_dataset
from core.typical_year_model.variables import initialize_vars as initialize_typical_vars
from core.typical_year_model.constraints import initialize_constraints as initialize_typical_constraints


def _write_curve_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "relative_power_pu,charge_efficiency,discharge_efficiency",
                *rows,
            ]
        ),
        encoding="utf-8",
    )


def _write_calendar_curve_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "soc_pu,calendar_fade_coefficient_per_year",
                *rows,
            ]
        ),
        encoding="utf-8",
    )


def test_battery_loss_model_defaults_to_constant_efficiency() -> None:
    assert get_battery_loss_model_from_formulation({}) == CONSTANT_EFFICIENCY


def test_battery_loss_curve_builds_epigraph_dataset(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_efficiency_curve.csv"
    _write_curve_csv(
        curve_path,
        [
            "0.10,0.9766,0.9870",
            "0.25,0.9709,0.9825",
            "0.50,0.9615,0.9750",
            "0.75,0.9524,0.9675",
            "1.00,0.9434,0.9600",
        ],
    )

    ds = load_battery_loss_curve_dataset(curve_path)
    assert "battery_charge_loss_slope" in ds
    assert "battery_discharge_loss_intercept" in ds
    assert int(ds.sizes["battery_curve_point"]) == 6
    assert int(ds.sizes["battery_loss_segment"]) == 5
    assert float(ds["battery_charge_loss_curve_pu"].min()) >= 0.0
    assert float(ds["battery_discharge_loss_curve_pu"].min()) >= 0.0
    assert (ds["battery_charge_loss_curve_pu"].diff("battery_curve_point") >= 0.0).all().item()
    assert (ds["battery_discharge_loss_curve_pu"].diff("battery_curve_point") >= 0.0).all().item()
    assert (ds["battery_charge_loss_slope"].diff("battery_loss_segment") >= -1e-9).all().item()
    assert (ds["battery_discharge_loss_slope"].diff("battery_loss_segment") >= -1e-9).all().item()


def test_battery_loss_curve_matches_expected_efficiency_transform(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_efficiency_curve.csv"
    _write_curve_csv(
        curve_path,
        [
            "0.50,0.9523809524,0.9600",
            "1.00,0.9090909091,0.9000",
        ],
    )

    ds = load_battery_loss_curve_dataset(curve_path)

    charge_loss = ds["battery_charge_loss_curve_pu"].values.astype(float)
    discharge_loss = ds["battery_discharge_loss_curve_pu"].values.astype(float)

    assert np.isclose(charge_loss[0], 0.0)
    assert np.isclose(charge_loss[-1], 0.1, atol=1e-9)
    assert np.isclose(discharge_loss[0], 0.0)
    assert np.isclose(discharge_loss[-1], 0.1, atol=1e-9)


def test_invalid_non_monotone_power_curve_is_rejected(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_efficiency_curve.csv"
    _write_curve_csv(
        curve_path,
        [
            "0.10,0.97,0.98",
            "0.50,0.95,0.96",
            "0.40,0.94,0.95",
            "1.00,0.92,0.93",
        ],
    )

    with pytest.raises(RuntimeError, match="strictly increasing"):
        load_battery_loss_curve_dataset(curve_path)


def test_invalid_curve_without_full_load_point_is_rejected(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_efficiency_curve.csv"
    _write_curve_csv(
        curve_path,
        [
            "0.10,0.97,0.98",
            "0.50,0.95,0.96",
            "0.90,0.92,0.93",
        ],
    )

    with pytest.raises(RuntimeError, match="must be 1.0"):
        load_battery_loss_curve_dataset(curve_path)


def test_invalid_nonconvex_curve_is_rejected(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_efficiency_curve.csv"
    _write_curve_csv(
        curve_path,
        [
            "0.10,0.90,0.90",
            "0.25,0.93,0.93",
            "0.50,0.96,0.96",
            "0.75,0.965,0.965",
            "1.00,0.97,0.97",
        ],
    )

    with pytest.raises(RuntimeError, match="not convex"):
        load_battery_loss_curve_dataset(curve_path)


def test_invalid_battery_loss_model_raises() -> None:
    with pytest.raises(RuntimeError):
        get_battery_loss_model_from_formulation({"battery_model": {"loss_model": "not_valid"}})


def test_explicit_advanced_mode_is_detected() -> None:
    formulation = {"battery_model": {"loss_model": CONVEX_LOSS_EPIGRAPH}}
    assert get_battery_loss_model_from_formulation(formulation) == CONVEX_LOSS_EPIGRAPH


def test_cycle_fade_requires_convex_loss_model() -> None:
    with pytest.raises(RuntimeError, match="requires battery_model.loss_model='convex_loss_epigraph'"):
        get_battery_degradation_settings(
            {
                "core_formulation": "dynamic",
                "battery_model": {
                    "degradation_model": {
                        "cycle_fade_enabled": True,
                    }
                }
            },
            battery_loss_model=CONSTANT_EFFICIENCY,
        )


def test_calendar_fade_requires_convex_loss_model() -> None:
    with pytest.raises(RuntimeError, match="requires battery_model.loss_model='convex_loss_epigraph'"):
        get_battery_degradation_settings(
            {
                "core_formulation": "dynamic",
                "battery_model": {
                    "degradation_model": {
                        "calendar_fade_enabled": True,
                        "battery_calendar_fade_curve_csv": "battery_calendar_fade_curve.csv",
                    }
                }
            },
            battery_loss_model=CONSTANT_EFFICIENCY,
        )


def test_degradation_requires_dynamic_formulation() -> None:
    with pytest.raises(RuntimeError, match="dynamic multi-year formulation"):
        get_battery_degradation_settings(
            {
                "core_formulation": "steady_state",
                "battery_model": {
                    "degradation_model": {
                        "cycle_fade_enabled": True,
                    }
                }
            },
            battery_loss_model=CONVEX_LOSS_EPIGRAPH,
        )


def test_calendar_fade_suppresses_exogenous_battery_capacity_degradation() -> None:
    rate = xr.DataArray([[0.01, 0.02]], dims=("scenario", "inv_step"), coords={"scenario": ["s1"], "inv_step": ["1", "2"]})
    active_rate, ignored = suppress_exogenous_battery_capacity_degradation_when_endogenous(
        rate,
        calendar_fade_enabled=True,
    )
    assert ignored is True
    assert isinstance(active_rate, xr.DataArray)
    assert np.allclose(active_rate.values, 0.0)


def test_cycle_fade_only_keeps_exogenous_battery_capacity_degradation() -> None:
    rate = xr.DataArray([[0.01, 0.02]], dims=("scenario", "inv_step"), coords={"scenario": ["s1"], "inv_step": ["1", "2"]})
    active_rate, ignored = suppress_exogenous_battery_capacity_degradation_when_endogenous(
        rate,
        calendar_fade_enabled=False,
    )
    assert ignored is False
    assert active_rate is rate


def test_exogenous_battery_capacity_degradation_is_kept_when_endogenous_disabled() -> None:
    rate = xr.DataArray([[0.01, 0.02]], dims=("scenario", "inv_step"), coords={"scenario": ["s1"], "inv_step": ["1", "2"]})
    active_rate, ignored = suppress_exogenous_battery_capacity_degradation_when_endogenous(
        rate,
        calendar_fade_enabled=False,
    )
    assert ignored is False
    assert active_rate is rate


def test_calendar_curve_builds_dataset(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_calendar_fade_curve.csv"
    _write_calendar_curve_csv(
        curve_path,
        [
            "0.0,1.0e-6",
            "0.2,1.2e-6",
            "0.5,1.8e-6",
            "0.8,3.0e-6",
            "1.0,4.5e-6",
        ],
    )

    ds = load_battery_calendar_fade_curve_dataset(curve_path)
    assert "battery_calendar_fade_slope" in ds
    assert "battery_calendar_fade_intercept" in ds
    assert (ds["battery_calendar_fade_curve_coefficient_per_year"] >= 0.0).all().item()


def test_legacy_calendar_curve_column_is_still_accepted(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_calendar_fade_curve.csv"
    curve_path.write_text(
        "\n".join(
            [
                "soc_pu,calendar_fade_coefficient_per_step",
                "0.0,1.0e-6",
                "0.5,2.0e-6",
                "1.0,4.0e-6",
            ]
        ),
        encoding="utf-8",
    )

    ds = load_battery_calendar_fade_curve_dataset(curve_path)
    assert ds.attrs["battery_calendar_curve_input_column"] == "calendar_fade_coefficient_per_step"
    assert "battery_calendar_fade_curve_coefficient_per_year" in ds


def test_invalid_nonconvex_calendar_curve_is_rejected(tmp_path: Path) -> None:
    curve_path = tmp_path / "battery_calendar_fade_curve.csv"
    _write_calendar_curve_csv(
        curve_path,
        [
            "0.0,5.0e-6",
            "0.5,2.0e-6",
            "1.0,1.0e-6",
        ],
    )

    with pytest.raises(RuntimeError, match="not convex"):
        load_battery_calendar_fade_curve_dataset(curve_path)


def test_legacy_calendar_time_increment_mode_is_normalized() -> None:
    settings = get_battery_degradation_settings(
        {
            "core_formulation": "dynamic",
            "battery_model": {
                "loss_model": CONVEX_LOSS_EPIGRAPH,
                "degradation_model": {
                    "calendar_fade_enabled": True,
                    "battery_calendar_fade_curve_csv": "battery_calendar_fade_curve.csv",
                    "battery_calendar_time_increment_mode": "constant_per_step",
                    "battery_calendar_time_increment_per_step": 1.0,
                },
            },
        },
        battery_loss_model=CONVEX_LOSS_EPIGRAPH,
    )
    assert settings["battery_calendar_time_increment_mode"] == "constant_per_year"
    assert settings["battery_calendar_time_increment_per_year"] == 1.0


def _build_minimal_typical_data(
    *,
    advanced: bool,
    cycle_fade: bool = False,
    calendar_fade: bool = False,
    initial_soh: float = 1.0,
) -> tuple[xr.Dataset, xr.Dataset]:
    sets = xr.Dataset(
        coords={
            "period": ("period", [0, 1]),
            "scenario": ("scenario", ["scenario_1"]),
            "resource": ("resource", ["Solar"]),
        }
    )
    data = xr.Dataset(
        data_vars={
            "load_demand": xr.DataArray([[0.0], [0.0]], dims=("period", "scenario"), coords={"period": sets.period, "scenario": sets.scenario}),
            "resource_availability": xr.DataArray(
                np.zeros((2, 1, 1), dtype=float),
                dims=("period", "scenario", "resource"),
                coords={"period": sets.period, "scenario": sets.scenario, "resource": sets.resource},
            ),
            "scenario_weight": xr.DataArray([1.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "min_renewable_penetration": xr.DataArray(0.0),
            "max_lost_load_fraction": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "lost_load_cost_per_kwh": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "land_availability_m2": xr.DataArray(1.0e9),
            "emission_cost_per_kgco2e": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "res_nominal_capacity_kw": xr.DataArray([1.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_specific_investment_cost_per_kw": xr.DataArray([0.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_lifetime_years": xr.DataArray([20.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_wacc": xr.DataArray([0.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_grant_share_of_capex": xr.DataArray([0.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_fixed_om_share_per_year": xr.DataArray([0.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_production_subsidy_per_kwh": xr.DataArray([[0.0]], dims=("scenario", "resource"), coords={"scenario": sets.scenario, "resource": sets.resource}),
            "res_embedded_emissions_kgco2e_per_kw": xr.DataArray([[0.0]], dims=("scenario", "resource"), coords={"scenario": sets.scenario, "resource": sets.resource}),
            "res_inverter_efficiency": xr.DataArray([1.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_specific_area_m2_per_kw": xr.DataArray([0.0], dims=("resource",), coords={"resource": sets.resource}),
            "res_max_installable_capacity_kw": xr.DataArray([np.nan], dims=("resource",), coords={"resource": sets.resource}),
            "battery_nominal_capacity_kwh": xr.DataArray(1.0),
            "battery_specific_investment_cost_per_kwh": xr.DataArray(0.0),
            "battery_calendar_lifetime_years": xr.DataArray(10.0),
            "battery_wacc": xr.DataArray(0.0),
            "battery_fixed_om_share_per_year": xr.DataArray(0.0),
            "battery_embedded_emissions_kgco2e_per_kwh": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_charge_efficiency": xr.DataArray([0.95], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_discharge_efficiency": xr.DataArray([0.95], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_initial_soc": xr.DataArray([0.5], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_depth_of_discharge": xr.DataArray([0.8], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_max_charge_time_hours": xr.DataArray([5.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_max_discharge_time_hours": xr.DataArray([5.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "battery_max_installable_capacity_kwh": xr.DataArray(2.0),
            "generator_nominal_capacity_kw": xr.DataArray(1.0),
            "generator_specific_investment_cost_per_kw": xr.DataArray(0.0),
            "generator_lifetime_years": xr.DataArray(10.0),
            "generator_wacc": xr.DataArray(0.0),
            "generator_fixed_om_share_per_year": xr.DataArray(0.0),
            "generator_embedded_emissions_kgco2e_per_kw": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "generator_nominal_efficiency_full_load": xr.DataArray(0.3),
            "fuel_lhv_kwh_per_unit_fuel": xr.DataArray([1.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "fuel_fuel_cost_per_unit_fuel": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
            "fuel_direct_emissions_kgco2e_per_unit_fuel": xr.DataArray([0.0], dims=("scenario",), coords={"scenario": sets.scenario}),
        }
    )
    data.attrs["settings"] = {
        "unit_commitment": False,
        "multi_scenario": {"enabled": False, "n_scenarios": 1},
        "resources": {"n_resources": 1, "resource_labels": ["Solar"]},
        "optimization_constraints": {"enforcement": "scenario_wise"},
        "grid": {"on_grid": False, "allow_export": False},
        "battery_model": {
            "loss_model": CONVEX_LOSS_EPIGRAPH if advanced else CONSTANT_EFFICIENCY,
            "degradation_model": {
                "cycle_fade_enabled": cycle_fade,
                "calendar_fade_enabled": calendar_fade,
                "cycle_fade_coefficient_per_kwh_throughput": 0.01,
                "battery_calendar_fade_curve_csv": "battery_calendar_fade_curve.csv",
                "battery_calendar_time_increment_mode": "constant_per_step",
                "battery_calendar_time_increment_per_year": 1.0,
                "initial_soh": initial_soh,
            },
        },
    }
    if cycle_fade or calendar_fade:
        data = xr.merge(
            [
                data,
                xr.Dataset(
                    {
                        "battery_cycle_fade_coefficient_per_kwh_throughput": xr.DataArray(0.01),
                        "battery_calendar_time_increment_per_year": xr.DataArray(1.0),
                        "battery_initial_soh": xr.DataArray(initial_soh),
                    }
                ),
            ],
            compat="override",
        )
    if calendar_fade:
        calendar_curve_ds = xr.Dataset(
            data_vars={
                "battery_calendar_fade_slope": xr.DataArray([1.0e-6, 2.0e-6], dims=("battery_calendar_segment",), coords={"battery_calendar_segment": [0, 1]}),
                "battery_calendar_fade_intercept": xr.DataArray([1.0e-6, 0.0], dims=("battery_calendar_segment",), coords={"battery_calendar_segment": [0, 1]}),
            }
        )
        data = xr.merge([data, calendar_curve_ds], compat="override")
    if advanced:
        curve_ds = xr.Dataset(
            data_vars={
                "battery_charge_loss_slope": xr.DataArray([0.05, 0.10], dims=("battery_loss_segment",), coords={"battery_loss_segment": [0, 1]}),
                "battery_charge_loss_intercept": xr.DataArray([0.0, -0.025], dims=("battery_loss_segment",), coords={"battery_loss_segment": [0, 1]}),
                "battery_discharge_loss_slope": xr.DataArray([0.02, 0.04], dims=("battery_loss_segment",), coords={"battery_loss_segment": [0, 1]}),
                "battery_discharge_loss_intercept": xr.DataArray([0.0, -0.01], dims=("battery_loss_segment",), coords={"battery_loss_segment": [0, 1]}),
            }
        )
        data = xr.merge([data, curve_ds], compat="override")
    return sets, data


def test_typical_advanced_mode_adds_dc_vars_and_constraints() -> None:
    sets, data = _build_minimal_typical_data(advanced=True)
    model = lp.Model()
    vars_dict = initialize_typical_vars(sets, data, model)
    initialize_typical_constraints(sets, data, vars_dict, model)

    assert "battery_charge_dc" in vars_dict
    assert "battery_discharge_loss" in vars_dict
    assert "battery_charge_dc_limit" in model.constraints
    assert "battery_discharge_dc_limit" in model.constraints
    assert "battery_charge_ac_dc_coupling" in model.constraints
    assert "battery_discharge_ac_dc_coupling" in model.constraints
    assert "battery_charge_loss_epigraph" in model.constraints
    assert "battery_discharge_loss_epigraph" in model.constraints


def test_typical_constant_mode_keeps_legacy_public_interface() -> None:
    sets, data = _build_minimal_typical_data(advanced=False)
    model = lp.Model()
    vars_dict = initialize_typical_vars(sets, data, model)
    initialize_typical_constraints(sets, data, vars_dict, model)

    assert "battery_charge_dc" not in vars_dict
    assert "battery_discharge_loss" not in vars_dict
    assert "battery_charge_ac_dc_coupling" not in model.constraints
    assert "battery_discharge_loss_epigraph" not in model.constraints


def test_typical_cycle_fade_adds_soh_state_and_constraints() -> None:
    sets, data = _build_minimal_typical_data(advanced=True, cycle_fade=True)
    model = lp.Model()
    with pytest.raises(RuntimeError, match="not available in the steady_state typical-year formulation"):
        initialize_typical_vars(sets, data, model)


def test_typical_calendar_fade_adds_constraints() -> None:
    sets, data = _build_minimal_typical_data(advanced=True, calendar_fade=True)
    model = lp.Model()
    with pytest.raises(RuntimeError, match="not available in the steady_state typical-year formulation"):
        initialize_typical_vars(sets, data, model)


def test_typical_lower_initial_soh_still_builds_soh_capacity_restriction() -> None:
    sets, data = _build_minimal_typical_data(advanced=True, cycle_fade=True, initial_soh=0.8)
    model = lp.Model()
    with pytest.raises(RuntimeError, match="not available in the steady_state typical-year formulation"):
        initialize_typical_vars(sets, data, model)


def test_typical_degradation_capacity_restriction_requires_positive_max_installable_capacity() -> None:
    sets, data = _build_minimal_typical_data(advanced=True, cycle_fade=True)
    data["battery_max_installable_capacity_kwh"] = xr.DataArray(np.nan)
    model = lp.Model()
    with pytest.raises(RuntimeError, match="not available in the steady_state typical-year formulation"):
        initialize_typical_vars(sets, data, model)


def test_typical_degradation_capacity_restriction_rejects_negative_max_installable_capacity() -> None:
    sets, data = _build_minimal_typical_data(advanced=True, cycle_fade=True)
    data["battery_max_installable_capacity_kwh"] = xr.DataArray(-1.0)
    model = lp.Model()
    with pytest.raises(RuntimeError, match="not available in the steady_state typical-year formulation"):
        initialize_typical_vars(sets, data, model)
