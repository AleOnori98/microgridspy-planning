from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
import yaml

from core.io.templates import TemplateSettings, _safe_step_keys
from core.multi_year_model.data import (
    InputValidationError,
    _load_battery_yaml,
    _load_generator_and_fuel_yaml,
    _load_renewables_yaml,
    _remap_by_step_dict,
)


def _scenario_coord() -> xr.DataArray:
    labels = ["scenario_1"]
    return xr.DataArray(labels, dims=("scenario",), coords={"scenario": labels})


def _resource_coord() -> xr.DataArray:
    labels = ["Solar"]
    return xr.DataArray(labels, dims=("resource",), coords={"resource": labels})


def _step_coord_single() -> xr.DataArray:
    labels = [1]
    return xr.DataArray(labels, dims=("inv_step",), coords={"inv_step": labels})


def _step_coord_two() -> xr.DataArray:
    labels = [1, 2]
    return xr.DataArray(labels, dims=("inv_step",), coords={"inv_step": labels})


def _year_coord() -> xr.DataArray:
    labels = [2026, 2027]
    return xr.DataArray(labels, dims=("year",), coords={"year": labels})


def test_multi_year_templates_use_numeric_step_keys() -> None:
    settings = TemplateSettings(
        formulation="dynamic",
        system_type="off_grid",
        allow_export=False,
        multi_scenario=False,
        n_scenarios=1,
        scenario_labels=["scenario_1"],
        scenario_weights=[1.0],
        start_year_label="2026",
        horizon_years=10,
        capacity_expansion=True,
        investment_steps_years=[5, 5],
        n_res_sources=1,
        resource_labels=["Solar"],
        conversion_labels=["Solar_PV"],
        battery_label="Battery",
        battery_loss_model="constant_efficiency",
        battery_cycle_fade_enabled=False,
        battery_calendar_fade_enabled=False,
        battery_efficiency_curve_csv="battery_efficiency_curve.csv",
        battery_cycle_lifetime_to_eol_cycles=6000.0,
        battery_calendar_fade_curve_csv="battery_calendar_fade_curve.csv",
        battery_calendar_time_increment_per_step=1.0,
        battery_end_of_life_soh=0.8,
        generator_label="Generator",
        generator_efficiency_model="constant_efficiency",
        generator_efficiency_curve_csv="generator_efficiency_curve.csv",
        fuel_label="Fuel",
    )

    assert _safe_step_keys(settings) == ["1", "2"]


def test_multi_year_step_aliases_are_centralized_for_backward_compatibility(tmp_path: Path) -> None:
    remapped = _remap_by_step_dict(
        tmp_path / "dummy.yaml",
        {"step_1": {"x": 1}, "step_2": {"x": 2}},
        expected_steps=["1", "2"],
        context="component.investment.by_step",
    )

    assert list(remapped.keys()) == ["1", "2"]


def test_multi_year_base_alias_rejected_for_multi_step_projects(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="legacy key 'base'"):
        _remap_by_step_dict(
            tmp_path / "dummy.yaml",
            {"base": {"x": 1}},
            expected_steps=["1", "2"],
            context="component.investment.by_step",
        )


def test_multi_year_battery_requires_initial_soh_when_degradation_is_active(tmp_path: Path) -> None:
    path = tmp_path / "battery.yaml"
    payload = {
        "battery": {
            "label": "Battery",
            "investment": {
                "by_step": {
                    "1": {
                        "nominal_capacity_kwh": 1.0,
                        "specific_investment_cost_per_kwh": 350.0,
                        "wacc": 0.05,
                        "calendar_lifetime_years": 10,
                        "embedded_emissions_kgco2e_per_kwh": 0.0,
                        "fixed_om_share_per_year": 0.02,
                    }
                }
            },
            "technical": {
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.96,
                "initial_soc": 0.5,
                "depth_of_discharge": 0.8,
                "max_discharge_time_hours": 5.0,
                "max_charge_time_hours": 5.0,
                "max_installable_capacity_kwh": None,
                "efficiency_curve_csv": "battery_efficiency_curve.csv",
                "end_of_life_soh": 0.8,
                "cycle_lifetime_to_eol_cycles": 6000.0,
            },
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InputValidationError, match="initial_soh"):
        _load_battery_yaml(
            path,
            scenario_coord=_scenario_coord(),
            inv_step_coord=_step_coord_single(),
            require_initial_soh=True,
        )


def test_multi_year_battery_reads_initial_soh_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "battery.yaml"
    payload = {
        "battery": {
            "label": "Battery",
            "investment": {
                "by_step": {
                    "1": {
                        "nominal_capacity_kwh": 1.0,
                        "specific_investment_cost_per_kwh": 350.0,
                        "wacc": 0.05,
                        "calendar_lifetime_years": 10,
                        "embedded_emissions_kgco2e_per_kwh": 0.0,
                        "fixed_om_share_per_year": 0.02,
                    }
                }
            },
            "technical": {
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.96,
                "initial_soc": 0.5,
                "initial_soh": 1.0,
                "depth_of_discharge": 0.8,
                "max_discharge_time_hours": 5.0,
                "max_charge_time_hours": 5.0,
                "max_installable_capacity_kwh": None,
                "efficiency_curve_csv": "battery_efficiency_curve.csv",
                "end_of_life_soh": 0.8,
                "cycle_lifetime_to_eol_cycles": 6000.0,
            },
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    ds = _load_battery_yaml(
        path,
        scenario_coord=_scenario_coord(),
        inv_step_coord=_step_coord_single(),
        require_initial_soh=True,
    )

    assert float(ds["battery_initial_soh"]) == 1.0


def test_multi_year_renewables_reject_legacy_operation_block(tmp_path: Path) -> None:
    path = tmp_path / "renewables.yaml"
    payload = {
        "renewables": [
            {
                "id": "res_1",
                "conversion_technology": "Solar_PV",
                "resource": "Solar",
                "investment": {
                    "by_step": {
                        "1": {
                            "nominal_capacity_kw": 1.0,
                            "specific_investment_cost_per_kw": 800.0,
                            "wacc": 0.05,
                            "grant_share_of_capex": 0.0,
                            "lifetime_years": 25,
                            "embedded_emissions_kgco2e_per_kw": 0.0,
                            "fixed_om_share_per_year": 0.02,
                            "production_subsidy_per_kwh": 0.0,
                        }
                    }
                },
                "technical": {
                    "inverter_efficiency": 1.0,
                    "specific_area_m2_per_kw": None,
                    "max_installable_capacity_kw": None,
                    "capacity_degradation_rate_per_year": 0.0,
                },
                "operation": {"by_scenario": {"scenario_1": {"fixed_om_share_per_year": 0.02}}},
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InputValidationError, match="legacy `operation`"):
        _load_renewables_yaml(
            path,
            scenario_coord=_scenario_coord(),
            resource_coord=_resource_coord(),
            inv_step_coord=_step_coord_single(),
        )


def test_multi_year_generator_rejects_legacy_fuel_by_scenario(tmp_path: Path) -> None:
    path = tmp_path / "generator.yaml"
    payload = {
        "generator": {
            "label": "Generator",
            "investment": {
                "by_step": {
                    "1": {
                        "nominal_capacity_kw": 1.0,
                        "lifetime_years": 10,
                        "specific_investment_cost_per_kw": 400.0,
                        "wacc": 0.07,
                        "embedded_emissions_kgco2e_per_kw": 0.0,
                        "fixed_om_share_per_year": 0.03,
                    }
                }
            },
            "technical": {
                "nominal_efficiency_full_load": 0.30,
                "efficiency_curve_csv": None,
                "max_installable_capacity_kw": None,
                "capacity_degradation_rate_per_year": 0.0,
            },
        },
        "fuel": {
            "label": "Diesel",
            "by_scenario": {
                "scenario_1": {
                    "lhv_kwh_per_unit_fuel": 10.0,
                    "direct_emissions_kgco2e_per_unit_fuel": 0.0,
                    "by_year_cost_per_unit_fuel": [1.2, 1.2],
                }
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InputValidationError, match="legacy `by_scenario`"):
        _load_generator_and_fuel_yaml(
            path,
            inputs_dir=tmp_path,
            scenario_coord=_scenario_coord(),
            inv_step_coord=_step_coord_single(),
            year_coord=_year_coord(),
        )
