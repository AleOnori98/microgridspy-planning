from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
import yaml

from core.data_pipeline.typical_year_parsing import (
    InputValidationError,
    _load_battery_yaml,
    _load_generator_and_fuel_yaml,
    _load_renewables_yaml,
)


def _scenario_coord() -> xr.DataArray:
    labels = ["scenario_1"]
    return xr.DataArray(labels, dims=("scenario",), coords={"scenario": labels})


def _resource_coord() -> xr.DataArray:
    labels = ["Solar"]
    return xr.DataArray(labels, dims=("resource",), coords={"resource": labels})


def test_typical_generator_accepts_single_non_base_step_block(tmp_path: Path) -> None:
    path = tmp_path / "generator.yaml"
    payload = {
        "generator": {
            "label": "Generator",
            "investment": {
                "by_step": {
                    "step_1": {
                        "nominal_capacity_kw": 1.0,
                        "lifetime_years": 10,
                        "specific_investment_cost_per_kw": 400.0,
                        "wacc": 0.07,
                    }
                }
            },
            "technical": {
                "nominal_efficiency_full_load": 0.30,
                "efficiency_curve_csv": None,
                "max_installable_capacity_kw": None,
            },
        },
        "fuel": {
            "label": "Diesel",
            "by_scenario": {
                "scenario_1": {
                    "lhv_kwh_per_unit_fuel": 10.0,
                    "direct_emissions_kgco2e_per_unit_fuel": 0.0,
                    "fuel_cost_per_unit_fuel": 1.2,
                }
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    gen_ds, fuel_ds, curve_ds, meta = _load_generator_and_fuel_yaml(
        path,
        inputs_dir=tmp_path,
        scenario_coord=_scenario_coord(),
    )

    assert float(gen_ds["generator_nominal_capacity_kw"]) == 1.0
    assert float(fuel_ds["fuel_fuel_cost_per_unit_fuel"].sel(scenario="scenario_1")) == 1.2
    assert curve_ds is None
    assert meta["partial_load_modelling_enabled"] is False


def test_typical_battery_rejects_initial_soh_in_yaml(tmp_path: Path) -> None:
    path = tmp_path / "battery.yaml"
    payload = {
        "battery": {
            "label": "Battery",
            "investment": {
                "by_step": {
                    "base": {
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
                "efficiency_curve_csv": None,
            },
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InputValidationError, match="initial_soh"):
        _load_battery_yaml(path, scenario_coord=_scenario_coord())


def test_typical_renewables_reject_legacy_operation_block(tmp_path: Path) -> None:
    path = tmp_path / "renewables.yaml"
    payload = {
        "renewables": [
            {
                "id": "res_1",
                "conversion_technology": "Solar_PV",
                "resource": "Solar",
                "investment": {
                    "by_step": {
                        "base": {
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
                },
                "operation": {
                    "by_scenario": {
                        "scenario_1": {
                            "fixed_om_share_per_year": 0.02,
                        }
                    }
                },
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InputValidationError, match="legacy `operation` block"):
        _load_renewables_yaml(
            path,
            scenario_coord=_scenario_coord(),
            resource_coord=_resource_coord(),
        )
