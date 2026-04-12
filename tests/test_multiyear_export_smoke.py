from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from core.export.multi_year_results import export_multi_year_results
from core.multi_year_model.data import initialize_data
from core.multi_year_model.sets import initialize_sets


class _VarStub:
    def __init__(self, solution: xr.DataArray) -> None:
        self.solution = solution


def _build_stub_vars(sets: xr.Dataset) -> dict[str, _VarStub]:
    period = sets.coords["period"]
    year = sets.coords["year"]
    scenario = sets.coords["scenario"]
    resource = sets.coords["resource"]
    inv_step = sets.coords["inv_step"]

    t, y, s, r, k = int(period.size), int(year.size), int(scenario.size), int(resource.size), int(inv_step.size)

    shape_pys = (t, y, s)
    shape_pysk = (t, y, s, k)
    shape_pysr = (t, y, s, r)

    res_generation = xr.DataArray(
        np.full(shape_pysr, 0.2, dtype=float),
        dims=("period", "year", "scenario", "resource"),
        coords={"period": period, "year": year, "scenario": scenario, "resource": resource},
    )
    generator_generation = xr.DataArray(
        np.full(shape_pysk, 0.1, dtype=float),
        dims=("period", "year", "scenario", "inv_step"),
        coords={"period": period, "year": year, "scenario": scenario, "inv_step": inv_step},
    )
    battery_charge = xr.DataArray(
        np.zeros(shape_pysk, dtype=float),
        dims=("period", "year", "scenario", "inv_step"),
        coords={"period": period, "year": year, "scenario": scenario, "inv_step": inv_step},
    )
    battery_discharge = xr.DataArray(
        np.zeros(shape_pysk, dtype=float),
        dims=("period", "year", "scenario", "inv_step"),
        coords={"period": period, "year": year, "scenario": scenario, "inv_step": inv_step},
    )
    battery_soc = xr.DataArray(
        np.zeros(shape_pysk, dtype=float),
        dims=("period", "year", "scenario", "inv_step"),
        coords={"period": period, "year": year, "scenario": scenario, "inv_step": inv_step},
    )
    lost_load = xr.DataArray(
        np.zeros((t, y, s), dtype=float),
        dims=("period", "year", "scenario"),
        coords={"period": period, "year": year, "scenario": scenario},
    )
    fuel_consumption = xr.DataArray(
        np.full(shape_pysk, 0.01, dtype=float),
        dims=("period", "year", "scenario", "inv_step"),
        coords={"period": period, "year": year, "scenario": scenario, "inv_step": inv_step},
    )
    grid_import = xr.DataArray(
        np.zeros(shape_pys, dtype=float),
        dims=("period", "year", "scenario"),
        coords={"period": period, "year": year, "scenario": scenario},
    )
    grid_export = xr.DataArray(
        np.zeros(shape_pys, dtype=float),
        dims=("period", "year", "scenario"),
        coords={"period": period, "year": year, "scenario": scenario},
    )

    return {
        "res_units": _VarStub(
            xr.DataArray(
                np.ones((k, r), dtype=float),
                dims=("inv_step", "resource"),
                coords={"inv_step": inv_step, "resource": resource},
            )
        ),
        "battery_units": _VarStub(
            xr.DataArray(
                np.ones((k,), dtype=float),
                dims=("inv_step",),
                coords={"inv_step": inv_step},
            )
        ),
        "generator_units": _VarStub(
            xr.DataArray(
                np.ones((k,), dtype=float),
                dims=("inv_step",),
                coords={"inv_step": inv_step},
            )
        ),
        "res_generation": _VarStub(res_generation),
        "generator_generation": _VarStub(generator_generation),
        "battery_charge": _VarStub(battery_charge),
        "battery_discharge": _VarStub(battery_discharge),
        "battery_soc": _VarStub(battery_soc),
        "lost_load": _VarStub(lost_load),
        "fuel_consumption": _VarStub(fuel_consumption),
        "grid_import": _VarStub(grid_import),
        "grid_export": _VarStub(grid_export),
    }


def _build_stub_vars_with_battery_diagnostics(sets: xr.Dataset) -> dict[str, _VarStub]:
    vars_dict = _build_stub_vars(sets)
    period = sets.coords["period"]
    year = sets.coords["year"]
    scenario = sets.coords["scenario"]
    inv_step = sets.coords["inv_step"]
    shape_pysk = (int(period.size), int(year.size), int(scenario.size), int(inv_step.size))
    shape_ysk = (int(year.size), int(scenario.size), int(inv_step.size))
    shape_yk = (int(year.size), int(inv_step.size))
    coords = {"period": period, "year": year, "scenario": scenario, "inv_step": inv_step}
    coords_year = {"year": year, "scenario": scenario, "inv_step": inv_step}
    coords_year_noscen = {"year": year, "inv_step": inv_step}
    soh = np.ones(shape_pysk, dtype=float)
    if int(inv_step.size) > 1:
        soh[..., 1] = 0.0
    eff_cap = np.ones(shape_ysk, dtype=float)
    if int(inv_step.size) > 1:
        eff_cap[:, :, 0] = 3.0
        eff_cap[:, :, 1] = 1.0
    vars_dict["battery_charge_dc"] = _VarStub(xr.DataArray(np.zeros(shape_pysk, dtype=float), dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_discharge_dc"] = _VarStub(xr.DataArray(np.zeros(shape_pysk, dtype=float), dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_charge_loss"] = _VarStub(xr.DataArray(np.zeros(shape_pysk, dtype=float), dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_discharge_loss"] = _VarStub(xr.DataArray(np.zeros(shape_pysk, dtype=float), dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_cycle_fade"] = _VarStub(xr.DataArray(np.zeros(shape_pysk, dtype=float), dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_average_soc"] = _VarStub(xr.DataArray(np.zeros(shape_yk, dtype=float), dims=("year", "inv_step"), coords=coords_year_noscen))
    vars_dict["battery_calendar_fade"] = _VarStub(xr.DataArray(np.zeros(shape_yk, dtype=float), dims=("year", "inv_step"), coords=coords_year_noscen))
    vars_dict["battery_soh"] = _VarStub(xr.DataArray(soh, dims=("period", "year", "scenario", "inv_step"), coords=coords))
    vars_dict["battery_effective_energy_capacity"] = _VarStub(xr.DataArray(eff_cap, dims=("year", "scenario", "inv_step"), coords=coords_year))
    return vars_dict


def test_multiyear_export_smoke(tmp_path: Path) -> None:
    project_name = "test_multiyear"
    sets = initialize_sets(project_name)
    data = initialize_data(project_name, sets)
    vars_dict = _build_stub_vars(sets)

    out = export_multi_year_results(
        project_name=project_name,
        sets=sets,
        data=data,
        model=None,
        vars=vars_dict,
        solution=None,
        out_dir=tmp_path,
    )

    assert Path(out["dispatch_timeseries_csv"]).exists()
    assert Path(out["energy_balance_csv"]).exists()
    assert Path(out["design_by_step_csv"]).exists()
    assert Path(out["kpis_yearly_csv"]).exists()
    assert Path(out["cashflows_discounted_csv"]).exists()

    dispatch = pd.read_csv(out["dispatch_timeseries_csv"])
    balance = pd.read_csv(out["energy_balance_csv"])
    design = pd.read_csv(out["design_by_step_csv"])
    kpis = pd.read_csv(out["kpis_yearly_csv"])
    cash = pd.read_csv(out["cashflows_discounted_csv"])

    assert {"period", "year", "scenario", "load_demand", "res_generation_total"}.issubset(dispatch.columns)
    assert {"period", "year", "scenario", "balance_residual"}.issubset(balance.columns)
    assert {"inv_step", "technology", "installed_capacity"}.issubset(design.columns)
    assert {"year", "scenario", "total_demand_kwh", "renewable_penetration"}.issubset(kpis.columns)
    assert {"year", "discounted_total", "salvage_credit_discounted", "salvage_tail_memo_discounted"}.issubset(cash.columns)


def test_multiyear_export_includes_optional_battery_diagnostics(tmp_path: Path) -> None:
    project_name = "test_multiyear"
    sets = initialize_sets(project_name)
    data = initialize_data(project_name, sets)
    vars_dict = _build_stub_vars_with_battery_diagnostics(sets)

    out = export_multi_year_results(
        project_name=project_name,
        sets=sets,
        data=data,
        model=None,
        vars=vars_dict,
        solution=None,
        out_dir=tmp_path / "diag",
    )

    dispatch = pd.read_csv(out["dispatch_timeseries_csv"])
    assert {
        "battery_charge_dc",
        "battery_discharge_dc",
        "battery_charge_loss",
        "battery_discharge_loss",
        "battery_cycle_fade",
        "battery_calendar_fade",
        "battery_soh",
        "battery_effective_energy_capacity",
    }.issubset(dispatch.columns)
    assert np.allclose(dispatch["battery_soh"].to_numpy(dtype=float), 0.75)


def test_multiyear_export_derives_battery_soh_from_effective_capacity_when_missing(tmp_path: Path) -> None:
    project_name = "test_multiyear"
    sets = initialize_sets(project_name)
    data = initialize_data(project_name, sets)
    vars_dict = _build_stub_vars_with_battery_diagnostics(sets)
    vars_dict.pop("battery_soh")

    out = export_multi_year_results(
        project_name=project_name,
        sets=sets,
        data=data,
        model=None,
        vars=vars_dict,
        solution=None,
        out_dir=tmp_path / "derived_soh",
    )

    dispatch = pd.read_csv(out["dispatch_timeseries_csv"])
    assert "battery_soh" in dispatch.columns
    assert np.all(np.isfinite(dispatch["battery_soh"].to_numpy(dtype=float)))
