from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import linopy as lp

from core.export.common import (
    InputValidationError,
    ensure_results_dir,
    get_var_solution,
    require_data_array,
    safe_float,
    write_csv_outputs,
)
from core.multi_year_model.lifecycle import (
    discounted_annuity_tail_memo,
    replacement_active_mask,
    replacement_commission_mask,
    year_ordinal,
)
from core.multi_year_model.params import get_params

def _scenario_weights(p: Any, scenario_coord: xr.DataArray) -> xr.DataArray:
    if isinstance(p.scenario_weight, xr.DataArray):
        return p.scenario_weight.sel(scenario=scenario_coord)
    n = int(scenario_coord.size)
    return xr.DataArray(
        np.ones((n,), dtype=float) / float(n),
        dims=("scenario",),
        coords={"scenario": scenario_coord},
    )


def _sel_year_scenario_or_self(x: Any, *, year: Any, scenario: Any) -> Any:
    if not isinstance(x, xr.DataArray):
        return x
    indexers = {}
    if "year" in x.dims:
        indexers["year"] = year
    if "scenario" in x.dims:
        indexers["scenario"] = scenario
    return x.sel(**indexers) if indexers else x


def _scalarize_da(x: Any, **indexers: Any) -> float:
    if not isinstance(x, xr.DataArray):
        return float(x)
    da = x
    valid_indexers = {k: v for k, v in indexers.items() if k in da.dims}
    if valid_indexers:
        da = da.sel(**valid_indexers)
    extra_dims = [d for d in da.dims if da.sizes.get(d, 1) > 1]
    if extra_dims:
        da = da.isel({d: 0 for d in extra_dims})
    vals = np.asarray(da.values, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    return float(vals[0])


def _crf(r: xr.DataArray | float, n: xr.DataArray | float) -> xr.DataArray:
    rr = xr.DataArray(r)
    nn = xr.DataArray(n)
    a = (1.0 + rr) ** nn
    out = (rr * a) / (a - 1.0)
    out = xr.where(np.abs(rr) < 1e-12, 1.0 / nn, out)
    return xr.where(nn > 0, out, 0.0)


def build_dispatch_timeseries_table_multi_year(
    *,
    data: xr.Dataset,
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
) -> pd.DataFrame:
    p = get_params(data)
    load = require_data_array("load_demand", p.load_demand)
    res = require_data_array("res_generation", get_var_solution(vars_dict=vars, solution=solution, name="res_generation"))
    gen = require_data_array("generator_generation", get_var_solution(vars_dict=vars, solution=solution, name="generator_generation"))
    bch = require_data_array("battery_charge", get_var_solution(vars_dict=vars, solution=solution, name="battery_charge"))
    bdis = require_data_array("battery_discharge", get_var_solution(vars_dict=vars, solution=solution, name="battery_discharge"))
    bsoc = require_data_array("battery_soc", get_var_solution(vars_dict=vars, solution=solution, name="battery_soc"))
    ll = require_data_array("lost_load", get_var_solution(vars_dict=vars, solution=solution, name="lost_load"))

    gimp = get_var_solution(vars_dict=vars, solution=solution, name="grid_import") if p.is_grid_on() else None
    gexp = get_var_solution(vars_dict=vars, solution=solution, name="grid_export") if p.is_grid_export_enabled() else None

    idx = load.to_series().index
    df = pd.DataFrame(index=idx).reset_index()
    df["load_demand"] = load.to_series().values.astype(float)
    df["res_generation_total"] = res.sum("resource").to_series().values.astype(float)
    df["generator_generation"] = gen.to_series().values.astype(float)
    df["battery_charge"] = bch.to_series().values.astype(float)
    df["battery_discharge"] = bdis.to_series().values.astype(float)
    df["battery_soc"] = bsoc.to_series().values.astype(float)
    df["lost_load"] = ll.to_series().values.astype(float)
    df["grid_import"] = gimp.to_series().values.astype(float) if isinstance(gimp, xr.DataArray) else 0.0
    df["grid_export"] = gexp.to_series().values.astype(float) if isinstance(gexp, xr.DataArray) else 0.0
    return df


def build_energy_balance_table_multi_year(dispatch_df: pd.DataFrame) -> pd.DataFrame:
    df = dispatch_df.copy()
    df["supply_renewable"] = df["res_generation_total"]
    df["supply_generator"] = df["generator_generation"]
    df["supply_grid_import"] = df["grid_import"]
    df["supply_battery_discharge"] = df["battery_discharge"]
    df["supply_lost_load"] = df["lost_load"]
    df["sink_battery_charge"] = df["battery_charge"]
    df["sink_grid_export"] = df["grid_export"]
    df["demand"] = df["load_demand"]
    df["balance_lhs"] = (
        df["supply_renewable"]
        + df["supply_generator"]
        + df["supply_grid_import"]
        + df["supply_battery_discharge"]
        + df["supply_lost_load"]
        - df["sink_battery_charge"]
        - df["sink_grid_export"]
    )
    df["balance_residual"] = df["balance_lhs"] - df["demand"]
    cols = [
        "period",
        "year",
        "scenario",
        "demand",
        "supply_renewable",
        "supply_generator",
        "supply_grid_import",
        "supply_battery_discharge",
        "supply_lost_load",
        "sink_battery_charge",
        "sink_grid_export",
        "balance_lhs",
        "balance_residual",
    ]
    return df[cols]


def build_design_by_step_table_multi_year(
    *,
    sets: xr.Dataset,
    data: xr.Dataset,
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
) -> pd.DataFrame:
    p = get_params(data)
    res_units = require_data_array("res_units", get_var_solution(vars_dict=vars, solution=solution, name="res_units"))
    bat_units = require_data_array("battery_units", get_var_solution(vars_dict=vars, solution=solution, name="battery_units"))
    gen_units = require_data_array("generator_units", get_var_solution(vars_dict=vars, solution=solution, name="generator_units"))
    res_nom = require_data_array("res_nominal_capacity_kw", p.res_nominal_capacity_kw)
    bat_nom = require_data_array("battery_nominal_capacity_kwh", p.battery_nominal_capacity_kwh)
    gen_nom = require_data_array("generator_nominal_capacity_kw", p.generator_nominal_capacity_kw)

    rows = []
    for s in sets.coords["inv_step"].values:
        start_y = str(sets["inv_step_start_year"].sel(inv_step=s).item()) if "inv_step_start_year" in sets else ""
        for r in res_units.coords["resource"].values:
            u = float(res_units.sel(inv_step=s, resource=r))
            rows.append(
                {
                    "inv_step": s,
                    "inv_step_start_year": start_y,
                    "technology": "renewable",
                    "resource": str(r),
                    "units": u,
                    "installed_capacity": u * _scalarize_da(res_nom, inv_step=s, resource=r),
                    "capacity_unit": "kW",
                }
            )
        bu = float(bat_units.sel(inv_step=s))
        rows.append(
            {
                "inv_step": s,
                "inv_step_start_year": start_y,
                "technology": "battery",
                "resource": "",
                "units": bu,
                "installed_capacity": bu * _scalarize_da(bat_nom, inv_step=s),
                "capacity_unit": "kWh",
            }
        )
        gu = float(gen_units.sel(inv_step=s))
        rows.append(
            {
                "inv_step": s,
                "inv_step_start_year": start_y,
                "technology": "generator",
                "resource": "",
                "units": gu,
                "installed_capacity": gu * _scalarize_da(gen_nom, inv_step=s),
                "capacity_unit": "kW",
            }
        )
    return pd.DataFrame(rows)


def build_yearly_kpis_table_multi_year(
    *,
    data: xr.Dataset,
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
    objective_value: Optional[float] = None,
) -> pd.DataFrame:
    p = get_params(data)
    dispatch = build_dispatch_timeseries_table_multi_year(data=data, vars=vars, solution=solution)
    fuel = get_var_solution(vars_dict=vars, solution=solution, name="fuel_consumption")
    w = _scenario_weights(p, p.load_demand.coords["scenario"])

    rows = []
    for (year, scenario), g in dispatch.groupby(["year", "scenario"], as_index=False):
        demand = float(g["load_demand"].sum())
        ll = float(g["lost_load"].sum())
        served = demand - ll
        res = float(g["res_generation_total"].sum())
        gen = float(g["generator_generation"].sum())
        imp = float(g["grid_import"].sum())
        grid_eta = 1.0
        if p.grid_transmission_efficiency is not None:
            grid_eta = float(p.grid_transmission_efficiency.sel(scenario=scenario))
        grid_ren_share = 0.0
        if p.grid_renewable_share is not None:
            grid_ren_share = float(p.grid_renewable_share.sel(scenario=scenario))
        imp_delivered = imp * grid_eta
        imp_renewable = imp_delivered * grid_ren_share
        denom = res + gen + imp_delivered
        ren_pen = ((res + imp_renewable) / denom) if denom > 0 else 0.0
        ll_frac = (ll / demand) if demand > 0 else 0.0
        fuel_y = 0.0
        if isinstance(fuel, xr.DataArray):
            fuel_y = float(fuel.sel(year=year, scenario=scenario).sum("period"))
        em = 0.0
        if p.fuel_direct_emissions_kgco2e_per_unit_fuel is not None:
            em = fuel_y * float(
                _sel_year_scenario_or_self(
                    p.fuel_direct_emissions_kgco2e_per_unit_fuel,
                    year=year,
                    scenario=scenario,
                )
            )
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "total_demand_kwh": demand,
                "served_energy_kwh": served,
                "lost_load_kwh": ll,
                "lost_load_fraction": ll_frac,
                "total_res_kwh": res,
                "grid_renewable_kwh": imp_renewable,
                "renewable_penetration": ren_pen,
                "fuel_consumption": fuel_y,
                "emissions_kgco2e": em,
                "objective_value": safe_float(objective_value),
            }
        )

    out = pd.DataFrame(rows)
    w_map = {str(s): float(w.sel(scenario=s)) for s in w.coords["scenario"].values}
    out["weight"] = out["scenario"].astype(str).map(w_map).fillna(0.0)
    num_cols = [c for c in out.columns if c not in ("year", "scenario", "weight")]
    expected_rows = []
    for year, gy in out.groupby("year", as_index=False):
        row = {"year": year, "scenario": "expected"}
        for c in num_cols:
            if c == "objective_value":
                row[c] = safe_float(objective_value)
            else:
                row[c] = float((gy[c] * gy["weight"]).sum())
        expected_rows.append(row)
    return pd.concat([out.drop(columns=["weight"]), pd.DataFrame(expected_rows)], ignore_index=True)


def build_discounted_cashflows_table_multi_year(
    *,
    sets: xr.Dataset,
    data: xr.Dataset,
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
) -> pd.DataFrame:
    p = get_params(data)
    w = _scenario_weights(p, sets.coords["scenario"])
    rs = float((p.settings.get("social_discount_rate", 0.0) or 0.0))
    disc = 1.0 / ((1.0 + rs) ** year_ordinal(sets))

    res_units = require_data_array("res_units", get_var_solution(vars_dict=vars, solution=solution, name="res_units"))
    bat_units = require_data_array("battery_units", get_var_solution(vars_dict=vars, solution=solution, name="battery_units"))
    gen_units = require_data_array("generator_units", get_var_solution(vars_dict=vars, solution=solution, name="generator_units"))
    res_gen = require_data_array("res_generation", get_var_solution(vars_dict=vars, solution=solution, name="res_generation"))
    fuel_cons = require_data_array("fuel_consumption", get_var_solution(vars_dict=vars, solution=solution, name="fuel_consumption"))
    lost_load = require_data_array("lost_load", get_var_solution(vars_dict=vars, solution=solution, name="lost_load"))
    gimp = get_var_solution(vars_dict=vars, solution=solution, name="grid_import")
    gexp = get_var_solution(vars_dict=vars, solution=solution, name="grid_export")

    res_nom = require_data_array("res_nominal_capacity_kw", p.res_nominal_capacity_kw)
    res_capex = require_data_array("res_specific_investment_cost_per_kw", p.res_specific_investment_cost_per_kw)
    res_life = require_data_array("res_lifetime_years", p.res_lifetime_years)
    res_wacc = require_data_array("res_wacc", p.res_wacc)
    res_grant = require_data_array("res_grant_share_of_capex", p.res_grant_share_of_capex)
    bat_nom = require_data_array("battery_nominal_capacity_kwh", p.battery_nominal_capacity_kwh)
    bat_capex = require_data_array("battery_specific_investment_cost_per_kwh", p.battery_specific_investment_cost_per_kwh)
    bat_life = require_data_array("battery_calendar_lifetime_years", p.battery_calendar_lifetime_years)
    bat_wacc = require_data_array("battery_wacc", p.battery_wacc)
    gen_nom = require_data_array("generator_nominal_capacity_kw", p.generator_nominal_capacity_kw)
    gen_capex = require_data_array("generator_specific_investment_cost_per_kw", p.generator_specific_investment_cost_per_kw)
    gen_life = require_data_array("generator_lifetime_years", p.generator_lifetime_years)
    gen_wacc = require_data_array("generator_wacc", p.generator_wacc)

    res_inv = res_units * res_nom * res_capex * (1.0 - res_grant)
    bat_inv = bat_units * bat_nom * bat_capex
    gen_inv = gen_units * gen_nom * gen_capex
    ann_res = res_inv * _crf(res_wacc, res_life)
    ann_bat = bat_inv * _crf(bat_wacc, bat_life)
    ann_gen = gen_inv * _crf(gen_wacc, gen_life)
    act_res = replacement_active_mask(sets)
    act_bat = replacement_active_mask(sets)
    act_gen = replacement_active_mask(sets)
    ann_res_y = (ann_res * act_res).sum("inv_step").sum("resource")
    ann_bat_y = (ann_bat * act_bat).sum("inv_step")
    ann_gen_y = (ann_gen * act_gen).sum("inv_step")

    fuel_price = p.fuel_cost_per_unit_fuel if p.fuel_cost_per_unit_fuel is not None else p.fuel_fuel_cost_per_unit_fuel
    fuel_price = require_data_array("fuel_cost_per_unit_fuel", fuel_price)
    opex_y_s = (fuel_cons * fuel_price).sum("period")
    if p.is_grid_on() and isinstance(gimp, xr.DataArray) and p.grid_import_price is not None:
        opex_y_s = opex_y_s + (gimp * p.grid_import_price).sum("period")
    if p.is_grid_export_enabled() and isinstance(gexp, xr.DataArray) and p.grid_export_price is not None:
        opex_y_s = opex_y_s - (gexp * p.grid_export_price).sum("period")
    if p.res_production_subsidy_per_kwh is not None:
        subsidy = p.res_production_subsidy_per_kwh
        if "inv_step" in subsidy.dims:
            subsidy = subsidy.isel(inv_step=0, drop=True)
        opex_y_s = opex_y_s - (res_gen * subsidy).sum("period").sum("resource")

    ext_y_s = xr.DataArray(0.0).broadcast_like(opex_y_s)
    if p.lost_load_cost_per_kwh is not None:
        ext_y_s = ext_y_s + lost_load.sum("period") * p.lost_load_cost_per_kwh
    if p.fuel_direct_emissions_kgco2e_per_unit_fuel is not None and p.emission_cost_per_kgco2e is not None:
        ext_y_s = ext_y_s + fuel_cons.sum("period") * p.fuel_direct_emissions_kgco2e_per_unit_fuel * p.emission_cost_per_kgco2e

    commission_res = replacement_commission_mask(sets, res_life)
    commission_bat = replacement_commission_mask(sets, bat_life)
    commission_gen = replacement_commission_mask(sets, gen_life)
    emb_y = xr.DataArray(0.0).broadcast_like(ann_res_y)
    em_cost_exp = 0.0
    if p.emission_cost_per_kgco2e is not None:
        em_cost_exp = (p.emission_cost_per_kgco2e * w).sum("scenario") if "scenario" in p.emission_cost_per_kgco2e.dims else p.emission_cost_per_kgco2e
    if p.res_embedded_emissions_kgco2e_per_kw is not None:
        emb_y = emb_y + (res_units * res_nom * p.res_embedded_emissions_kgco2e_per_kw * commission_res).sum("inv_step").sum("resource") * em_cost_exp
    if p.battery_embedded_emissions_kgco2e_per_kwh is not None:
        emb_y = emb_y + (bat_units * bat_nom * p.battery_embedded_emissions_kgco2e_per_kwh * commission_bat).sum("inv_step") * em_cost_exp
    if p.generator_embedded_emissions_kgco2e_per_kw is not None:
        emb_y = emb_y + (gen_units * gen_nom * p.generator_embedded_emissions_kgco2e_per_kw * commission_gen).sum("inv_step") * em_cost_exp

    opex_exp_y = (opex_y_s * w).sum("scenario")
    ext_exp_y = (ext_y_s * w).sum("scenario")
    gross_y = ann_res_y + ann_bat_y + ann_gen_y + opex_exp_y + ext_exp_y + emb_y
    discounted_y = gross_y * disc

    salvage_tail_memo = (
        discounted_annuity_tail_memo(sets, ann_res, res_life, rs).sum("inv_step").sum("resource")
        + discounted_annuity_tail_memo(sets, ann_bat, bat_life, rs).sum("inv_step")
        + discounted_annuity_tail_memo(sets, ann_gen, gen_life, rs).sum("inv_step")
    )

    rows = []
    years = sets.coords["year"].values.tolist()
    last_year = years[-1]
    for y in years:
        salvage = safe_float(salvage_tail_memo) if y == last_year else 0.0
        row = {
            "year": y,
            "discount_factor": float(disc.sel(year=y)),
            "annuity_res": float(ann_res_y.sel(year=y)),
            "annuity_battery": float(ann_bat_y.sel(year=y)),
            "annuity_generator": float(ann_gen_y.sel(year=y)),
            "opex_expected": float(opex_exp_y.sel(year=y)),
            "externalities_expected": float(ext_exp_y.sel(year=y)),
            "embedded_expected": float(emb_y.sel(year=y)),
            "total_before_discount": float(gross_y.sel(year=y)),
            "discounted_total": float(discounted_y.sel(year=y)),
            "salvage_tail_memo_discounted": salvage,
            "discounted_objective_contribution": float(discounted_y.sel(year=y)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def export_multi_year_results(
    project_name: str,
    sets: xr.Dataset,
    data: xr.Dataset,
    model: Optional[lp.Model],
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
    out_dir: Path | None = None,
) -> dict:
    if out_dir is None:
        out_dir = ensure_results_dir(project_name)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    obj = None
    if model is not None and hasattr(model, "objective"):
        obj = safe_float(getattr(model.objective, "value", None))

    dispatch = build_dispatch_timeseries_table_multi_year(data=data, vars=vars, solution=solution)
    balance = build_energy_balance_table_multi_year(dispatch)
    design = build_design_by_step_table_multi_year(sets=sets, data=data, vars=vars, solution=solution)
    kpis = build_yearly_kpis_table_multi_year(data=data, vars=vars, solution=solution, objective_value=obj)
    cash = build_discounted_cashflows_table_multi_year(sets=sets, data=data, vars=vars, solution=solution)

    return write_csv_outputs(
        out_dir,
        {
            "dispatch_timeseries.csv": dispatch,
            "energy_balance.csv": balance,
            "design_by_step.csv": design,
            "kpis_yearly.csv": kpis,
            "cashflows_discounted.csv": cash,
        },
    )
