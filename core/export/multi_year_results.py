from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import linopy as lp

from core.io.utils import project_paths
from core.multi_year_model.params import get_params


class InputValidationError(RuntimeError):
    pass


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")


def _get_var_solution(
    *,
    vars: Optional[Dict[str, Any]],
    solution: Optional[xr.Dataset],
    name: str,
) -> Optional[xr.DataArray]:
    if isinstance(solution, xr.Dataset) and name in solution:
        da = solution[name]
        if isinstance(da, xr.DataArray):
            return da
    if isinstance(vars, dict):
        v = vars.get(name, None)
        try:
            if v is not None and hasattr(v, "solution") and isinstance(v.solution, xr.DataArray):
                return v.solution
        except Exception:
            pass
    return None


def _require_da(name: str, da: Optional[xr.DataArray]) -> xr.DataArray:
    if not isinstance(da, xr.DataArray):
        raise InputValidationError(f"Missing solved variable '{name}' in vars/solution.")
    return da


def _scenario_weights(p: Any, scenario_coord: xr.DataArray) -> xr.DataArray:
    if isinstance(p.scenario_weight, xr.DataArray):
        return p.scenario_weight.sel(scenario=scenario_coord)
    n = int(scenario_coord.size)
    return xr.DataArray(
        np.ones((n,), dtype=float) / float(n),
        dims=("scenario",),
        coords={"scenario": scenario_coord},
    )


def _crf(r: xr.DataArray | float, n: xr.DataArray | float) -> xr.DataArray:
    rr = xr.DataArray(r)
    nn = xr.DataArray(n)
    a = (1.0 + rr) ** nn
    out = (rr * a) / (a - 1.0)
    out = xr.where(np.abs(rr) < 1e-12, 1.0 / nn, out)
    return xr.where(nn > 0, out, 0.0)


def _year_ordinal(sets: xr.Dataset) -> xr.DataArray:
    year = sets.coords["year"]
    return xr.DataArray(
        np.arange(1, int(year.size) + 1, dtype=float),
        dims=("year",),
        coords={"year": year},
    )


def _inv_start_ord(sets: xr.Dataset) -> xr.DataArray:
    year_vals = [str(v) for v in sets.coords["year"].values.tolist()]
    start_vals = [str(v) for v in sets["inv_step_start_year"].values.tolist()]
    idx = {y: i + 1 for i, y in enumerate(year_vals)}
    return xr.DataArray(
        [float(idx[v]) for v in start_vals],
        dims=("inv_step",),
        coords={"inv_step": sets.coords["inv_step"]},
    )


def _svc_years(sets: xr.Dataset) -> xr.DataArray:
    return (_year_ordinal(sets) - _inv_start_ord(sets) + 1.0).transpose("inv_step", "year")


def _active_mask(sets: xr.Dataset, lt: xr.DataArray | float) -> xr.DataArray:
    svc = _svc_years(sets)
    return ((svc >= 1.0) & (svc <= xr.DataArray(lt))).astype(float)


def _commission_mask(sets: xr.Dataset) -> xr.DataArray:
    svc = _svc_years(sets)
    return (svc == 1.0).astype(float)


def _salvage_factor(sets: xr.Dataset, rate: xr.DataArray | float, lt: xr.DataArray | float) -> xr.DataArray:
    r = xr.DataArray(rate)
    n = xr.DataArray(lt)
    H = float(int(sets.coords["year"].size))
    used = (H - _inv_start_ord(sets) + 1.0).clip(min=0.0)
    rem = n - used
    a_n = (1.0 + r) ** n
    a_used = (1.0 + r) ** used
    frac = (a_n - a_used) / (a_n - 1.0)
    frac0 = xr.where(n > 0.0, rem / n, 0.0)
    frac = xr.where(np.abs(r) < 1e-12, frac0, frac)
    frac = xr.where(rem > 0.0, frac, 0.0)
    return xr.where(used > 0.0, frac, 0.0)


def build_dispatch_timeseries_table_multi_year(
    *,
    data: xr.Dataset,
    vars: Dict[str, Any],
    solution: Optional[xr.Dataset],
) -> pd.DataFrame:
    p = get_params(data)
    load = _require_da("load_demand", p.load_demand)
    res = _require_da("res_generation", _get_var_solution(vars=vars, solution=solution, name="res_generation"))
    gen = _require_da("generator_generation", _get_var_solution(vars=vars, solution=solution, name="generator_generation"))
    bch = _require_da("battery_charge", _get_var_solution(vars=vars, solution=solution, name="battery_charge"))
    bdis = _require_da("battery_discharge", _get_var_solution(vars=vars, solution=solution, name="battery_discharge"))
    bsoc = _require_da("battery_soc", _get_var_solution(vars=vars, solution=solution, name="battery_soc"))
    ll = _require_da("lost_load", _get_var_solution(vars=vars, solution=solution, name="lost_load"))

    gimp = _get_var_solution(vars=vars, solution=solution, name="grid_import") if p.is_grid_on() else None
    gexp = _get_var_solution(vars=vars, solution=solution, name="grid_export") if p.is_grid_export_enabled() else None

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
    res_units = _require_da("res_units", _get_var_solution(vars=vars, solution=solution, name="res_units"))
    bat_units = _require_da("battery_units", _get_var_solution(vars=vars, solution=solution, name="battery_units"))
    gen_units = _require_da("generator_units", _get_var_solution(vars=vars, solution=solution, name="generator_units"))
    res_nom = _require_da("res_nominal_capacity_kw", p.res_nominal_capacity_kw)
    bat_nom = _require_da("battery_nominal_capacity_kwh", p.battery_nominal_capacity_kwh)
    gen_nom = _require_da("generator_nominal_capacity_kw", p.generator_nominal_capacity_kw)

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
                    "installed_capacity": u * float(res_nom.sel(resource=r)),
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
                "installed_capacity": bu * float(bat_nom),
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
                "installed_capacity": gu * float(gen_nom),
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
    fuel = _get_var_solution(vars=vars, solution=solution, name="fuel_consumption")
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
            em = fuel_y * float(p.fuel_direct_emissions_kgco2e_per_unit_fuel.sel(year=year, scenario=scenario))
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
                "objective_value": _safe_float(objective_value),
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
                row[c] = _safe_float(objective_value)
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
    disc = 1.0 / ((1.0 + rs) ** _year_ordinal(sets))

    res_units = _require_da("res_units", _get_var_solution(vars=vars, solution=solution, name="res_units"))
    bat_units = _require_da("battery_units", _get_var_solution(vars=vars, solution=solution, name="battery_units"))
    gen_units = _require_da("generator_units", _get_var_solution(vars=vars, solution=solution, name="generator_units"))
    res_gen = _require_da("res_generation", _get_var_solution(vars=vars, solution=solution, name="res_generation"))
    fuel_cons = _require_da("fuel_consumption", _get_var_solution(vars=vars, solution=solution, name="fuel_consumption"))
    lost_load = _require_da("lost_load", _get_var_solution(vars=vars, solution=solution, name="lost_load"))
    gimp = _get_var_solution(vars=vars, solution=solution, name="grid_import")
    gexp = _get_var_solution(vars=vars, solution=solution, name="grid_export")

    res_nom = _require_da("res_nominal_capacity_kw", p.res_nominal_capacity_kw)
    res_capex = _require_da("res_specific_investment_cost_per_kw", p.res_specific_investment_cost_per_kw)
    res_life = _require_da("res_lifetime_years", p.res_lifetime_years)
    res_wacc = _require_da("res_wacc", p.res_wacc)
    res_grant = _require_da("res_grant_share_of_capex", p.res_grant_share_of_capex)
    bat_nom = _require_da("battery_nominal_capacity_kwh", p.battery_nominal_capacity_kwh)
    bat_capex = _require_da("battery_specific_investment_cost_per_kwh", p.battery_specific_investment_cost_per_kwh)
    bat_life = _require_da("battery_calendar_lifetime_years", p.battery_calendar_lifetime_years)
    bat_wacc = _require_da("battery_wacc", p.battery_wacc)
    gen_nom = _require_da("generator_nominal_capacity_kw", p.generator_nominal_capacity_kw)
    gen_capex = _require_da("generator_specific_investment_cost_per_kw", p.generator_specific_investment_cost_per_kw)
    gen_life = _require_da("generator_lifetime_years", p.generator_lifetime_years)
    gen_wacc = _require_da("generator_wacc", p.generator_wacc)

    res_inv = res_units * res_nom * res_capex * (1.0 - res_grant)
    bat_inv = bat_units * bat_nom * bat_capex
    gen_inv = gen_units * gen_nom * gen_capex
    ann_res = res_inv * _crf(res_wacc, res_life)
    ann_bat = bat_inv * _crf(bat_wacc, bat_life)
    ann_gen = gen_inv * _crf(gen_wacc, gen_life)
    act_res = _active_mask(sets, res_life)
    act_bat = _active_mask(sets, bat_life)
    act_gen = _active_mask(sets, gen_life)
    ann_res_y = (ann_res * act_res).sum("inv_step").sum("resource")
    ann_bat_y = (ann_bat * act_bat).sum("inv_step")
    ann_gen_y = (ann_gen * act_gen).sum("inv_step")

    fuel_price = p.fuel_cost_per_unit_fuel if p.fuel_cost_per_unit_fuel is not None else p.fuel_fuel_cost_per_unit_fuel
    fuel_price = _require_da("fuel_cost_per_unit_fuel", fuel_price)
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

    commission = _commission_mask(sets)
    emb_y = xr.DataArray(0.0).broadcast_like(ann_res_y)
    em_cost_exp = 0.0
    if p.emission_cost_per_kgco2e is not None:
        em_cost_exp = (p.emission_cost_per_kgco2e * w).sum("scenario") if "scenario" in p.emission_cost_per_kgco2e.dims else p.emission_cost_per_kgco2e
    if p.res_embedded_emissions_kgco2e_per_kw is not None:
        emb_y = emb_y + (res_units * res_nom * p.res_embedded_emissions_kgco2e_per_kw * commission).sum("inv_step").sum("resource") * em_cost_exp
    if p.battery_embedded_emissions_kgco2e_per_kwh is not None:
        emb_y = emb_y + (bat_units * bat_nom * p.battery_embedded_emissions_kgco2e_per_kwh * commission).sum("inv_step") * em_cost_exp
    if p.generator_embedded_emissions_kgco2e_per_kw is not None:
        emb_y = emb_y + (gen_units * gen_nom * p.generator_embedded_emissions_kgco2e_per_kw * commission).sum("inv_step") * em_cost_exp

    opex_exp_y = (opex_y_s * w).sum("scenario")
    ext_exp_y = (ext_y_s * w).sum("scenario")
    gross_y = ann_res_y + ann_bat_y + ann_gen_y + opex_exp_y + ext_exp_y + emb_y
    discounted_y = gross_y * disc

    sv = (
        (res_inv * _salvage_factor(sets, res_wacc, res_life)).sum("inv_step").sum("resource")
        + (bat_inv * _salvage_factor(sets, bat_wacc, bat_life)).sum("inv_step")
        + (gen_inv * _salvage_factor(sets, gen_wacc, gen_life)).sum("inv_step")
    )
    H = float(int(sets.coords["year"].size))
    sv_disc = float(sv) / ((1.0 + rs) ** H)

    rows = []
    years = sets.coords["year"].values.tolist()
    last_year = years[-1]
    for y in years:
        salvage = sv_disc if y == last_year else 0.0
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
            "salvage_credit_discounted": salvage,
            "discounted_net_with_salvage": float(discounted_y.sel(year=y)) - salvage,
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
        out_dir = project_paths(project_name).results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = None
    if model is not None and hasattr(model, "objective"):
        obj = _safe_float(getattr(model.objective, "value", None))

    dispatch = build_dispatch_timeseries_table_multi_year(data=data, vars=vars, solution=solution)
    balance = build_energy_balance_table_multi_year(dispatch)
    design = build_design_by_step_table_multi_year(sets=sets, data=data, vars=vars, solution=solution)
    kpis = build_yearly_kpis_table_multi_year(data=data, vars=vars, solution=solution, objective_value=obj)
    cash = build_discounted_cashflows_table_multi_year(sets=sets, data=data, vars=vars, solution=solution)

    p_dispatch = out_dir / "dispatch_timeseries.csv"
    p_balance = out_dir / "energy_balance.csv"
    p_design = out_dir / "design_by_step.csv"
    p_kpis = out_dir / "kpis_yearly.csv"
    p_cash = out_dir / "cashflows_discounted.csv"

    dispatch.to_csv(p_dispatch, index=False)
    balance.to_csv(p_balance, index=False)
    design.to_csv(p_design, index=False)
    kpis.to_csv(p_kpis, index=False)
    cash.to_csv(p_cash, index=False)

    return {
        "dispatch_timeseries_csv": str(p_dispatch),
        "energy_balance_csv": str(p_balance),
        "design_by_step_csv": str(p_design),
        "kpis_yearly_csv": str(p_kpis),
        "cashflows_discounted_csv": str(p_cash),
        "out_dir": str(out_dir),
    }
