from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr
import linopy as lp

from core.multi_year_model.params import get_params


class InputValidationError(RuntimeError):
    pass


def _require_da(name: str, da: xr.DataArray | None) -> xr.DataArray:
    if da is None:
        raise InputValidationError(f"Missing required parameter/data variable: '{name}'")
    return da


def _factor_from_degradation(
    *,
    sets: xr.Dataset,
    degradation_rate: xr.DataArray | None,
) -> xr.DataArray:
    inv_step = sets.coords["inv_step"]
    year = sets.coords["year"]
    step_start = sets["inv_step_start_year"]

    age = (year - step_start).clip(min=0).transpose("inv_step", "year").astype(float)
    factor = xr.ones_like(age, dtype=float)

    if degradation_rate is None:
        return factor

    rate = degradation_rate.clip(min=0.0)
    factor = (1.0 - rate) ** age
    return factor


def _available_capacity_by_year(
    *,
    sets: xr.Dataset,
    units: lp.Variable,
    nominal_capacity: xr.DataArray,
    degradation_rate: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Generic linear capacity availability from incremental investments.

    Returns capacity with dims that include `year` and any non-investment dims
    from `units/nominal_capacity/degradation_rate` (e.g. scenario, resource).
    """
    inv_active = sets["inv_active_in_year"]  # (inv_step, year)
    invest_cap = units * nominal_capacity
    factor = _factor_from_degradation(sets=sets, degradation_rate=degradation_rate)
    available = (invest_cap * inv_active * factor).sum("inv_step")
    return available


def _is_effectively_positive(da: xr.DataArray | None) -> bool:
    if da is None:
        return False
    vals = np.asarray(da.values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    return float(vals.max()) > 0.0


def validate_constraint_shapes(
    *,
    sets: xr.Dataset,
    p: object,
    vars: Dict[str, lp.Variable],
) -> None:
    required_sets = ("period", "year", "inv_step", "scenario", "resource")
    for c in required_sets:
        if c not in sets.coords:
            raise InputValidationError(f"initialize_constraints: missing required coord in sets: '{c}'")

    required_vars = (
        "res_units",
        "battery_units",
        "generator_units",
        "res_generation",
        "generator_generation",
        "fuel_consumption",
        "battery_charge",
        "battery_discharge",
        "battery_soc",
        "lost_load",
    )
    for v in required_vars:
        if v not in vars:
            raise InputValidationError(f"initialize_constraints: missing required variable '{v}'")

    required_data = (
        ("load_demand", p.load_demand, {"period", "year", "scenario"}),
        ("resource_availability", p.resource_availability, {"period", "year", "scenario", "resource"}),
        ("scenario_weight", p.scenario_weight, {"scenario"}),
        ("min_renewable_penetration", p.min_renewable_penetration, set()),
        ("max_lost_load_fraction", p.max_lost_load_fraction, set()),
    )
    for name, da, expected_any in required_data:
        da_req = _require_da(name, da)
        if expected_any and not expected_any.issubset(set(da_req.dims)):
            raise InputValidationError(
                f"Parameter '{name}' has dims {da_req.dims}, expected at least {sorted(expected_any)}"
            )


def initialize_constraints(
    sets: xr.Dataset,
    data: xr.Dataset,
    vars: Dict[str, lp.Variable],
    model: lp.Model,
) -> None:
    if not isinstance(sets, xr.Dataset):
        raise InputValidationError("initialize_constraints: `sets` must be an xarray.Dataset.")
    if not isinstance(data, xr.Dataset):
        raise InputValidationError("initialize_constraints: `data` must be an xarray.Dataset.")
    if not isinstance(vars, dict):
        raise InputValidationError("initialize_constraints: `vars` must be a dict of linopy variables.")

    period = sets.coords["period"]

    p = get_params(data)
    validate_constraint_shapes(sets=sets, p=p, vars=vars)

    on_grid = p.is_grid_on()
    allow_export = p.is_grid_export_enabled()
    enforcement = str((p.settings.get("optimization_constraints", {}) or {}).get("enforcement", "scenario_wise"))
    if enforcement not in ("expected", "scenario_wise"):
        raise InputValidationError(
            f"Invalid optimization_constraints.enforcement='{enforcement}'. "
            "Allowed: 'expected' | 'scenario_wise'."
        )

    load_demand = _require_da("load_demand", p.load_demand)
    resource_availability = _require_da("resource_availability", p.resource_availability)
    scenario_weight = _require_da("scenario_weight", p.scenario_weight)

    min_res_pen = _require_da("min_renewable_penetration", p.min_renewable_penetration)
    max_ll_frac = _require_da("max_lost_load_fraction", p.max_lost_load_fraction)

    res_nom_kw = _require_da("res_nominal_capacity_kw", p.res_nominal_capacity_kw)
    res_inv_eta = _require_da("res_inverter_efficiency", p.res_inverter_efficiency)
    res_max_kw = _require_da("res_max_installable_capacity_kw", p.res_max_installable_capacity_kw)

    bat_nom_kwh = _require_da("battery_nominal_capacity_kwh", p.battery_nominal_capacity_kwh)
    eta_c = _require_da("battery_charge_efficiency", p.battery_charge_efficiency)
    eta_d = _require_da("battery_discharge_efficiency", p.battery_discharge_efficiency)
    soc0 = _require_da("battery_initial_soc", p.battery_initial_soc)
    dod = _require_da("battery_depth_of_discharge", p.battery_depth_of_discharge)
    t_ch = _require_da("battery_max_charge_time_hours", p.battery_max_charge_time_hours)
    t_dis = _require_da("battery_max_discharge_time_hours", p.battery_max_discharge_time_hours)

    gen_nom_kw = _require_da("generator_nominal_capacity_kw", p.generator_nominal_capacity_kw)
    gen_max_kw = _require_da("generator_max_installable_capacity_kw", p.generator_max_installable_capacity_kw)
    gen_eta_full = _require_da("generator_nominal_efficiency_full_load", p.generator_nominal_efficiency_full_load)
    fuel_lhv = _require_da("fuel_lhv_kwh_per_unit_fuel", p.fuel_lhv_kwh_per_unit_fuel)

    land_m2 = _require_da("land_availability_m2", p.land_availability_m2)
    res_area_m2_per_kw = _require_da("res_specific_area_m2_per_kw", p.res_specific_area_m2_per_kw)

    if on_grid:
        grid_line_cap = _require_da("grid_line_capacity_kw", p.grid_line_capacity_kw)
        grid_eta = _require_da("grid_transmission_efficiency", p.grid_transmission_efficiency)
        grid_availability = _require_da("grid_availability", p.grid_availability)

    res_units = vars["res_units"]  # (inv_step, resource)
    bat_units = vars["battery_units"]  # (inv_step,)
    gen_units = vars["generator_units"]  # (inv_step,)

    res_gen = vars["res_generation"]  # (period, year, scenario, resource)
    gen_gen = vars["generator_generation"]  # (period, year, scenario)
    fuel_cons = vars["fuel_consumption"]  # (period, year, scenario)
    bat_ch = vars["battery_charge"]  # (period, year, scenario)
    bat_dis = vars["battery_discharge"]  # (period, year, scenario)
    soc = vars["battery_soc"]  # (period, year, scenario)
    ll = vars["lost_load"]  # (period, year, scenario)
    grid_imp = vars.get("grid_import")
    grid_exp = vars.get("grid_export")
    gen_seg = vars.get("gen_segment_energy")

    # ------------------------------------------------------------------
    # 1) Renewable generation capacity with year availability
    # ------------------------------------------------------------------
    res_cap_available = _available_capacity_by_year(
        sets=sets,
        units=res_units,
        nominal_capacity=res_nom_kw,
        degradation_rate=p.res_capacity_degradation_rate_per_year,
    ) * res_inv_eta
    model.add_constraints(
        res_gen <= (resource_availability * res_cap_available),
        name="res_generation_cap",
    )

    finite_res_max = np.isfinite(res_max_kw)
    res_max_kw_finite = res_max_kw.where(finite_res_max, drop=True)
    if res_max_kw_finite.sizes.get("resource", 0) > 0:
        lhs_res_total = (res_units * res_nom_kw).sum("inv_step").sel(resource=res_max_kw_finite.resource)
        model.add_constraints(lhs_res_total <= res_max_kw_finite, name="res_max_installable_capacity")

    # ------------------------------------------------------------------
    # 2) Generator capacity with year availability
    # ------------------------------------------------------------------
    gen_cap_available = _available_capacity_by_year(
        sets=sets,
        units=gen_units,
        nominal_capacity=gen_nom_kw,
        degradation_rate=p.generator_capacity_degradation_rate_per_year,
    )
    model.add_constraints(gen_gen <= gen_cap_available, name="generator_generation_cap")

    finite_gen_max = np.isfinite(gen_max_kw)
    gen_max_finite = gen_max_kw.where(finite_gen_max, drop=True)
    if gen_max_finite.size > 0:
        model.add_constraints((gen_units * gen_nom_kw).sum("inv_step") <= gen_max_finite.max(), name="generator_max_capacity")

    # ------------------------------------------------------------------
    # 3) Fuel-to-power relation
    # ------------------------------------------------------------------
    if gen_seg is not None and p.generator_eff_curve_eff is not None and p.generator_eff_curve_rel_power is not None:
        model.add_constraints(gen_seg.sum("curve_point") == gen_gen, name="generator_segment_link")

        eff_curve = p.generator_eff_curve_eff.fillna(gen_eta_full)
        denom = fuel_lhv * eff_curve
        model.add_constraints(
            fuel_cons >= (gen_seg / denom).sum("curve_point"),
            name="fuel_to_power_partial_load_lb",
        )
    else:
        model.add_constraints(
            gen_gen == fuel_cons * fuel_lhv * gen_eta_full,
            name="fuel_to_power_nominal_eta",
        )

    # ------------------------------------------------------------------
    # 4) Battery limits and SOC dynamics
    # ------------------------------------------------------------------
    bat_cap_available = _available_capacity_by_year(
        sets=sets,
        units=bat_units,
        nominal_capacity=bat_nom_kwh,
        degradation_rate=p.battery_capacity_degradation_rate_per_year,
    )

    model.add_constraints(bat_ch <= (bat_cap_available / t_ch), name="battery_charge_limit")
    model.add_constraints(bat_dis <= (bat_cap_available / t_dis), name="battery_discharge_limit")

    T = int(period.size)
    model.add_constraints(soc.isel(period=0) == soc0 * bat_cap_available, name="soc_initial")

    if T > 1:
        model.add_constraints(
            soc.isel(period=slice(1, None))
            == soc.isel(period=slice(0, -1))
            + eta_c * bat_ch.isel(period=slice(0, -1))
            - bat_dis.isel(period=slice(0, -1)) / eta_d,
            name="soc_balance",
        )

    model.add_constraints(
        soc.isel(period=T - 1)
        + eta_c * bat_ch.isel(period=T - 1)
        - bat_dis.isel(period=T - 1) / eta_d
        == soc0 * bat_cap_available,
        name="soc_cyclic",
    )
    model.add_constraints(soc <= bat_cap_available, name="soc_upper")
    model.add_constraints(soc >= (1.0 - dod) * bat_cap_available, name="soc_lower")

    # ------------------------------------------------------------------
    # 5) Grid limits and nodal balance
    # ------------------------------------------------------------------
    if on_grid and grid_imp is not None:
        model.add_constraints(
            grid_imp <= (grid_availability * grid_line_cap),
            name="grid_import_cap",
        )
        if allow_export and grid_exp is not None:
            model.add_constraints(
                grid_exp <= (grid_availability * grid_line_cap),
                name="grid_export_cap",
            )

    res_sum = res_gen.sum("resource")
    lhs = res_sum + gen_gen + (bat_dis - bat_ch) + ll

    if on_grid and grid_imp is not None:
        lhs = lhs + (grid_imp * grid_eta)
        if allow_export and grid_exp is not None:
            lhs = lhs - (grid_exp * grid_eta)

    model.add_constraints(lhs == load_demand, name="energy_balance")

    # ------------------------------------------------------------------
    # 6) Policy constraints (year/scenario or expected-by-year)
    # ------------------------------------------------------------------
    e_demand = load_demand.sum("period")  # (year, scenario)
    e_ll = ll.sum("period")  # (year, scenario)
    e_res = res_sum.sum("period")  # (year, scenario)
    e_gen = gen_gen.sum("period")  # (year, scenario)

    if on_grid and grid_imp is not None:
        e_grid = (grid_imp * grid_eta).sum("period")
    else:
        # e_res is a linopy expression, not an xarray object; keep zero as scalar.
        e_grid = 0.0

    if enforcement == "scenario_wise":
        model.add_constraints(e_ll <= (max_ll_frac * e_demand), name="max_lost_load_share")
    else:
        lhs_ll = (e_ll * scenario_weight).sum("scenario")
        if "scenario" in max_ll_frac.dims:
            rhs_ll = (max_ll_frac * e_demand * scenario_weight).sum("scenario")
        else:
            rhs_ll = max_ll_frac * (e_demand * scenario_weight).sum("scenario")
        model.add_constraints(lhs_ll <= rhs_ll, name="max_lost_load_share_expected")

    if _is_effectively_positive(min_res_pen):
        e_total = e_res + e_gen + e_grid
        if enforcement == "scenario_wise":
            model.add_constraints(e_res >= (min_res_pen * e_total), name="min_renewable_penetration")
        else:
            lhs_res = (e_res * scenario_weight).sum("scenario")
            if "scenario" in min_res_pen.dims:
                rhs_res = (min_res_pen * e_total * scenario_weight).sum("scenario")
            else:
                rhs_res = min_res_pen * (e_total * scenario_weight).sum("scenario")
            model.add_constraints(lhs_res >= rhs_res, name="min_renewable_penetration_expected")

    # ------------------------------------------------------------------
    # 7) Land use (design-side, scenario-independent investments)
    # ------------------------------------------------------------------
    area_used = res_units * res_nom_kw * res_area_m2_per_kw
    if "inv_step" in area_used.dims:
        area_used = area_used.sum("inv_step")
    if "resource" in area_used.dims:
        area_used = area_used.sum("resource")
    model.add_constraints(area_used <= land_m2, name="land_availability")
