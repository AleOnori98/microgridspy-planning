# generation_planning/modeling/constraints.py
from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr
import linopy as lp

from core.multi_year_model import model


class InputValidationError(RuntimeError):
    pass


def _bool_from_attrs(obj: xr.Dataset, path: list[str], default: bool = False) -> bool:
    cur = obj.attrs
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return bool(cur)


def _str_from_attrs(obj: xr.Dataset, path: list[str], default: str = "") -> str:
    cur = obj.attrs
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur)


def initialize_constraints(
    sets: xr.Dataset,
    data: xr.Dataset,
    vars: Dict[str, lp.Variable],
    model: lp.Model,
) -> None:
    """
    Add steady_state (typical-year) constraints to the linopy model.

    Conventions (consistent with your current implementation):
      - period = 0..8759
      - scenario optional but present always (scenario_1 if disabled)
      - renewables availability in data["resource_availability"] with dims (period, scenario, resource)
      - "units" design variables scale nominal capacities from YAML templates
    """
    if not isinstance(sets, xr.Dataset):
        raise InputValidationError("initialize_constraints: `sets` must be an xarray.Dataset.")
    if not isinstance(data, xr.Dataset):
        raise InputValidationError("initialize_constraints: `data` must be an xarray.Dataset.")
    if not isinstance(vars, dict):
        raise InputValidationError("initialize_constraints: `vars` must be a dict of linopy variables.")

    # ---------------------------------------------------------------------
    # Coords
    # ---------------------------------------------------------------------
    for c in ("period", "year", "inv_step", "scenario", "resource"):
        if c not in sets.coords:
            raise InputValidationError(f"initialize_constraints: missing required coord in sets: '{c}'")

    period = sets.coords["period"]
    year = sets.coords["year"]
    inv_step = sets.coords["inv_step"]
    scenario = sets.coords["scenario"]

    # ---------------------------------------------------------------------
    # Flags and enforcement mode (from attrs)
    # ---------------------------------------------------------------------
    on_grid = _bool_from_attrs(data, ["settings", "grid", "on_grid"], default=False)
    allow_export = _bool_from_attrs(data, ["settings", "grid", "allow_export"], default=False)
    enforcement = _str_from_attrs(
        data,
        ["settings", "optimization_constraints", "enforcement"],
        default="scenario_wise",
    )
    if enforcement not in ("expected", "scenario_wise"):
        raise InputValidationError(
            f"Invalid optimization_constraints.enforcement='{enforcement}'. "
            "Allowed: 'expected' | 'scenario_wise'."
        )

    # ---------------------------------------------------------------------
    # Parameters (data vars)
    # ---------------------------------------------------------------------
    # Time series
    load_demand = data["load_demand"]  # (period, year, scenario)
    resource_availability = data["resource_availability"]  # (period, year, scenario, resource)

    # Scenario weights (always present)
    w_s = data["scenario_weight"]  # (scenario,)

    # -----------------------------
    # Renewables
    # -----------------------------
    # Technical (step-invariant)
    res_nom_kw = data["res_nominal_capacity_kw"]          # (resource,)
    res_inv_eta = data["res_inverter_efficiency"]         # (resource,)

    # Constraints / siting (step-invariant)
    land_m2 = data["land_availability_m2"]                        # scalar
    res_area_m2_per_kw = data["res_specific_area_m2_per_kw"]      # (resource,)
    res_max_kw = data["res_max_installable_capacity_kw"]          # (resource,) may contain NaN

    # Dynamic-only degradation (scenario-dependent only)
    res_deg_capacity = data["res_capacity_degradation_rate_per_year"] # (scenario, resource)

    # -----------------------------
    # Battery
    # -----------------------------
    # Investment-side (step-dependent, scenario-invariant)
    bat_nom_kwh = data["battery_nominal_capacity_kwh"]                 # (inv_step,)
    bat_capex_kwh = data["battery_specific_investment_cost_per_kwh"]   # (inv_step,)
    bat_wacc = data["battery_wacc"]                                     # (inv_step,)
    bat_cal_life = data["battery_calendar_lifetime_years"]             # (inv_step,)
    bat_max_kwh = data["battery_max_installable_capacity_kwh"]         # (inv_step,) may contain NaN

    # Technical (step-invariant scalars) 
    bat_eta_c = data["battery_charge_efficiency"]          # ()  
    bat_eta_d = data["battery_discharge_efficiency"]       # ()
    bat_soc0 = data["battery_initial_soc"]                 # ()
    bat_dod = data["battery_depth_of_discharge"]           # ()
    bat_t_ch = data["battery_max_charge_time_hours"]       # ()
    bat_t_dis = data["battery_max_discharge_time_hours"]   # ()

    # Dynamic-only degradation (scenario-dependent, optional)
    bat_deg_capacity = data["battery_capacity_degradation_rate_per_year"] # (scenario,) default 0.0 if missing

    # -----------------------------
    # Generator
    # -----------------------------
    # Investment-side (step-dependent)
    gen_nom_kw = data["generator_nominal_capacity_kw"]                 # (inv_step,)
    gen_max_kw = data["generator_max_installable_capacity_kw"]         # (inv_step,)

    # Technical (step-invariant scalar)
    gen_eta_full = data["generator_nominal_efficiency_full_load"]      # ()

    # Operation (scenario-dependent)
    gen_deg_capacity = data.get("generator_capacity_degradation_rate_per_year", 0.0) # (scenario,) if present
    fuel_lhv = data["fuel_lhv_kwh_per_unit_fuel"]                                    # (scenario,)

    # -----------------------------
    # System constraints params
    # -----------------------------
    min_res_pen = data["min_renewable_penetration"]   # scalar or (scenario,)
    max_ll_frac = data["max_lost_load_fraction"]      # scalar or (scenario,)

    # -----------------------------
    # Grid params (if on-grid)
    # -----------------------------
    if on_grid:
        line_cap_kw = data["grid_line_capacity_kw"]            # (scenario,) 
        grid_eta = data["grid_transmission_efficiency"]        # (scenario,) 
        grid_avail = data["grid_availability"]                 # (period, year, scenario)
        allow_export = bool(data.attrs.get("settings", {}).get("grid", {}).get("allow_export", False))

    # ---------------------------------------------------------------------
    # Variables (aliases) 
    # ---------------------------------------------------------------------
    # Design / sizing (scenario-invariant)
    res_units = vars["res_units"]          # (inv_step, resource)
    bat_units = vars["battery_units"]      # (inv_step,)
    gen_units = vars["generator_units"]    # (inv_step,)

    # Operation
    res_gen = vars["res_generation"]          # (period, year, scenario, resource)
    gen_gen = vars["generator_generation"]    # (period, year, scenario)
    fuel_cons = vars["fuel_consumption"]      # (period, year, scenario)
    bat_ch = vars["battery_charge"]           # (period, year, scenario)
    bat_dis = vars["battery_discharge"]       # (period, year, scenario)
    soc = vars["battery_soc"]                 # (period, year, scenario)
    ll = vars["lost_load"]                    # (period, year, scenario)

    grid_imp = vars.get("grid_import", None)  # (period, year, scenario) if on_grid
    grid_exp = vars.get("grid_export", None)  # (period, year, scenario) if allow_export

    # Optional partial-load scaffolding
    gen_seg = vars.get("gen_segment_energy", None)  # (period, year, scenario, curve_point) if enabled

    # ---------------------------------------------------------------------
    # 1) Renewable generation limited by availability and installed capacity with degradation (energy + capacity), cohort-based
    # res_gen(t,y,s,r) <= avail(t,y,s,r) * sum_k [ active(k,y) * units(k,r) * nom_kw(r) * inv_eta(r)
    #                                             * (1 - cap_deg(s,r))^age(k,y)
    #                                             * (1 - energy_deg(s,r))^age(k,y) ]
    # ---------------------------------------------------------------------

    inv_active = sets["inv_active_in_year"]        # (inv_step, year) {0,1}
    step_start = sets["inv_step_start_year"]       # (inv_step,) year label (e.g. 2026)
    years = sets.coords["year"]                    # (year,) year labels (e.g. 2026..)

    # cohort age in each year: age(inv_step, year)
    # age = max(0, year - step_start)
    age = (years - step_start).clip(min=0)         # xarray broadcasting -> (inv_step, year)
    age = age.astype(float)

    # Broadcast age to include scenario/resource
    age = age.expand_dims({"scenario": sets.coords["scenario"], "resource": sets.coords["resource"]})

    # Broadcast rates to include inv_step/year
    cap_rate = res_deg_capacity.expand_dims({"inv_step": sets.coords["inv_step"], "year": years})

    # Degradation factors per cohort/year/scenario/resource
    cap_factor = (1.0 - cap_rate) ** age

    # Put dimensions in a convenient order for later broadcasting
    cap_factor = cap_factor.transpose("inv_step", "year", "scenario", "resource")

    # Convert units to kW per cohort
    res_kw_by_step = res_units * res_nom_kw       # (inv_step, resource)

    # Apply active mask (inv_step, year) to cohort capacity
    active = inv_active.expand_dims({"scenario": sets.coords["scenario"], "resource": sets.coords["resource"]})
    active = active.transpose("inv_step", "year", "scenario", "resource")

    # Expand res_kw_by_step to include year/scenario for broadcasting
    res_kw = res_kw_by_step.expand_dims({"year": years, "scenario": sets.coords["scenario"]})
    res_kw = res_kw.transpose("inv_step", "year", "scenario", "resource")

    # Effective available kW (after cohort degradation) before inverter efficiency
    res_kw_eff = (active * res_kw * cap_factor).sum("inv_step")     # (year, scenario, resource)

    # Apply inverter efficiency (resource,) -> broadcast to (year, scenario, resource)
    res_kw_eff = res_kw_eff * res_inv_eta

    rhs_res = resource_availability * res_kw_eff   # (period, year, scenario, resource)
    model.add_constraints(res_gen <= rhs_res, name="res_generation_cap")

    # ---------------------------------------------------------------------
    # 2) Renewable max installable capacity (global cap across horizon)
    # Constraint:
    #   sum_k res_units(k,r) * res_nom_kw(r) <= res_max_kw(r)   for finite res_max_kw
    # ---------------------------------------------------------------------
    finite = np.isfinite(res_max_kw)
    res_max_kw_finite = res_max_kw.where(finite, drop=True)

    if res_max_kw_finite.sizes.get("resource", 0) > 0:
        r = res_max_kw_finite.resource

        lhs = (res_units.sel(resource=r).sum("inv_step") * res_nom_kw.sel(resource=r))
        rhs = res_max_kw_finite

        model.add_constraints(lhs <= rhs, name="res_max_installable_capacity")

    # ---------------------------------------------------------------------
    # 3) Generator production capacity (multi-year)
    #
    # gen_gen(t,y,s) <= available_gen_kw(y,s)
    # where available_gen_kw(y,s) = [ sum_k inv_active_in_year(k,y) * gen_units(k) * gen_nom_kw(k) ] * (1 - cap_deg)^(age)
    # ---------------------------------------------------------------------

    inv_active = sets["inv_active_in_year"]  # (inv_step, year) 0/1

    # cohort age matrix (inv_step, year) in years
    # uses your sets metadata:
    step_start_year = sets["inv_step_start_year"]  # (inv_step,) year labels like 2026
    year_labels = sets.coords["year"]              # (year,) same type (ints)

    # age(k,y) = max(0, year - step_start_year[k])
    age = (year_labels - step_start_year).clip(min=0)  # broadcasts to (inv_step, year)

    # derating factor per (scenario, inv_step, year)
    # factor = (1 - rate) ** age
    derate = (1.0 - gen_deg_capacity).clip(min=0.0) ** age  # broadcasts to (scenario, inv_step, year)

    # degraded available kW per (scenario, year):
    gen_kw_in_year_degraded = (inv_active * gen_units * gen_nom_kw * derate).sum("inv_step")  # (scenario, year)

    # align to (year, scenario) for multiplication with gen_gen(period, year, scenario)
    gen_kw_rhs = gen_kw_in_year_degraded.transpose("year", "scenario")  # (year, scenario)

    # final constraint: broadcast over period
    model.add_constraints(gen_gen <= gen_kw_rhs, name="generator_generation_cap")

    # ---------------------------------------------------------------------
    # 4) Fuel ↔ energy relationship
    #   - If no partial-load curve: equality with nominal efficiency
    #   - If partial-load enabled: convex piecewise lower bound on fuel_cons
    # ---------------------------------------------------------------------

    pl_rel = data.get("generator_eff_curve_rel_power", None)
    pl_eff = data.get("generator_eff_curve_eff", None)

    pl_points_ok = (
        pl_rel is not None and pl_eff is not None
        and ("curve_point" in pl_rel.dims) and ("curve_point" in pl_eff.dims)
        and (pl_rel.sizes["curve_point"] >= 2)
    )

    # Helper: if a scenario row is all-NaN, treat as no-curve for that scenario
    def _scenario_has_curve(scen: str) -> bool:
        if not pl_points_ok:
            return False
        r = pl_rel.sel(scenario=scen)
        e = pl_eff.sel(scenario=scen)
        return (np.isfinite(r.values).any() and np.isfinite(e.values).any())

    scenario_labels = [str(s) for s in sets.coords["scenario"].values.tolist()]
    scenarios_with_pl = [s for s in scenario_labels if _scenario_has_curve(s)]
    scenarios_without_pl = [s for s in scenario_labels if s not in scenarios_with_pl]

    # --- Case A) scenarios without PL curve: nominal efficiency equality
    if len(scenarios_without_pl) > 0:
        model.add_constraints(
            gen_gen.sel(scenario=scenarios_without_pl)
            == fuel_cons.sel(scenario=scenarios_without_pl) * fuel_lhv.sel(scenario=scenarios_without_pl) * gen_eta_full,
            name="fuel_to_power_nominal_eta",
        )

    # --- Case B) scenarios with PL curve: convex piecewise lower bound
    if pl_points_ok and len(scenarios_with_pl) > 0:
        # segment coordinate: 0..P-2
        P = int(pl_rel.sizes["curve_point"])
        seg = xr.IndexVariable("segment", np.arange(P - 1))

        for s in scenarios_with_pl:
            # Scalar per scenario
            lhv_s = fuel_lhv.sel(scenario=s)  # scalar DA
            cap = gen_nom_kw                  # scalar DA (kW per unit)

            # Breakpoints for this scenario
            r_full = pl_rel.sel(scenario=s)   # (curve_point,)
            e_full = pl_eff.sel(scenario=s)   # (curve_point,)

            # Basic sanity (optional but recommended)
            # - drop NaNs (if present) by requiring all finite
            if not (np.isfinite(r_full.values).all() and np.isfinite(e_full.values).all()):
                raise ValueError(f"Partial-load curve for scenario '{s}' contains NaNs; provide full curve_point series.")

            # r0,r1 and eta0,eta1
            r0 = r_full.isel(curve_point=seg)         # (segment,)
            r1 = r_full.isel(curve_point=seg + 1)     # (segment,)
            eta0 = e_full.isel(curve_point=seg)       # (segment,)
            eta1 = e_full.isel(curve_point=seg + 1)   # (segment,)

            # Convert to power at breakpoints for ONE unit (kW)
            p0 = cap * r0                              # (segment,)
            p1 = cap * r1                              # (segment,)

            # Fuel at breakpoints for ONE unit (unit_fuel/h since Δt=1h)
            # fc = P / (eta * LHV)
            fc0 = p0 / (eta0 * lhv_s)                  # (segment,)
            fc1 = p1 / (eta1 * lhv_s)                  # (segment,)

            # Segment slope in fuel per kWh (unit_fuel/kWh)
            denom = (p1 - p0)
            slope = xr.where(denom != 0.0, (fc1 - fc0) / denom, 0.0)  # (segment,)

            # Variables for this scenario
            gen_ts = gen_gen.sel(scenario=s)           # (period,)
            fuel_ts = fuel_cons.sel(scenario=s)        # (period,)

            # Broadcast to (period, segment)
            gen_b = gen_ts.expand_dims({"segment": seg})
            fuel_b = fuel_ts.expand_dims({"segment": seg})

            # RHS: slope*(gen - p0*units) + fc0*units
            rhs = slope * (gen_b - p0 * gen_units) + fc0 * gen_units

            model.add_constraints(
                fuel_b >= rhs,
                name=f"fuel_to_power_partial_load_{s}",
            )


    # ---------------------------------------------------------------------
    # 5) Battery charge/discharge power limits
    #
    # battery_charge(t,s)    <= (battery_units * bat_nom_kwh) / t_ch
    # battery_discharge(t,s) <= (battery_units * bat_nom_kwh) / t_dis
    # ---------------------------------------------------------------------
    model.add_constraints(bat_ch <= (bat_units * bat_nom_kwh) / t_ch, name="battery_charge_limit")
    model.add_constraints(bat_dis <= (bat_units * bat_nom_kwh) / t_dis, name="battery_discharge_limit")

    # ---------------------------------------------------------------------
    # 6) Battery SOC dynamics (hourly Δt=1h) with cyclic end condition
    #
    # soc[0,s] = soc0(s)*Ecap   (initial condition)
    # soc[t,s] = soc[t-1,s] + eta_c(s)*bat_ch[t-1,s] - (1/eta_d(s))*bat_dis
    # soc[T-1,s] = soc0(s)*Ecap  (cyclic condition: end returns to initial)
    # where Ecap = battery_units * bat_nom_kwh
    # ---------------------------------------------------------------------
    T = int(period.size)
    Ecap = bat_units * bat_nom_kwh

    model.add_constraints(soc.isel(period=0) == soc0 * Ecap, name="soc_initial")

    if T > 1:
        model.add_constraints(
            soc.isel(period=slice(1, None))
            == soc.isel(period=slice(0, -1))
            + eta_c * bat_ch.isel(period=slice(0, -1))
            - bat_dis.isel(period=slice(0, -1)) / eta_d,
            name="soc_balance",
        )

    # cyclic closure (do NOT also force soc[T-1]==soc0*Ecap)
    model.add_constraints(
        soc.isel(period=T-1)
        + eta_c * bat_ch.isel(period=T-1)
        - bat_dis.isel(period=T-1) / eta_d
        == soc0 * Ecap,
        name="soc_cyclic",
    )

    model.add_constraints(soc <= Ecap, name="soc_upper")
    model.add_constraints(soc >= (1.0 - dod) * Ecap, name="soc_lower")

    # ---------------------------------------------------------------------
    # 7) Grid import/export limits (if enabled)
    # Constraints:
    #   grid_import(t,s) <= grid_availability(t,s) * line_cap_kw(s)
    #   grid_export(t,s) <= grid_availability(t,s) * line_cap_kw(s)
    # ---------------------------------------------------------------------
    if on_grid:
        # Import limited by availability and line capacity and efficiency
        rhs_imp = grid_avail * line_cap_kw
        model.add_constraints(grid_imp <= rhs_imp, name="grid_import_cap")

        if allow_export:
            rhs_exp = grid_avail * line_cap_kw
            model.add_constraints(grid_exp <= rhs_exp, name="grid_export_cap")

    # ---------------------------------------------------------------------
    # 8) Energy balance per (period, scenario)
    # Energy balance:
    #   Σ_r res_generation(t,s,r)
    # + generator_generation(t,s)
    # + grid_import(t,s) [if on_grid]
    # - grid_export(t,s) [if allow_export]
    # + battery_discharge(t,s)
    # - battery_charge(t,s)
    # + lost_load(t,s)
    # == load_demand(t,s)
    # ---------------------------------------------------------------------
    res_sum = res_gen.sum("resource")  # (period, scenario)

    if on_grid and allow_export:
        model.add_constraints(
            res_sum + gen_gen + ((grid_imp * grid_eta) - (grid_exp * grid_eta)) + (bat_dis - bat_ch) + ll == load_demand,
            name="energy_balance",
        )
    elif on_grid and not allow_export:
        model.add_constraints(
            res_sum + gen_gen + (grid_imp * grid_eta) + (bat_dis - bat_ch) + ll == load_demand,
            name="energy_balance",
        )
    else:
        model.add_constraints(
            res_sum + gen_gen + (bat_dis - bat_ch) + ll == load_demand,
            name="energy_balance",
        )

    # ---------------------------------------------------------------------
    # 9) System-level constraints: min renewable penetration & max lost load
    #
    # Enforcement modes:
    #  - scenario_wise: constraints hold for each scenario separately
    #  - expected: constraints hold on scenario-weighted totals
    # ---------------------------------------------------------------------

    # Total demand energy per scenario
    E_demand_s = load_demand.sum("period")  # (scenario,)
    E_ll_s     = ll.sum("period")          # (scenario,)
    E_res_s    = res_sum.sum("period")     # (scenario,)
    E_gen_s    = gen_gen.sum("period")     # (scenario,)

    if on_grid:
        E_grid_s = grid_imp.sum("period")  # (scenario,)
    else:
        E_grid_s = xr.DataArray(
            np.zeros((int(scenario.size),), dtype=float),
            coords={"scenario": scenario},
            dims=("scenario",),
        )

    # --- (A) Max lost-load share
    #     sum_t lost_load(t,s) <= max_ll_frac(s) * sum_t load(t,s)
    if enforcement == "scenario_wise":
        model.add_constraints(E_ll_s <= max_ll_frac * E_demand_s, name="max_lost_load_share")
    else:
        # expected: Σ_s w_s * LL_s <= max_ll_frac * Σ_s w_s * Demand_s
        E_ll_exp     = (E_ll_s * w_s).sum("scenario")       # scalar
        E_demand_exp = (E_demand_s * w_s).sum("scenario")   # scalar
        model.add_constraints(E_ll_exp <= max_ll_frac * E_demand_exp, name="max_lost_load_share_expected")

    # --- (B) Minimum renewable penetration
    #   E_total = E_res + E_gen + E_grid_import
    #   E_renew = E_res
    # Only add if min_res_pen > 0.0
    if min_res_pen > 0.0:
        if enforcement == "scenario_wise":
            E_total_s = E_res_s + E_gen_s + E_grid_s
            model.add_constraints(E_res_s >= min_res_pen * E_total_s, name="min_renewable_penetration")
        else:
            E_total_exp = ((E_res_s + E_gen_s + E_grid_s) * w_s).sum("scenario")  # scalar
            E_res_exp   = (E_res_s * w_s).sum("scenario")                         # scalar
            model.add_constraints(E_res_exp >= min_res_pen * E_total_exp, name="min_renewable_penetration_expected")

    # ---------------------------------------------------------------------
    # 10) Land availability (renewables only)
    #
    # area_used = Σ_r (res_units(r) * res_nominal_kw(r) * res_area_m2_per_kw(r))
    # ---------------------------------------------------------------------
    area_used = (res_units * res_nom_kw * res_area_m2_per_kw).sum("resource") # scalar
    model.add_constraints(area_used <= land_m2, name="land_availability") 



