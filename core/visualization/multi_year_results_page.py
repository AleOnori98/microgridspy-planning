from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from core.export.multi_year_results import (
    build_design_by_step_table_multi_year,
    build_dispatch_timeseries_table_multi_year,
)
from core.export.results_bundle import ResultsBundle


C_RES = "#FFD700"
C_BAT = "#00ACC1"
C_GEN = "#546E7A"
C_IMP = "#9C27B0"
C_EXP = "#9C27B0"
C_LL = "#E53935"
C_LOAD = "#111111"


def _get_settings(data: xr.Dataset) -> Dict[str, Any]:
    settings = (data.attrs or {}).get("settings", {})
    return settings if isinstance(settings, dict) else {}


def _days_to_slice(T: int, start_day: int, ndays: int) -> Tuple[slice, int]:
    ndays = int(np.clip(ndays, 1, 7))
    max_day = max(1, int(np.ceil(T / 24)))
    start_day = int(np.clip(start_day, 1, max_day))
    window = ndays * 24
    i0 = (start_day - 1) * 24
    if i0 + window > T:
        i0 = max(0, T - window)
    i1 = min(T, i0 + window)
    return slice(i0, i1), i1 - i0


def _scenario_options(settings: Dict[str, Any], data: xr.Dataset) -> list[str]:
    ms_enabled = bool(((settings.get("multi_scenario", {}) or {}).get("enabled", False)))
    scen_labels = [str(s) for s in data.coords["scenario"].values.tolist()]
    if not ms_enabled:
        return scen_labels
    return ["Expected"] + scen_labels


def _build_expected_dispatch(view: pd.DataFrame, weights: Optional[xr.DataArray]) -> pd.DataFrame:
    if not isinstance(weights, xr.DataArray):
        return view.groupby("period", as_index=False).mean(numeric_only=True)
    w_map = {str(s): float(weights.sel(scenario=s)) for s in weights.coords["scenario"].values}
    weighted = view.copy()
    weighted["w"] = weighted["scenario"].astype(str).map(w_map).fillna(0.0)
    agg_cols = [
        "load_demand",
        "res_generation_total",
        "generator_generation",
        "battery_charge",
        "battery_discharge",
        "lost_load",
        "grid_import",
        "grid_export",
    ]
    return (
        weighted.groupby("period", as_index=False)
        .apply(lambda g: pd.Series({c: float((g[c] * g["w"]).sum()) for c in agg_cols}))
        .reset_index(drop=True)
    )


def _average_daily_profile(view: pd.DataFrame, *, start_day: int, ndays: int) -> pd.DataFrame:
    ordered = view.sort_values("period").reset_index(drop=True)
    idx, window = _days_to_slice(len(ordered), start_day=start_day, ndays=ndays)
    window_df = ordered.iloc[idx].copy().reset_index(drop=True)
    if window <= 0:
        return pd.DataFrame()
    window_df["hour_of_day"] = (np.arange(len(window_df)) % 24) + 1
    cols = [
        "load_demand",
        "res_generation_total",
        "generator_generation",
        "battery_charge",
        "battery_discharge",
        "lost_load",
        "grid_import",
        "grid_export",
    ]
    return window_df.groupby("hour_of_day", as_index=False)[cols].mean()


def _plot_dispatch_stack(*, ax: Any, profile: pd.DataFrame, title_suffix: str) -> None:
    x = profile["hour_of_day"].to_numpy(dtype=float)
    y_res = profile["res_generation_total"].to_numpy(dtype=float)
    y_bdis = profile["battery_discharge"].to_numpy(dtype=float)
    y_gen = profile["generator_generation"].to_numpy(dtype=float)
    y_gimp = profile["grid_import"].to_numpy(dtype=float)
    y_ll = profile["lost_load"].to_numpy(dtype=float)
    y_bch = profile["battery_charge"].to_numpy(dtype=float)
    y_gexp = profile["grid_export"].to_numpy(dtype=float)
    y_load = profile["load_demand"].to_numpy(dtype=float)

    p1 = y_res
    p2 = p1 + y_bdis
    p3 = p2 + y_gen
    p4 = p3 + y_gimp
    p5 = p4 + y_ll

    n1 = -y_bch
    n2 = n1 - y_gexp

    ax.fill_between(x, 0, p1, color=C_RES, alpha=0.85, label="Renewables")
    ax.fill_between(x, p1, p2, color=C_BAT, alpha=0.35, label="Battery discharge")
    ax.fill_between(x, p2, p3, color=C_GEN, alpha=0.85, label="Generator")
    if np.any(y_gimp > 0):
        ax.fill_between(x, p3, p4, color=C_IMP, alpha=0.85, label="Grid import")
    if np.any(y_ll > 0):
        ax.fill_between(x, p4, p5, color=C_LL, alpha=0.45, label="Lost load")
    if np.any(y_bch > 0):
        ax.fill_between(x, 0, n1, color=C_BAT, alpha=0.35, label="Battery charge")
    if np.any(y_gexp > 0):
        ax.fill_between(x, n1, n2, color=C_EXP, alpha=0.75, label="Grid export")

    ax.plot(x, y_load, color=C_LOAD, linewidth=1.8, label="Load")
    ax.set_title(f"Average daily profile - {title_suffix}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kWh per hour (approx. kW)")
    ax.set_xticks(np.arange(1, 25, 1))
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(ncols=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.22))


def _capacity_by_year(sets_ds: xr.Dataset, design_df: pd.DataFrame) -> pd.DataFrame:
    if "year" in sets_ds.coords:
        years = [str(y) for y in sets_ds.coords["year"].values.tolist()]
    else:
        years = sorted(design_df["inv_step_start_year"].astype(str).unique().tolist())
    rows = []
    for year in years:
        active = design_df[design_df["inv_step_start_year"].astype(str) <= str(year)].copy()
        renewable_kw = float(active.loc[active["technology"] == "renewable", "installed_capacity"].sum())
        battery_kwh = float(active.loc[active["technology"] == "battery", "installed_capacity"].sum())
        generator_kw = float(active.loc[active["technology"] == "generator", "installed_capacity"].sum())
        rows.append(
            {
                "year": year,
                "renewables_kw": renewable_kw,
                "battery_kwh": battery_kwh,
                "generator_kw": generator_kw,
            }
        )
    return pd.DataFrame(rows)


def render_multi_year_results(bundle: ResultsBundle, project_name: Optional[str]) -> None:
    data = bundle.data
    if not isinstance(data, xr.Dataset):
        st.error("Missing data dataset.")
        return

    settings = _get_settings(data)
    vars_dict = bundle.vars if isinstance(bundle.vars, dict) else {}
    sol_ds = bundle.solution if isinstance(bundle.solution, xr.Dataset) else None
    sets_ds = bundle.sets if isinstance(bundle.sets, xr.Dataset) else xr.Dataset()

    dispatch = build_dispatch_timeseries_table_multi_year(data=data, vars=vars_dict, solution=sol_ds)
    design = build_design_by_step_table_multi_year(sets=sets_ds, data=data, vars=vars_dict, solution=sol_ds)

    st.subheader("Total Capacity Installed")
    st.dataframe(_capacity_by_year(sets_ds, design), hide_index=True, width="stretch")

    with st.expander("Investment-step breakdown", expanded=False):
        st.dataframe(design, hide_index=True, width="stretch")

    st.subheader("Least-Cost Energy Mix")
    st.caption("Average daily dispatch profile over a selected day window, keeping the same color convention as the typical-year results.")

    years = sorted(dispatch["year"].astype(str).unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        year_sel = st.selectbox("Year", years, key="my_results_year")
    with c2:
        scenario_sel = st.selectbox("Scenario", _scenario_options(settings, data), key="my_results_scenario")

    year_view = dispatch[dispatch["year"].astype(str) == str(year_sel)].copy()
    if scenario_sel == "Expected":
        series_view = _build_expected_dispatch(year_view, data.get("scenario_weight"))
        scenario_label = "Expected"
    else:
        series_view = year_view[year_view["scenario"].astype(str) == str(scenario_sel)].copy()
        scenario_label = str(scenario_sel)

    T = int(len(series_view))
    with st.expander("Time window", expanded=False):
        st.caption("Pick the start day and how many days to average (1-7).")
        start_day = st.slider(
            "Start day",
            min_value=1,
            max_value=max(1, int(np.ceil(T / 24))),
            value=1,
            step=1,
            key="my_results_start_day",
        )
        ndays = st.slider(
            "Number of days",
            min_value=1,
            max_value=7,
            value=1,
            step=1,
            key="my_results_ndays",
        )

    profile = _average_daily_profile(series_view, start_day=start_day, ndays=ndays)
    if profile.empty:
        st.warning("No dispatch data available for the selected year/scenario window.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    _plot_dispatch_stack(ax=ax, profile=profile, title_suffix=f"Year {year_sel} - {scenario_label}")
    st.pyplot(fig)

    with st.expander("Average daily profile table", expanded=False):
        st.dataframe(profile, hide_index=True, width="stretch")
