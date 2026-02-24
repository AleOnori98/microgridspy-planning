# pages/2_inputs_viewer.py
from __future__ import annotations

from pathlib import Path
import streamlit as st

from core.io.utils import project_paths
from core.export.manifest import read_manifest
from core.export.csv_reader import (
    read_csv_2level_timeseries,
    read_csv_3level_timeseries,
)
from core.export.plots import plot_8760, plot_daily_profile_band


# ----------------------------
# wrappers
# ----------------------------
def _load_manifest(project_name: str):
    return read_manifest(project_name)

def _load_demand_csv(path_str: str):
    return read_csv_2level_timeseries(Path(path_str))


def _load_resource_csv(path_str: str):
    return read_csv_3level_timeseries(Path(path_str))


def _list_projects(projects_root: Path) -> list[str]:
    if not projects_root.exists():
        return []
    out = []
    for p in projects_root.iterdir():
        if p.is_dir():
            out.append(p.name)
    return sorted(out)


def _render_setup_summary(bundle) -> None:
    s = bundle.sets

    # Compact but informative
    txt = (
        f"**Project**: `{s.project_name}`  •  "
        f"**Formulation**: `{s.formulation}`  •  "
        f"**System**: `{s.system_type}`"
    )
    if s.on_grid:
        txt += f"  •  **Export**: `{s.allow_export}`"
    txt += (
        f"\n\n**Scenarios**: {', '.join(s.scenarios)}"
        f"  •  **Years**: {', '.join(s.years[:5])}{'…' if len(s.years) > 5 else ''}"
    )
    if s.formulation == "dynamic":
        txt += f"  •  **Capacity expansion**: `{s.capacity_expansion}`"
        if s.capacity_expansion and s.investment_steps_years:
            txt += f"  •  **Steps**: {', '.join(s.steps)} (durations={s.investment_steps_years})"
    txt += (
        f"\n\n**Renewables**: {s.n_sources} source(s)  •  "
        f"Technologies: {', '.join(s.conversion_technologies)}  •  "
        f"Resources: {', '.join(s.resources)}"
    )
    st.caption(txt)


def _render_selectors(sets):
    c1, c2 = st.columns([1, 1])
    with c1:
        scenario = st.selectbox(
            "Scenario",
            options=sets.scenarios,
            index=0,
        )
    with c2:
        year = st.selectbox(
            "Year",
            options=sets.years,
            index=0,
        )
    return scenario, year


def _render_demand_section(paths, sets, scenario: str, year: str):
    st.subheader("Load demand")
    demand_path = paths.inputs_dir / "load_demand.csv"
    if not demand_path.exists():
        st.error(f"Missing file: {demand_path}")
        return

    da = _load_demand_csv(str(demand_path))

    # sanity: ensure selection exists
    if scenario not in da.coords["scenario"].values:
        st.error(f"Scenario '{scenario}' not found in load_demand.csv.")
        return
    if year not in da.coords["year"].values:
        st.error(f"Year '{year}' not found in load_demand.csv.")
        return

    series = da.sel(scenario=scenario, year=year).values

    c1, c2 = st.columns([1.6, 1.0])
    with c1:
        fig1 = plot_8760(
            series,
            title=f"Hourly demand (8760) — {scenario} / {year}",
            y_label="Demand [kWh/h]",
        )
        st.pyplot(fig1, use_container_width=True)

    with c2:
        fig2 = plot_daily_profile_band(
            series,
            title="Average daily profile (mean + min–max band)",
            y_label="Demand [kWh/h]",
        )
        st.pyplot(fig2, use_container_width=True)

    with st.expander("Diagnostics", expanded=False):
        import numpy as np

        y = series.astype(float)
        st.write(
            {
                "n": int(len(y)),
                "nan": int(np.isnan(y).sum()),
                "min": float(np.nanmin(y)) if len(y) else None,
                "max": float(np.nanmax(y)) if len(y) else None,
                "mean": float(np.nanmean(y)) if len(y) else None,
                "annual_energy_kwh": float(np.nansum(y)) if len(y) else None,
            }
        )


def _render_resource_section(paths, sets, scenario: str, year: str):
    st.subheader("Resource availability")
    res_path = paths.inputs_dir / "resource_availability.csv"
    if not res_path.exists():
        st.error(f"Missing file: {res_path}")
        return

    da = _load_resource_csv(str(res_path))

    # resource selector (from file coords; usually matches manifest)
    resources_in_file = [str(x) for x in da.coords["resource"].values]
    resource = st.selectbox("Resource", options=resources_in_file, index=0)

    if scenario not in da.coords["scenario"].values:
        st.error(f"Scenario '{scenario}' not found in resource_availability.csv.")
        return
    if year not in da.coords["year"].values:
        st.error(f"Year '{year}' not found in resource_availability.csv.")
        return
    if resource not in da.coords["resource"].values:
        st.error(f"Resource '{resource}' not found in resource_availability.csv.")
        return

    series = da.sel(scenario=scenario, year=year, resource=resource).values

    c1, c2 = st.columns([1.6, 1.0])
    with c1:
        fig1 = plot_8760(
            series,
            title=f"Hourly availability (8760) — {resource} — {scenario} / {year}",
            y_label="Capacity factor [-]",
        )
        st.pyplot(fig1, use_container_width=True)

    with c2:
        fig2 = plot_daily_profile_band(
            series,
            title="Average daily profile (mean + min–max band)",
            y_label="Capacity factor [-]",
        )
        st.pyplot(fig2, use_container_width=True)

    with st.expander("Diagnostics", expanded=False):
        import numpy as np

        y = series.astype(float)
        st.write(
            {
                "n": int(len(y)),
                "nan": int(np.isnan(y).sum()),
                "min": float(np.nanmin(y)) if len(y) else None,
                "max": float(np.nanmax(y)) if len(y) else None,
                "mean": float(np.nanmean(y)) if len(y) else None,
                "pct_outside_0_1": float(np.mean((y < 0) | (y > 1))) if len(y) else None,
            }
        )


def render_inputs_viewer_page():
    st.title("Inputs Viewer")
    st.markdown(
        "Visualize and sanity-check input time series before running the optimization."
    )

    # Choose project
    active = st.session_state.get("active_project")
    projects_root = Path("projects")  # adjust if your repo uses a different root
    projects = _list_projects(projects_root)

    if active and active in projects:
        project_name = active
        st.info(f"Using active project: `{project_name}`")
    else:
        if not projects:
            st.error("No projects found. Create a project first in the Project Setup page.")
            return
        project_name = st.selectbox("Select project", options=projects, index=0)
        st.session_state["active_project"] = project_name

    # Load manifest and show summary caption
    bundle = _load_manifest(project_name)
    _render_setup_summary(bundle)

    paths = project_paths(project_name)
    sets = bundle.sets

    st.markdown("---")

    # Always-available selectors
    scenario, year = _render_selectors(sets)

    st.markdown("### Time series")
    _render_demand_section(paths, sets, scenario, year)
    st.markdown("---")
    _render_resource_section(paths, sets, scenario, year)


render_inputs_viewer_page()
