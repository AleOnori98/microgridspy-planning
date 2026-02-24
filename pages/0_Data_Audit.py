from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import json

from core.data_pipeline.loader import load_project_dataset
from core.io.utils import project_paths
from core.typical_year_model.sets import initialize_sets
from core.multi_year_model.sets import initialize_sets as initialize_sets_multi_year
from core.multi_year_model.model import MultiYearModel


REQUIRED_INPUTS = [
    "formulation.json",
    "load_demand.csv",
    "resource_availability.csv",
    "renewables.yaml",
    "battery.yaml",
    "generator.yaml",
]

REQUIRED_SETTINGS_KEYS = [
    "project_name",
    "formulation",
    "unit_commitment",
    "multi_scenario",
    "resources",
    "optimization_constraints",
    "inputs_loaded",
    "generator",
    "fuel",
    "grid",
]

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _list_projects(projects_root: Path) -> List[str]:
    if not projects_root.exists():
        return []
    return sorted([p.name for p in projects_root.iterdir() if p.is_dir()])


def _presence_checks(project_name: str) -> Dict[str, bool]:
    paths = project_paths(project_name)
    found: Dict[str, bool] = {}
    for name in REQUIRED_INPUTS:
        found[name] = (paths.inputs_dir / name).exists()
    return found


def _dims_coords_summary(ds: xr.Dataset) -> pd.DataFrame:
    rows = []
    for dim, size in ds.sizes.items():
        preview = ""
        if dim in ds.coords:
            vals = ds.coords[dim].values
            preview = ", ".join(map(str, vals[:5])) + (" ..." if len(vals) > 5 else "")
        rows.append({"dim": dim, "size": int(size), "coord_preview": preview})
    return pd.DataFrame(rows)


def _var_stats(ds: xr.Dataset) -> pd.DataFrame:
    rows = []
    for name in sorted(ds.data_vars):
        da = ds[name]
        vals = np.asarray(da.values).reshape(-1)
        is_num = np.issubdtype(vals.dtype, np.number)
        if is_num:
            v = vals.astype(float, copy=False)
            min_v = float(np.nanmin(v)) if v.size else np.nan
            max_v = float(np.nanmax(v)) if v.size else np.nan
            mean_v = float(np.nanmean(v)) if v.size else np.nan
            nan_count = int(np.isnan(v).sum())
        else:
            min_v = np.nan
            max_v = np.nan
            mean_v = np.nan
            nan_count = int(pd.isna(vals).sum())

        rows.append(
            {
                "var": name,
                "dims": ", ".join(da.dims) if da.dims else "(scalar)",
                "shape": tuple(int(x) for x in da.shape),
                "dtype": str(da.dtype),
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "nan_count": nan_count,
            }
        )
    return pd.DataFrame(rows)


def _cost_vars(ds: xr.Dataset) -> List[str]:
    keys = []
    for v in ds.data_vars:
        n = str(v).lower()
        if "cost" in n or "capex" in n or "subsidy" in n:
            keys.append(v)
    return sorted(keys)


def render_data_audit_page() -> None:
    st.title("Data Audit")
    st.caption("Data pipeline audit (read-only).")

    projects_root = Path("projects")
    projects = _list_projects(projects_root)
    if not projects:
        st.error("No project folders found under ./projects.")
        return

    active = st.session_state.get("active_project")
    if active in projects:
        idx = projects.index(active)
    else:
        idx = 0

    project_name = st.selectbox("Select project", options=projects, index=idx)
    st.session_state["active_project"] = project_name
    paths = project_paths(project_name)
    formulation = _read_json(paths.formulation_json)
    formulation_mode = str(formulation.get("core_formulation", "steady_state")).strip()
    loader_mode = "multi_year" if formulation_mode == "dynamic" else "typical_year"
    st.info(f"Auditing project: `{paths.root}`")
    st.caption(f"Detected formulation mode: `{formulation_mode}`")

    with st.expander("Section A: Required Input Files", expanded=True):
        checks = _presence_checks(project_name)
        present = [k for k, v in checks.items() if v]
        missing = [k for k, v in checks.items() if not v]

        st.dataframe(
            pd.DataFrame(
                [{"file": k, "present": checks[k]} for k in REQUIRED_INPUTS]
            ),
            use_container_width=True,
            hide_index=True,
        )

        if missing:
            st.error(f"FAIL: Missing required input files: {', '.join(missing)}")
        else:
            st.success("PASS: All required input files are present.")

        if present:
            st.caption(f"Found: {', '.join(present)}")

    ds = None
    with st.expander("Section B: Dataset Load + Summary", expanded=True):
        try:
            if loader_mode == "multi_year":
                sets = initialize_sets_multi_year(project_name)
            else:
                sets = initialize_sets(project_name)
            ds = load_project_dataset(project_name, sets, mode=loader_mode)
        except Exception as e:
            st.error(f"FAIL: Could not load dataset via shared loader: {e}")
            st.stop()

        st.success("PASS: Dataset loaded.")

        st.markdown("**Dims/Coords**")
        st.dataframe(_dims_coords_summary(ds), use_container_width=True, hide_index=True)

        st.markdown("**Variables Stats (min/max/mean, NaN counts)**")
        stats_df = _var_stats(ds)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("**Scenario Weights Check**")
        if "scenario_weight" not in ds.data_vars:
            st.error("FAIL: `scenario_weight` variable not found.")
        else:
            w = ds["scenario_weight"].values.astype(float)
            wsum = float(np.sum(w))
            if np.isclose(wsum, 1.0, atol=1e-8):
                st.success(f"PASS: scenario_weight sums to 1.0 (sum={wsum:.8f})")
            else:
                st.warning(f"WARN: scenario_weight does not sum to 1.0 (sum={wsum:.8f})")

    with st.expander("Section C: Sanity Assertions (Soft Warnings)", expanded=True):
        if ds is None:
            st.warning("Dataset not loaded.")
            return

        # 1) load_demand non-negative
        if "load_demand" in ds:
            arr = np.asarray(ds["load_demand"].values, dtype=float)
            if np.nanmin(arr) < 0:
                st.warning(f"WARN: load_demand has negative values (min={np.nanmin(arr):.6g}).")
            else:
                st.success("PASS: load_demand is non-negative.")
        else:
            st.warning("WARN: load_demand not found.")

        # 2) resource_availability in [0,1]
        if "resource_availability" in ds:
            arr = np.asarray(ds["resource_availability"].values, dtype=float)
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if mn < 0 or mx > 1:
                st.warning(f"WARN: resource_availability outside [0,1] (min={mn:.6g}, max={mx:.6g}).")
            else:
                st.success("PASS: resource_availability is within [0,1].")
        else:
            st.warning("WARN: resource_availability not found.")

        # 3) costs non-negative
        cost_names = _cost_vars(ds)
        neg_costs = []
        for v in cost_names:
            vals = np.asarray(ds[v].values)
            if np.issubdtype(vals.dtype, np.number):
                if np.nanmin(vals.astype(float, copy=False)) < 0:
                    neg_costs.append(v)
        if neg_costs:
            st.warning(f"WARN: negative values found in cost-like variables: {', '.join(neg_costs)}")
        else:
            st.success("PASS: no negative values in cost-like variables.")

        # 4) required settings keys
        settings = ds.attrs.get("settings", {})
        missing = [k for k in REQUIRED_SETTINGS_KEYS if k not in settings]
        if missing:
            st.warning(f"WARN: missing settings keys: {', '.join(missing)}")
        else:
            st.success("PASS: required settings keys are present.")

    if loader_mode == "multi_year":
        with st.expander("Section D: Multi-Year Build-Only Check", expanded=True):
            if st.button("Build multi-year model (vars+constraints)", type="secondary"):
                try:
                    m = MultiYearModel(project_name)
                    m._initialize_sets()
                    m._initialize_data()
                    m._initialize_vars()
                    m._initialize_constraints()

                    n_vars = len(m.vars or {})
                    n_constraints = len(m.model.constraints) if (m.model is not None and hasattr(m.model, "constraints")) else 0
                    st.success("PASS: Build-only completed for multi-year.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("# Variables", n_vars)
                    with c2:
                        st.metric("# Constraints", n_constraints)
                except Exception as e:
                    st.error(f"FAIL: build-only step failed: {e}")


render_data_audit_page()
