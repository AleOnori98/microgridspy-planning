from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd
import xarray as xr

from core.export.common import get_bundle_formulation, get_var_solution as _get_var_solution_common
from core.export.results_bundle import ResultsBundle, build_results_bundle
from core.export.typical_year_results import (
    build_dispatch_timeseries_table,
    build_energy_balance_table,
    export_typical_year_results,
)
from core.export.multi_year_results import (
    build_dispatch_timeseries_table_multi_year,
    build_energy_balance_table_multi_year,
    export_multi_year_results,
)


def get_results_bundle_from_session(session_state: Mapping[str, Any]) -> Optional[ResultsBundle]:
    raw = session_state.get("gp_results_bundle")
    if isinstance(raw, ResultsBundle):
        model_obj = session_state.get("gp_model_obj")
        model_sol = getattr(getattr(model_obj, "model", None), "solution", None)
        if isinstance(model_sol, xr.Dataset):
            raw.solution = model_sol
        return raw

    data = session_state.get("gp_data")
    vars_dict = session_state.get("gp_vars")
    if not isinstance(data, xr.Dataset) or not isinstance(vars_dict, dict):
        return None

    return build_results_bundle(
        sets=session_state.get("gp_sets"),
        data=data,
        vars=vars_dict,
        model_obj=session_state.get("gp_model_obj"),
        solution=session_state.get("gp_solution"),
        solution_summary=session_state.get("gp_solution_summary"),
        solver=None,
    )

def get_var_solution(*, bundle: ResultsBundle, name: str) -> Optional[xr.DataArray]:
    return _get_var_solution_common(
        vars_dict=bundle.vars if isinstance(bundle.vars, dict) else None,
        solution=bundle.solution if isinstance(bundle.solution, xr.Dataset) else None,
        name=name,
    )


def build_energy_balance_dataframe(bundle: ResultsBundle) -> pd.DataFrame:
    if bundle.data is None or not isinstance(bundle.vars, dict):
        raise RuntimeError("Missing data/vars in ResultsBundle.")
    formulation = get_bundle_formulation(bundle)
    if formulation == "dynamic":
        dispatch = build_dispatch_timeseries_table_multi_year(
            data=bundle.data,
            vars=bundle.vars,
            solution=bundle.solution if isinstance(bundle.solution, xr.Dataset) else None,
        )
        return build_energy_balance_table_multi_year(dispatch)
    dispatch = build_dispatch_timeseries_table(
        data=bundle.data,
        vars=bundle.vars,
        solution=bundle.solution if isinstance(bundle.solution, xr.Dataset) else None,
    )
    return build_energy_balance_table(dispatch)


def export_results_from_bundle(
    project_name: str,
    bundle: ResultsBundle,
    model_obj: Any = None,
) -> Dict[str, str]:
    if bundle.data is None or not isinstance(bundle.vars, dict):
        raise RuntimeError("Missing data/vars in ResultsBundle.")

    sets_ds = bundle.sets if isinstance(bundle.sets, xr.Dataset) else xr.Dataset()
    formulation = get_bundle_formulation(bundle)
    if formulation == "dynamic":
        return export_multi_year_results(
            project_name=project_name,
            sets=sets_ds,
            data=bundle.data,
            model=getattr(model_obj, "model", None),
            vars=bundle.vars,
            solution=bundle.solution if isinstance(bundle.solution, xr.Dataset) else None,
            out_dir=None,
        )
    return export_typical_year_results(
        project_name=project_name,
        sets=sets_ds,
        data=bundle.data,
        model=getattr(model_obj, "model", None),
        vars=bundle.vars,
        solution=bundle.solution if isinstance(bundle.solution, xr.Dataset) else None,
        out_dir=None,
    )
