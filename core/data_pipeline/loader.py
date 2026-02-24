from __future__ import annotations

from typing import Literal

import xarray as xr

from core.data_pipeline.utils import as_str
from core.data_pipeline.multi_year_loader import load_multi_year_dataset
from core.data_pipeline.typical_year_loader import (
    InputValidationError as TypicalInputValidationError,
    load_typical_year_dataset,
)


def _coerce_sets(sets: dict | xr.Dataset) -> xr.Dataset:
    """
    Coerce `sets` to xarray.Dataset for compatibility with existing loaders.

    Notes:
    - Existing call sites currently pass xr.Dataset.
    - This helper keeps signature flexibility while preserving error style.
    """
    if isinstance(sets, xr.Dataset):
        return sets

    if isinstance(sets, dict):
        # Best effort for future call-sites that may pass {"coords": {...}}.
        if "coords" in sets and isinstance(sets["coords"], dict):
            try:
                return xr.Dataset(coords=sets["coords"])
            except Exception as e:
                raise TypicalInputValidationError(
                    f"initialize_data expects `sets` as an xarray.Dataset. Could not coerce dict['coords'] to Dataset: {e}"
                )

    raise TypicalInputValidationError("initialize_data expects `sets` as an xarray.Dataset.")


def load_project_dataset(
    project_name: str,
    sets: dict,
    mode: Literal["typical_year", "multi_year"],
) -> xr.Dataset:
    """
    Shared project dataset loader (parallel implementation).

    Mode behavior:
    - typical_year: delegates to the current typical-year loader for strict parity.
    - multi_year: delegates to the current multi-year shared loader.
    """
    mode_s = as_str(mode, name="mode", default="", error_cls=ValueError)

    sets_ds = _coerce_sets(sets)

    if mode_s == "typical_year":
        return load_typical_year_dataset(project_name, sets_ds)

    if mode_s == "multi_year":
        return load_multi_year_dataset(project_name, sets_ds)

    raise ValueError(f"Unsupported mode: {mode!r}. Expected 'typical_year' or 'multi_year'.")
