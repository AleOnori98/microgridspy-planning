from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Type

import json
import numpy as np
import xarray as xr
import yaml


def read_json_or_raise(path: Path, *, error_cls: Type[Exception] = RuntimeError) -> Dict[str, Any]:
    """Read and parse JSON file, raising error_cls on failure."""
    if not path.exists():
        raise error_cls(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise error_cls(f"Cannot parse JSON: {path}\nerror: {e}")


def read_yaml_or_raise(path: Path, *, error_cls: Type[Exception] = RuntimeError) -> Dict[str, Any]:
    """Read and parse YAML file, raising error_cls on failure."""
    if not path.exists():
        raise error_cls(f"Missing required file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise error_cls(f"Cannot parse YAML: {path}\nerror: {e}")


def as_float(x: Any, *, name: str, default: float = 0.0, error_cls: Type[Exception] = RuntimeError) -> float:
    """Convert x to float, with default if None. Raise error_cls on failure."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception as e:
        raise error_cls(f"Invalid value for '{name}': {x!r} (error: {e})")


def as_float_or_nan(x: Any, *, name: str, error_cls: Type[Exception] = RuntimeError) -> float:
    """Convert x to float, or NaN if None. Raise error_cls on failure."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception as e:
        raise error_cls(f"Invalid numeric value for '{name}': {x!r} (error: {e})")


def as_str(x: Any, *, name: str, default: str = "", error_cls: Type[Exception] = RuntimeError) -> str:
    """Convert x to str, with default if None. Raise error_cls on failure."""
    if x is None:
        return default
    try:
        return str(x)
    except Exception as e:
        raise error_cls(f"Invalid value for '{name}': {x!r} (error: {e})")


def normalize_weights(weights: Sequence[float], n: int) -> list[float]:
    """Normalize a list of weights to sum to 1.0 over n items."""
    if n <= 0:
        return [1.0]
    w = [float(x) for x in (weights or [])]
    if len(w) != n:
        w = [1.0 / n] * n
    s = float(sum(w))
    if s <= 0:
        return [1.0 / n] * n
    return [wi / s for wi in w]


def broadcast_to_scenario(value: xr.DataArray, scenario_coord: xr.DataArray) -> xr.DataArray:
    """Broadcast a scalar DataArray to scenario dimension."""
    if value.ndim != 0:
        return value
    return xr.DataArray(
        np.full((scenario_coord.size,), float(value.values)),
        coords={"scenario": scenario_coord},
        dims=("scenario",),
        name=value.name,
        attrs=dict(value.attrs or {}),
    )

