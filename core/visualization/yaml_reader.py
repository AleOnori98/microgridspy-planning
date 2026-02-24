# core/inputs/readers_yaml.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def flatten_yaml_placeholder(path: Path) -> pd.DataFrame:
    """
    MVP placeholder.
    Later: implement per-component flatteners for renewables/battery/generator/grid.
    """
    payload = read_yaml(path)
    return pd.DataFrame([{"path": str(path), "top_keys": list(payload.keys()) if isinstance(payload, dict) else None}])
