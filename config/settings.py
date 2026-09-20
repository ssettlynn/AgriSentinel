"""
config/settings.py — loads config/settings.yaml once.

Rule: run parameters (split years, model hyperparameters, thresholds) are
never hardcoded in a notebook cell or src module. Everything reads through
get_settings(), so changing a hyperparameter means editing one YAML file,
not hunting through pipeline code.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from config import paths


@functools.lru_cache(maxsize=1)
def get_settings() -> dict:
    with open(paths.PROJECT_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)
