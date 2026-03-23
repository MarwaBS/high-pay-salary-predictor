"""
tests/conftest.py
-----------------
Session-scoped fixtures shared across ALL test modules.

Having fixtures here (instead of duplicated in each test file) means
pytest auto-discovers them and any test can use them without importing.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline import engineer_features


# ---------------------------------------------------------------------------
# Session-scope: load once, reuse across the entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg() -> dict:
    """Parsed project configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def df(cfg: dict) -> pd.DataFrame:
    """Raw cleaned dataset (no engineered features)."""
    data_path = Path(__file__).parent.parent / cfg["data"]["cleaned"]
    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def edu_order(cfg: dict) -> dict[str, int]:
    return cfg["education_order"]


@pytest.fixture(scope="session")
def region_map(cfg: dict) -> dict[str, str]:
    return {
        state: region
        for region, states in cfg["regions"].items()
        for state in states
    }


@pytest.fixture(scope="session")
def df_engineered(df: pd.DataFrame, edu_order: dict, region_map: dict) -> pd.DataFrame:
    """Dataset with all model-ready derived columns (from pipeline.engineer_features)."""
    return engineer_features(df, edu_order, region_map)
