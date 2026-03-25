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

from pipeline import engineer_features, load_group_means, load_model

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
def group_means(cfg: dict) -> dict:
    """Training-set group means loaded from the saved artefact.

    Using the saved values (computed from the training split only) ensures
    tests exercise the same encoding path as production inference.
    """
    gm_path = Path(__file__).parent.parent / cfg["model"]["group_means_path"]
    return load_group_means(str(gm_path))


@pytest.fixture(scope="session")
def df_engineered(
    df: pd.DataFrame, edu_order: dict, region_map: dict, group_means: dict
) -> pd.DataFrame:
    """Dataset with all model-ready derived columns (from pipeline.engineer_features).

    Uses saved training-set group means so encoding is consistent with
    what the production model saw during training.
    """
    return engineer_features(
        df, edu_order, region_map,
        occ_means=group_means["occ_means"],
        state_means=group_means["state_means"],
    )


@pytest.fixture(scope="session")
def production_model(cfg: dict):
    """The trained production XGBoost model loaded from disk.

    Requires 'make model' (or 'python scripts/train_model.py') to have been
    run first. CI runs that step before pytest.
    """
    model_path = Path(__file__).parent.parent / cfg["model"]["model_path"]
    return load_model(str(model_path))
