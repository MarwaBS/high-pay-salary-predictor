"""
config_schema.py
----------------
Pydantic validation for config.yaml.

Catches typos and invalid values at startup rather than at runtime.
Usage: ``ProjectConfig.from_yaml("config.yaml")``
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    resources_dir: str
    data_dir: str
    images_dir: str
    models_dir: str
    raw_bls: str
    raw_census: str
    cleaned: str
    bls_processed: str
    census_processed: str


class ThresholdsConfig(BaseModel):
    min_annual_income: int = Field(ge=50_000)
    min_hourly_mean: float = Field(ge=20.0)


class ModelConfig(BaseModel):
    test_size: float = Field(ge=0.05, le=0.5)
    random_state: int
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1, le=20)
    learning_rate: float = Field(gt=0, le=1.0)
    subsample: float = Field(gt=0, le=1.0)
    colsample_bytree: float = Field(gt=0, le=1.0)
    reg_lambda: float = Field(ge=0)
    log_transform_target: bool
    cv_folds: int = Field(ge=2, le=20)
    model_path: str
    features_path: str
    metrics_path: str
    group_means_path: str
    # Premium-tier classifier head. Optional on purpose — pre-Phase-1
    # artefacts (any model trained before the classifier was added)
    # produced a config without these fields, and the API must stay
    # backwards-compatible against old config files.
    classifier_path: str | None = None
    premium_threshold: int | None = Field(default=None, ge=100_000)


class VisualizationColors(BaseModel):
    money_seq: str
    count_seq: str
    gender_male: str
    gender_female: str
    accent: str


class VisualizationConfig(BaseModel):
    dpi: int = Field(ge=72, le=600)
    figure_size: list[int]
    colors: VisualizationColors


class ProjectConfig(BaseModel):
    """Validated project configuration — single source of truth for all settings."""

    data: DataConfig
    thresholds: ThresholdsConfig
    model: ModelConfig
    visualization: VisualizationConfig
    education_order: dict[str, int]
    regions: dict[str, list[str]]

    @model_validator(mode="after")
    def _check_regions_cover_50_states(self) -> ProjectConfig:
        all_states = [s for states in self.regions.values() for s in states]
        if len(all_states) != 50:
            raise ValueError(f"regions must cover exactly 50 states, got {len(all_states)}")
        if len(set(all_states)) != len(all_states):
            dupes = [s for s in all_states if all_states.count(s) > 1]
            raise ValueError(f"duplicate states in regions: {set(dupes)}")
        return self

    @model_validator(mode="after")
    def _check_education_ordinal(self) -> ProjectConfig:
        values = sorted(self.education_order.values())
        if values != list(range(1, len(values) + 1)):
            raise ValueError(f"education_order values must be 1..N, got {values}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        """Load and validate config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
