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
    cleaned: str


class ThresholdsConfig(BaseModel):
    # Floors match the advertised ≥$100K cohort (hourly = 100000 / 2080).
    min_annual_income: int = Field(ge=100_000)
    min_hourly_mean: float = Field(ge=48.0)


class DriftConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # Only positivity is checkable here. The binding bound is the detector's
    # effect-floor handover, which depends on its tuning, so the API checks it
    # at startup where both are known.
    window: int = Field(ge=1)


class ModelConfig(BaseModel):
    # Unknown keys are rejected: a mistyped optional knob (``stabilty_seeds``)
    # would otherwise validate clean and silently fall back to a default.
    model_config = {"extra": "forbid"}

    test_size: float = Field(ge=0.05, le=0.5)
    random_state: int
    # Training thread count. XGBoost's hist tree method sums gradient
    # histograms in thread-partition order, so the fitted bytes depend on how
    # many threads ran; ``n_jobs: 1`` is the only setting reproducible on a
    # machine of any size. Seeds and pinned libraries alone do not fix it.
    n_jobs: int = Field(default=1, ge=1)
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1, le=20)
    learning_rate: float = Field(gt=0, le=1.0)
    subsample: float = Field(gt=0, le=1.0)
    colsample_bytree: float = Field(gt=0, le=1.0)
    reg_lambda: float = Field(ge=0)
    cv_folds: int = Field(ge=2, le=20)
    # Seeds the trainer refits under to report metric stability.
    stability_seeds: list[int] = Field(min_length=1)
    model_path: str
    features_path: str
    metrics_path: str
    group_means_path: str
    baseline_stats_path: str
    # Cross-conformal interval margin. Optional: when absent, the API serves
    # the raw (uncalibrated) interval.
    conformal_path: str | None = None
    # Premium-tier classifier head. Required together — the trainer trains both
    # heads. A configured artefact that is missing on disk still degrades to
    # ``p_above_premium_threshold: None``.
    classifier_path: str = Field(min_length=1)
    premium_threshold: int = Field(ge=100_000)
    classifier_n_estimators: int = Field(ge=1)
    classifier_max_depth: int = Field(ge=1, le=20)
    classifier_learning_rate: float = Field(gt=0, le=1.0)
    classifier_subsample: float = Field(gt=0, le=1.0)
    classifier_colsample_bytree: float = Field(gt=0, le=1.0)
    classifier_reg_lambda: float = Field(ge=0)


class VisualizationColors(BaseModel):
    money_seq: str
    count_seq: str
    gender_male: str
    gender_female: str


class VisualizationConfig(BaseModel):
    dpi: int = Field(ge=72, le=600)
    colors: VisualizationColors


class ProjectConfig(BaseModel):
    """Validated project configuration — single source of truth for all settings."""

    data: DataConfig
    thresholds: ThresholdsConfig
    drift: DriftConfig
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
