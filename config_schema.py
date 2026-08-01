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

from api.drift import MIN_WINDOW_FOR_VERDICT


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
    # Floors match the advertised ≥$100K cohort (hourly = 100000 / 2080).
    min_annual_income: int = Field(ge=100_000)
    min_hourly_mean: float = Field(ge=48.0)


class DriftConfig(BaseModel):
    # Same reason as ModelConfig: a mistyped knob would validate clean.
    model_config = {"extra": "forbid"}

    # A window under the verdict floor withholds every verdict forever, which
    # looks identical to a healthy monitor that has simply seen no drift.
    window: int = Field(ge=MIN_WINDOW_FOR_VERDICT)


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
    log_transform_target: bool
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
    # Premium-tier classifier head. Optional: when these fields are absent the
    # API runs without the classifier head (``p_above_premium_threshold`` is
    # ``None``).
    classifier_path: str | None = None
    premium_threshold: int | None = Field(default=None, ge=100_000)
    classifier_n_estimators: int | None = Field(default=None, ge=1)
    classifier_max_depth: int | None = Field(default=None, ge=1, le=20)
    classifier_learning_rate: float | None = Field(default=None, gt=0, le=1.0)
    classifier_subsample: float | None = Field(default=None, gt=0, le=1.0)
    classifier_colsample_bytree: float | None = Field(default=None, gt=0, le=1.0)
    classifier_reg_lambda: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _classifier_config_is_all_or_nothing(self) -> ModelConfig:
        """A configured classifier needs every hyperparameter the trainer reads.

        The trainer reads all seven unguarded, so a half-declared classifier
        dies partway through training instead of at load. Enforced at API
        startup and in CI, where ``ProjectConfig`` is loaded.
        """
        required = {
            "premium_threshold": self.premium_threshold,
            "classifier_n_estimators": self.classifier_n_estimators,
            "classifier_max_depth": self.classifier_max_depth,
            "classifier_learning_rate": self.classifier_learning_rate,
            "classifier_subsample": self.classifier_subsample,
            "classifier_colsample_bytree": self.classifier_colsample_bytree,
            "classifier_reg_lambda": self.classifier_reg_lambda,
        }
        if self.classifier_path:
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(f"classifier_path is set but these classifier settings are missing: {missing}")
        return self


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
