"""Pydantic request/response schemas for the salary prediction API."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input features for salary prediction.

    Required fields map to the demographic and geographic inputs a user
    would realistically know. BLS context fields (employment, lq, etc.)
    are optional — the API fills them with occupation/state medians when
    not supplied.
    """

    state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two-letter US state abbreviation (e.g. 'CA', 'NY').",
        examples=["CA"],
    )
    occupation: str = Field(
        ...,
        min_length=2,
        description="Occupation title (must match a title in the dataset).",
        examples=["Software Developers"],
    )
    education_level: str = Field(
        ...,
        description="Highest education level attained.",
        examples=["Bachelor's degree"],
    )
    gender: str = Field(
        ...,
        description="Gender ('Male' or 'Female').",
        examples=["Female"],
    )
    age: int = Field(
        ...,
        ge=18,
        le=80,
        description="Age in years (18–80).",
        examples=[32],
    )

    # Optional BLS context — filled from dataset medians if omitted
    employment: Optional[float] = Field(
        default=None,
        ge=0,
        description="State-occupation employment count (BLS OEWS). Defaults to dataset median.",
    )
    location_quotient: Optional[float] = Field(
        default=None,
        ge=0,
        description="Location Quotient for this occupation in the state. Defaults to dataset median.",
    )
    jobs_per_1000: Optional[float] = Field(
        default=None,
        ge=0,
        description="Jobs per 1,000 total employment. Defaults to dataset median.",
    )
    hourly_mean: Optional[float] = Field(
        default=None,
        ge=0,
        description="BLS hourly mean wage for this occupation in the state ($). Defaults to dataset median.",
    )
    annual_mean_wage: Optional[float] = Field(
        default=None,
        ge=0,
        description="BLS annual mean wage for this occupation in the state ($). Defaults to dataset median.",
    )

    @field_validator("state")
    @classmethod
    def state_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("gender")
    @classmethod
    def gender_title_case(cls, v: str) -> str:
        normalised = v.strip().title()
        if normalised not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return normalised

    model_config = {"json_schema_extra": {
        "example": {
            "state": "CA",
            "occupation": "Software Developers",
            "education_level": "Bachelor's degree",
            "gender": "Female",
            "age": 32,
        }
    }}


# ── Response ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Salary prediction result with contextual benchmarks."""

    predicted_salary: float = Field(
        ...,
        description="Predicted annual income in USD.",
        examples=[175000.0],
    )
    percentile_in_group: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Percentage of similar workers (same state + education level) "
            "in the dataset who earn less than the predicted salary."
        ),
        examples=[62.4],
    )
    group_median: float = Field(
        ...,
        description="Median annual income for the same state + education level ($).",
        examples=[148000.0],
    )
    group_mean: float = Field(
        ...,
        description="Mean annual income for the same state + education level ($).",
        examples=[160000.0],
    )
    group_size: int = Field(
        ...,
        description="Number of records in the comparison group.",
        examples=[214],
    )
    prediction_interval_low: float = Field(
        ...,
        description="Lower bound of the empirical 80% prediction interval ($).",
        examples=[101065.0],
    )
    prediction_interval_high: float = Field(
        ...,
        description="Upper bound of the empirical 80% prediction interval ($).",
        examples=[239101.0],
    )
    state: str
    occupation: str
    education_level: str
    gender: str
    age: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    dataset_rows: int
    version: str = "1.0.0"


class MetaResponse(BaseModel):
    states: list[str]
    occupations: list[str]
    education_levels: list[str]
