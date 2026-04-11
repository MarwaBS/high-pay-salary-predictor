"""Pydantic request/response schemas for the salary prediction API."""

from pydantic import BaseModel, Field, field_validator

from api import __version__

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
        max_length=200,
        description="Occupation title (must match a title in the dataset).",
        examples=["Software Developers"],
    )
    education_level: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Highest education level attained.",
        examples=["Bachelor's degree"],
    )
    gender: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description=(
            "Gender ('Male' or 'Female'). "
            "**Limitation**: the training data (US Census CPS) uses a binary "
            "gender coding; non-binary identities are not represented. "
            "The model encodes this as a binary feature and cannot produce "
            "meaningful predictions outside this binary."
        ),
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
    employment: float | None = Field(
        default=None,
        ge=0,
        description="State-occupation employment count (BLS OEWS). Defaults to dataset median.",
    )
    location_quotient: float | None = Field(
        default=None,
        ge=0,
        description="Location Quotient for this occupation in the state. Defaults to dataset median.",
    )
    jobs_per_1000: float | None = Field(
        default=None,
        ge=0,
        description="Jobs per 1,000 total employment. Defaults to dataset median.",
    )
    hourly_mean: float | None = Field(
        default=None,
        ge=0,
        description="BLS hourly mean wage for this occupation in the state ($). Defaults to dataset median.",
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "state": "CA",
                "occupation": "Software Developers",
                "education_level": "Bachelor's degree",
                "gender": "Female",
                "age": 32,
            }
        }
    }


# ── Response ──────────────────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    """Salary prediction result — quantile trio + contextual benchmarks.

    The model is a multi-quantile XGBoost regressor. ``predicted_p50`` is
    the median prediction, ``predicted_p10``/``predicted_p90`` are the
    lower/upper bounds of the 80% quantile interval. ``predicted_salary``
    is an alias for ``predicted_p50`` kept for backward compatibility with
    existing clients; new clients should prefer the explicit quantile
    fields.
    """

    predicted_salary: float = Field(
        ...,
        description=(
            "Alias for ``predicted_p50`` — the median (P50) prediction. "
            "Kept for backward compatibility. New clients should use the "
            "explicit quantile fields."
        ),
        examples=[175000.0],
    )
    predicted_p10: float = Field(
        ...,
        description="10th-percentile prediction from the quantile model ($).",
        examples=[125000.0],
    )
    predicted_p50: float = Field(
        ...,
        description="Median (50th-percentile) prediction from the quantile model ($).",
        examples=[175000.0],
    )
    predicted_p90: float = Field(
        ...,
        description="90th-percentile prediction from the quantile model ($).",
        examples=[245000.0],
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

    # Premium-tier classifier head (Gap 1 Phase 1). Both fields are
    # optional because pre-Phase-1 artefacts did not ship a classifier;
    # the API degrades gracefully to ``None`` in that case.
    p_above_premium_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Binary-classifier probability that ``Annual Income`` exceeds "
            "``premium_threshold`` for this profile. The classifier is a "
            "separate XGBoost head trained alongside the quantile regressor "
            "on the same engineered feature matrix. ``None`` on pre-Phase-1 "
            "artefacts where no classifier is available."
        ),
        examples=[0.72],
    )
    premium_threshold: int | None = Field(
        default=None,
        description=(
            "Dollar threshold used by the classifier head (from "
            "``config.yaml::model.premium_threshold``). ``None`` on "
            "pre-Phase-1 artefacts."
        ),
        examples=[150000],
    )


class PredictBatchRequest(BaseModel):
    """Batch prediction request.

    Bulk callers pass up to 1,000 ``PredictRequest`` items in a single
    HTTP call. The API validates each item, consults the cache per-item,
    and scores any un-cached rows in a single vectorised
    ``model.predict`` call so per-request overhead is amortised across
    the batch. Rate-limited more aggressively than ``/predict`` to
    prevent a single caller from monopolising the model.
    """

    items: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of 1–1000 prediction requests. Validation is all-or-nothing — if any item fails domain validation the whole batch returns 422.",
    )


class PredictBatchResponse(BaseModel):
    """Batch prediction result, in the same order as the input items."""

    items: list[PredictResponse] = Field(
        ...,
        description="Quantile predictions in input order. One result per input item.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    dataset_rows: int
    version: str = __version__
    model_version: str = Field(
        default="unknown",
        description=(
            "Composite model provenance string of the form "
            "``{service_version}+{git_sha}.{data_sha256}`` emitted by "
            "``scripts/train_quantile.py`` and persisted in "
            "``models/model_metrics.json``. Operators can paste the "
            "``git_sha`` fragment into ``git show`` to recover the exact "
            "training code, and compare the ``data_sha256`` fragment "
            "against a local re-hash of ``data/cleaned_high_pay_data.csv`` "
            'to prove the training data matches. Returns ``"unknown"`` '
            "for pre-provenance artefacts."
        ),
    )


class MetaResponse(BaseModel):
    states: list[str]
    occupations: list[str]
    education_levels: list[str]
