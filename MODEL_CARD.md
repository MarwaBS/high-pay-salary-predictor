# Model Card — US High-Pay Salary Quantile Model

## Model Details

| Field | Value |
|---|---|
| **Name** | XGBoost multi-quantile salary predictor |
| **Version** | 2.0.0 |
| **Type** | Gradient-boosted quantile regression (XGBoost `reg:quantileerror`, alphas = [0.10, 0.50, 0.90]) |
| **Output** | P10 / P50 / P90 dollar predictions per request — not a point estimate |
| **Artefact** | `models/xgb_salary_quantile.ubj` (XGBoost native binary, no pickle) |
| **Training script** | `scripts/train_quantile.py` |
| **Config** | `config.yaml` |

> **Change in 2.0.0.** Previous versions framed the task as a point-estimate
> regression of individual income. Within the $100K+ cohort, individual
> variance is dominated by unobserved factors (equity, bonuses, tenure,
> specific employer), so a point estimator cannot produce a useful R² on
> this data. The 2.0.0 model instead returns a calibrated quantile interval
> (P10/P50/P90) and is scored on empirical coverage rather than squared error.

## Intended Use

Given a demographic and occupational profile within the **$100K+ US
cohort**, return a calibrated income range (P10, P50, P90) the worker
can use as a directional benchmark. Intended for exploratory analysis
and portfolio demonstration — **not** for employment decisions,
compensation benchmarking, or any consequential use.

## Training Data

| Source | Description |
|---|---|
| US Census ACS/CPS microdata | Individual income, demographics, education, occupation |
| BLS OEWS | State × occupation employment, location quotient, hourly/annual mean wage |

- **Population**: workers with reported annual income ≥ $100 K
- **Geography**: all 50 US states
- **Features used**: 10 (see below)

### ⚠️ Data-prep caveat

`high_pay_jobs_data_cleaning.ipynb` double-filters the cohort:
- BLS rows are kept only if `A_MEAN ≥ $100K` or `H_MEAN ≥ $48` (cell 14)
- Census rows are kept only if `INCTOT ≥ $100K` (cell 21)
- The two are then inner-joined on `(OCC_CODE, STATE)` (cell 9)

This truncation pre-removes most of the occupation-wage signal, which is
why any point estimator on this cohort tops out near R² ≈ 0.10. The
quantile model still produces useful output because the P10/P50/P90
spread itself is informative, but a fundamental improvement would require
re-prepping the data using the full Census dataset with a binary
`≥ $100K` classifier target. Tracked as a future enhancement.

## Features

Feature set is unchanged from v1.0.0 — only the training objective changed.

| Feature | Type | Source | Notes |
|---|---|---|---|
| `Age` | int | Census | |
| `Education_Ord` | int (1–4 ordinal) | Census → `config.yaml` mapping | |
| `Gender_Bin` | binary (1=Male, 0=Female) | Census | |
| `Region_Code` | int (0–3) | Derived from state → US Census region | |
| `Employment` | float | BLS OEWS | |
| `Location Quotient` | float | BLS OEWS | |
| `Jobs per 1000` | float | BLS OEWS | |
| `Hourly Mean` | float | BLS OEWS | `Annual Mean Wage` dropped (VIF ≈ 5.4×10⁸, corr=0.9999) |
| `Occ_Mean_Income` | float | Derived from **training split only** | Fixed target encoding — no leakage |
| `State_Mean_Income` | float | Derived from **training split only** | Fixed target encoding — no leakage |

## Training Objective

```
objective = "reg:quantileerror"
quantile_alpha = [0.10, 0.50, 0.90]
target = log1p(Annual Income)
```

A single XGBoost model outputs all three quantiles simultaneously. At
inference the raw `(n, 3)` output is back-transformed via `expm1` into
dollar space. Hyperparameters are inherited from `config.yaml` — no HPO
re-run was needed since the objective change drives the improvement.

## Performance

Measured on a held-out 20% test split (2,051 rows). Retrain date
shown in `models/model_metrics.json::train_date`.

### Quantile metrics (the real SLO)

| Metric | Value | What it means |
|---|---|---|
| 80% empirical coverage | **~0.77** | Fraction of test targets that fall inside `[P10, P90]`. Target = 0.80 ± 0.05. |
| Median PI width | ~$112K | Typical spread of the 80% interval in dollar space. |
| Quantile crossings | **0** | Number of test rows where P10 > P50 or P50 > P90. Must be zero. |
| P10 pinball loss | ~$5.5K | Quantile loss at α=0.10. |
| P50 pinball loss | ~$25K | Quantile loss at α=0.50 (equals `0.5 × MAE`). |
| P90 pinball loss | ~$21K | Quantile loss at α=0.90. |

### Point-estimate metrics (backward compat, P50 column)

| Metric | Value | Honesty note |
|---|---|---|
| Test R² (P50) | ~0.026 | P50 under a quantile objective is the median-minimiser, not the mean-minimiser, so R² (which scores means) is a weak fit-statistic for this model. The real SLO is quantile coverage above. |
| Test MAE | ~$50K | |
| Test RMSE | ~$108K | |
| CV R² (5-fold, train only, dollar space) | ~0.029 ± 0.018 | Close to test R² — no overfitting, no space mismatch. |

### CV alignment

CV is computed **only on the training set**, in **dollar space**, using
a fresh fold model — exactly the same space as the test metric above.
This is enforced by `tests/test_pipeline.py::test_saved_cv_matches_test`.

## Subgroup Performance

Per-group P50 evaluation is tracked in `models/model_metrics.json::subgroup_metrics`
when produced by `scripts/train_model.py` (the HPO trainer). The lean
`scripts/train_quantile.py` skips subgroup logging for speed; re-run
the HPO trainer for a refresh.

The quantile reframe does not directly close the subgroup gap in this
dataset because the data-prep truncation affects both subgroups. The
gap is an argument for the data-prep rewrite follow-up, not a model tweak.

## Prediction Interval

The API endpoint `POST /predict` and the Streamlit dashboard now return:

| Field | Description |
|---|---|
| `predicted_p10` | 10th-percentile salary prediction (low end of 80% PI) |
| `predicted_p50` | Median prediction (point estimate for back-compat) |
| `predicted_p90` | 90th-percentile salary prediction (high end of 80% PI) |
| `predicted_salary` | Alias for `predicted_p50`, kept for v1 clients |
| `prediction_interval_low` / `prediction_interval_high` | Same as `p10` / `p90` |

Quantile crossings (P10 > P90) are clamped defensively inside
`api/inference.build_response`, so clients never see an inverted range
even if XGBoost emits one at a decision boundary.

## Limitations and Biases

1. **Binary gender**: the training data contains only "Male" / "Female"
   labels from Census CPS coding. Non-binary identities are not
   represented. The model cannot make predictions for genders outside
   this binary.

2. **Truncated cohort**: as noted in the data-prep caveat, the model is
   trained on a double-filtered slice of the population. It is
   well-defined *within* the $100K+ cohort but cannot answer "will
   this person earn more than $100K" — use a different model for that.

3. **Geographic coverage**: US data only.

4. **Temporal drift**: BLS OEWS and Census data are point-in-time
   snapshots. The Redis-backed drift monitor (`/drift` endpoint)
   aggregates observations cluster-wide so drift alerts are reliable
   across a multi-replica Deployment.

5. **Unobserved confounders**: equity compensation, bonuses, years of
   experience, specific employer, and negotiation history drive large
   income differences the model cannot capture. The quantile spread
   reflects this uncertainty honestly rather than pretending it away.

6. **Fairness**: group-level income disparities in the training data
   (by gender, region, occupation) are reflected in the quantile
   intervals. The model does not correct for historical discrimination
   embedded in wages. Subgroup calibration should be checked
   periodically — when the `quantile_coverage_80` metric is computed
   per subgroup, add an assertion to `tests/test_pipeline.py`.

## How to Retrain

```bash
# Quantile trainer (lean — no MLflow / Optuna)
python -m scripts.train_quantile

# OR the full trainer with HPO + MLflow logging (uses the old point
# estimator — kept for historical comparison)
python -m scripts.train_model --tune
```

Both scripts write artefacts to `models/` and metrics to
`models/model_metrics.json`. Test suite picks up changes automatically
— if the quantile coverage drifts outside `[0.72, 0.88]` or any
crossings appear, `tests/test_pipeline.py::test_saved_metrics_within_expected_range`
will fail loudly.
