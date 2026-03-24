# Model Card — US High-Pay Salary Predictor

## Model Details

| Field | Value |
|---|---|
| **Name** | XGBoost Salary Predictor |
| **Version** | 1.0.0 |
| **Type** | Gradient-boosted regression (XGBoost `reg:squarederror`) |
| **Artefact** | `models/xgb_salary_model.ubj` (XGBoost native binary, no pickle) |
| **Training script** | `scripts/train_model.py` |
| **Config** | `config.yaml` |

## Intended Use

Predict annual income for US workers in high-paying occupations (≥ $100 K/yr).
Intended for **exploratory analysis and portfolio demonstration** — not for
employment decisions, compensation benchmarking, or any consequential use.

## Training Data

| Source | Description |
|---|---|
| US Census ACS/CPS microdata | Individual income, demographics, education, occupation |
| BLS OEWS | State × occupation employment, location quotient, hourly/annual mean wage |

- **Population**: workers with reported annual income ≥ $100 K
- **Geography**: all 50 US states
- **Features used**: 10 (see below)

## Features

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

> **Collinearity note**: `Annual Mean Wage` is a near-perfect linear transform of
> `Hourly Mean` (× 2080 hours). Including both distorts feature importance and
> wastes a feature slot. It was removed after VIF analysis (VIF = 5.44×10⁸).

> **Target-encoding note**: `Occ_Mean_Income` and `State_Mean_Income` are computed
> from the **training set only** (after the 80/20 split) and saved as
> `models/group_means.json`. At inference the API loads these saved values,
> eliminating the leakage that would arise from computing means on the full dataset.

## Hyperparameters

Tuned via 30-trial **Optuna TPE** search on log1p-transformed target with
fixed training-set group-mean encoding.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 169 |
| `max_depth` | 3 |
| `learning_rate` | 0.045 |
| `subsample` | 0.741 |
| `colsample_bytree` | 0.829 |
| `reg_lambda` (L2) | 9.88 |
| Target transform | `log1p(Annual Income)` |

## Performance

| Metric | Value | Notes |
|---|---|---|
| Test R² | 0.077 | Dollar space after `expm1` back-transform |
| 5-fold CV R² | 0.155 ± 0.020 | Computed in **log space** (not directly comparable to test R²) |
| Test MAE | $52,279 | Dollar space |
| Test RMSE | $105,232 | Dollar space |
| 80% PI width | $127,950 | Empirical from 10th/90th percentile of test residuals |
| PI coverage | 80.0% | Verified on held-out test set |

> **Log-transform note**: the model is trained on `log1p(Annual Income)` and
> predicts in log space. All callers must apply `numpy.expm1()` to the raw
> output to recover dollar predictions. Test R² is computed in dollar space after
> back-transformation. CV R² is in log space for training stability — both are
> reported for full transparency, but they are not directly comparable.

**Why is R² moderate?**
Individual income within the $100 K+ cohort has extremely high within-occupation
variance driven by equity compensation, bonuses, tenure, and employer —
none of which are in the dataset. The model captures occupation- and state-level
income patterns reliably but cannot resolve individual-level variation. This is a
data-ceiling effect, not a modelling failure. The log transform and Optuna HPO
push R² from 0.040 → 0.077 (+93%).

## Subgroup Performance

Evaluated on held-out test set (20%); dollar-space metrics after `expm1`.

| Subgroup | n | R² | MAE |
|---|---|---|---|
| **Male** | 1,218 | 0.071 | $61,019 |
| **Female** | 833 | 0.023 | $39,499 |
| **Northeast** | 520 | 0.097 | $54,280 |
| **Midwest** | 248 | 0.088 | $58,009 |
| **West** | 698 | 0.053 | $56,225 |
| **South** | 585 | 0.067 | $43,362 |

The lower female R² reflects smaller sample size and higher income variance
within the female sub-cohort. Gender is encoded as binary (see Limitations).

## Permutation Importance (Top 5)

Computed over 50 repeats on the held-out test set (mean decrease in R², dollar space).

| Rank | Feature | Mean ΔR² | Std |
|---|---|---|---|
| 1 | `Age` | 0.112 | ±0.013 |
| 2 | `Occ_Mean_Income` | 0.085 | ±0.010 |
| 3 | `Gender_Bin` | 0.030 | ±0.005 |
| 4 | `Education_Ord` | 0.010 | ±0.002 |
| 5 | `Hourly Mean` | 0.008 | ±0.003 |

Note: `Gender_Bin` ranks 3rd by permutation importance, confirming a structural
income signal beyond occupation and wage-level controls.

## Prediction Interval

The API and dashboard expose an empirical **80% prediction interval** derived from
the 10th and 90th percentiles of test-set residuals. The interval is approximate
(income residuals are right-skewed / heteroscedastic) and is clearly labelled as
such in all user-facing surfaces. Treat it as a directional range, not a precise bound.

## Limitations and Biases

1. **Binary gender**: the training data contains only "Male" / "Female" labels
   (from Census CPS coding). Non-binary identities are not represented; gender
   is encoded as a binary feature (`Gender_Bin`). The model cannot make
   predictions for genders outside this binary.

2. **Income floor**: the dataset is filtered to ≥ $100 K. Predictions below that
   threshold are extrapolation outside the training distribution.

3. **Geographic coverage**: the model was trained on US data only and is not
   valid for other countries.

4. **Temporal drift**: BLS OEWS and Census data are point-in-time snapshots.
   Predictions may degrade as the labour market changes.

5. **Unobserved confounders**: equity compensation, bonuses, years of experience,
   specific employer, and negotiation history drive large income differences that
   the model cannot capture.

6. **Fairness**: group-level income disparities in the training data (by gender,
   region, occupation) are reflected in predictions. The model does not correct
   for historical discrimination embedded in wages.

## How to Retrain

```bash
make model          # trains and saves all artefacts to models/
# or
python scripts/train_model.py
```

All hyperparameters are controlled via `config.yaml` under the `model:` key.
Metrics are written to `models/model_metrics.json` after each training run.
