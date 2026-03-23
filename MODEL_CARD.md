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
- **Features used**: 11 (see below)

## Features

| Feature | Type | Source |
|---|---|---|
| `Age` | int | Census |
| `Education_Ord` | int (1–4 ordinal) | Census → `config.yaml` mapping |
| `Gender_Bin` | binary (1=Male, 0=Female) | Census |
| `Region_Code` | int (0–3) | Derived from state → US Census region |
| `Employment` | float | BLS OEWS |
| `Location Quotient` | float | BLS OEWS |
| `Jobs per 1000` | float | BLS OEWS |
| `Hourly Mean` | float | BLS OEWS |
| `Annual Mean Wage` | float | BLS OEWS |
| `Occ_Mean_Income` | float | Derived: per-occupation mean income in dataset |
| `State_Mean_Income` | float | Derived: per-state mean income in dataset |

## Performance

| Metric | Value |
|---|---|
| Test R² | ~0.07–0.10 |
| 5-fold CV R² | ~0.07–0.10 ± 0.01 |
| Test MAE | ~$35 K–$40 K |
| Test RMSE | ~$70 K–$80 K |

**Why is R² low?**
Individual income within the $100 K+ cohort has extremely high within-occupation
variance driven by equity compensation, bonuses, tenure, and employer —
none of which are in the dataset. The model captures occupation- and state-level
income patterns reliably but cannot resolve individual-level variation. This is a
data-ceiling effect, not a modelling failure.

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
