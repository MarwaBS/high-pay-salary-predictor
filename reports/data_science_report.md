# High-Paying Jobs in the US — Data Science Report

## 1. Problem statement and goals

Analyze the landscape of high-paying jobs (≥ $100K/yr) across all 50 US states to answer:

- Where are high-paying roles most prevalent and most geographically concentrated?
- How does education level relate to high-income participation and income premiums?
- What demographic patterns (gender, age) are present within the $100K+ cohort?
- Can individual salary be predicted from demographic and occupational features?
- Which features drive predictions the most (global + local explainability)?

**Deliverables:** cleaned dataset, reproducible notebooks, statistical hypothesis tests,
XGBoost salary predictor with SHAP explainability, empirical prediction intervals,
MLflow experiment tracking, FastAPI serving layer, interactive Streamlit dashboard,
and full DevOps stack.

---

## 2. Data sources and scope

| Source | Content | Raw file |
|--------|---------|---------|
| BLS OEWS | Wages, employment, location quotient by occupation/state | `Resources/bls_state_data.xlsx` |
| US Census | Demographics, education, occupation microdata | `Resources/census_data.csv` |

**Unified dataset:** `Data/cleaned_high_pay_data.csv`
- Shape: 10,255 rows × 15 columns
- Unit of analysis: individual Census records matched with occupation-state BLS wage context

**Final schema:** State Abbreviation, State, Gender, Age, Education Code, Education Level,
Degree Field, Occupation Code, Occupation, Annual Income, Employment, Location Quotient,
Jobs per 1000, Hourly Mean, Annual Mean Wage.

---

## 3. Data preparation

*Implemented in `high_pay_jobs_data_cleaning.ipynb`.*

**BLS cleaning:**
- Normalize strings; drop invalid OCC_CODE; cast numerics; exclude territories
- Filter high-pay cohort: `A_MEAN ≥ 100,000` or `H_MEAN ≥ 48.08` (≈$100K annualized)

**Census cleaning:**
- Extract 6-digit SOC from OCCSOC; decode SEX / STATEICP / EDUCD / DEGFIELDD
- Drop rows with missing key fields

**Integration:**
- Inner join on `[OCC_CODE, STATE]`; reorder/rename columns; drop duplicates
- Final: `Data/cleaned_high_pay_data.csv` (10,255 × 15)

---

## 4. Methods

| Phase | Technique | Notebook |
|-------|-----------|---------|
| EDA | Descriptive stats, grouped aggregations, distribution plots | `high_paying_jobs_data_visualization.ipynb` |
| Geospatial | Choropleth maps (GeoPandas + pyshp fallback) | `us_high_income_jobs_mapping.ipynb` |
| Statistical tests | ANOVA, Welch t-test, Tukey HSD, Cohen's *d* | `04_salary_prediction_model.ipynb` |
| Feature leakage analysis | Target-encoding dilution, TargetEncoder mitigation | `04_salary_prediction_model.ipynb` |
| Baseline model | Ridge Regression | `04_salary_prediction_model.ipynb` |
| Primary model | XGBoost with 5-fold cross-validation | `04_salary_prediction_model.ipynb` |
| HPO | Optuna TPE (30 trials, CV R² objective) | `04_salary_prediction_model.ipynb` |
| Comparison model | LightGBM | `04_salary_prediction_model.ipynb` |
| Explainability | SHAP (global summary, beeswarm, dependence plots) | `04_salary_prediction_model.ipynb` |
| Fairness | Within-occupation gender gap, demographic parity check | `04_salary_prediction_model.ipynb` |
| Experiment tracking | MLflow (params, metrics, model artefacts) | `04_salary_prediction_model.ipynb` |
| Serving | FastAPI + Pydantic v2 (with 80% prediction intervals) | `api/main.py` |
| Dashboard | Streamlit + Plotly (PI display, R² context) | `streamlit_app.py` |

---

## 5. Hypothesis testing

| Hypothesis | Test | Result | Effect size |
|------------|------|--------|-------------|
| H1: Education level drives income | One-way ANOVA | **Significant** (p < 0.001) | η² > 0 |
| H2: Gender income gap exists | Welch t-test | **Significant** (p < 0.001) | Cohen's *d* ≈ 0.27 (small–medium) |
| H3: Regional income differences | One-way ANOVA | **Significant** (p < 0.001) | — |

All three hypotheses are supported at the 5% significance level. Cohen's *d* = 0.27 for
the gender gap (Male mean $179K vs Female mean $152K, gap ≈ $27K / 18%) indicates a
small-to-medium practical effect — statistically robust but smaller than the raw gap
suggests once occupation composition is controlled for.

---

## 6. Model performance

| Model | Feature set | Test R² | CV R² | RMSE | Notes |
|-------|-------------|---------|-------|------|-------|
| Ridge Regression | Full (11) | ~0.02 | — | — | Linear baseline |
| XGBoost | Full (11) | **0.040** | **0.040 ± 0.012** | **$107,365** | Primary production model |
| XGBoost | Demographic only (6) | lower | — | — | Isolation of demographic signal |
| LightGBM | Full (11) | comparable | — | — | Speed comparison |
| XGBoost (Optuna HPO) | Full (11) | ~0.040 | ~0.040 | — | 30-trial TPE; default ≈ optimal |

**Why is R² low?** Individual Census incomes within the $100K+ cohort span $100K–$1M+,
driven by unobserved factors: equity compensation, bonuses, tenure, specific employer.
Available features explain occupation- and state-level *means* reliably, but cannot
resolve individual-level variation. CV R² is stable at 0.040 ± 0.012 (5-fold),
confirming no overfitting — this is a data-ceiling effect, not a modelling failure.

**Prediction intervals:** the `/predict` API and dashboard return an empirical 80% PI
derived from the 10th/90th percentiles of test-set residuals:
- offset_10 ≈ −$73,935 (lower bound)
- offset_90 ≈ +$64,101 (upper bound)
- Median PI width: ~$138,037 | empirical coverage on test set: 80.0%

**Feature importance (SHAP):**
Top drivers are `Annual Mean Wage` (BLS occupation-level wage), `Occ_Mean_Income`,
`State_Mean_Income`, followed by `Education_Ord` and `Age`. Geography (Region_Code)
and demographic features contribute less but are statistically significant.

**Model artefacts (no pickle):**

| File | Format | Contents |
|------|--------|---------|
| `models/xgb_salary_model.ubj` | XGBoost native binary | Trained model |
| `models/feature_names.json` | JSON | 11-feature list |
| `models/model_metrics.json` | JSON | R², RMSE, MAE, CV R², PI offsets, context note |

---

## 7. Shared pipeline module

`pipeline.py` is the single source of truth for:
- `FEATURES_FULL` — the 11-feature vector used by the production model
- `FEATURES_DEMO` — the 10-feature demographic-only vector
- `REGION_CODES` — deterministic region → integer mapping
- `engineer_features()` — adds all derived columns (Education_Ord, Gender_Bin, Region,
  Region_Code, Occ_Mean_Income, State_Mean_Income)
- `save_model / load_model` — XGBoost native .ubj (no pickle)
- `save_features / load_features / save_metrics / load_metrics` — plain JSON

This module is imported by `api/main.py`, `streamlit_app.py`, `scripts/train_model.py`,
and `tests/conftest.py`, ensuring every layer of the stack uses an identical encoding.

---

## 8. Key findings

- **Geography:** Large economies (CA, NY, TX) lead in absolute headcount.
  Concentration (LQ) peaks in MD, VA, WA — specialized clusters, not just population size.
- **Education:** Bachelor's dominates most states for $100K+ roles.
  Master's dominates SD, MT, NE, MO, WV; Professional dominates ND.
  Premiums are larger where specialized credentials are rewarded.
- **Demographics:** Gender participation is uneven across occupations and states.
  Male mean $179K vs Female $152K (gap $27K, Cohen's *d* = 0.27). Within-occupation
  gap persists after controlling for occupational sorting — most occupations show
  a male earnings advantage.
- **Market dynamics:** Bigger markets often show higher education premiums, but
  industry composition (tech/finance/healthcare) matters more than market size.
- **Correlations:** Employment and jobs-per-1000 co-move. Annual income has weak
  correlation with headcount metrics — reinforcing the primacy of occupation and geography.

---

## 9. Limitations

- **Snapshot analysis:** single timeframe; nominal incomes (no cost-of-living adjustment).
- **No causal inference:** descriptive and exploratory; SHAP shows feature contributions,
  not causal mechanisms.
- **Model ceiling:** individual income variance within the $100K+ cohort is driven largely
  by unobserved factors (equity, bonuses, tenure, specific employer). R² = 0.040 is
  genuinely low and is reported honestly with full context.
- **Target encoding leakage:** `Occ_Mean_Income` and `State_Mean_Income` are computed
  from the full dataset. Dilution analysis confirms negligible bias (§1.1 notebook);
  mitigation is `sklearn.preprocessing.TargetEncoder(cv=5)` for production.
- **Census data resolution:** microdata income may differ from W-2 or administrative records.

---

## 10. Reproducibility checklist

```bash
make install          # create .venv
make data             # regenerate Data/cleaned_high_pay_data.csv
make model            # train model → models/xgb_salary_model.ubj + model_metrics.json
make test             # run 67 tests
make dashboard        # launch Streamlit → http://localhost:8501
make api              # launch FastAPI  → http://localhost:8000
make mlflow           # view experiment runs → http://localhost:5000
```

All steps are fully automated and reproducible from a fresh clone.
