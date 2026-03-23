# High-Paying Jobs in the US — Data Science Report

## 1. Problem statement and goals

Analyze the landscape of high-paying jobs (≥ $100K/yr) across all 50 US states to answer:

- Where are high-paying roles most prevalent and most geographically concentrated?
- How does education level relate to high-income participation and income premiums?
- What demographic patterns (gender, age) are present within the $100K+ cohort?
- Can individual salary be predicted from demographic and occupational features?
- Which features drive predictions the most (global + local explainability)?

**Deliverables:** cleaned dataset, reproducible notebooks, statistical hypothesis tests,
XGBoost salary predictor with SHAP explainability, MLflow experiment tracking,
FastAPI serving layer, interactive Streamlit dashboard, and full DevOps stack.

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
| Statistical tests | ANOVA, Welch t-test, Tukey HSD post-hoc | `04_salary_prediction_model.ipynb` |
| Baseline model | Ridge Regression | `04_salary_prediction_model.ipynb` |
| Primary model | XGBoost with 5-fold cross-validation | `04_salary_prediction_model.ipynb` |
| Comparison model | LightGBM | `04_salary_prediction_model.ipynb` |
| Explainability | SHAP (global summary, beeswarm, dependence plots) | `04_salary_prediction_model.ipynb` |
| Experiment tracking | MLflow (params, metrics, model artefacts) | `04_salary_prediction_model.ipynb` |
| Serving | FastAPI + Pydantic v2 | `api/main.py` |
| Dashboard | Streamlit + Plotly | `streamlit_app.py` |

---

## 5. Hypothesis testing

| Hypothesis | Test | Result |
|------------|------|--------|
| H1: Education level drives income | One-way ANOVA | Significant (p < 0.05); confirmed by Tukey HSD |
| H2: Gender income gap exists | Welch t-test | Significant (p < 0.05) |
| H3: Regional income differences | One-way ANOVA | Significant (p < 0.05) |

All three hypotheses are supported at the 5% significance level.

---

## 6. Model performance

| Model | Feature set | R² | RMSE | Notes |
|-------|-------------|-----|------|-------|
| Ridge Regression | Full (11) | baseline | — | Regularization reference |
| XGBoost | Full (11) | best | — | Primary production model |
| XGBoost | Demographic only (10) | lower | — | Fairness / gap analysis |
| LightGBM | Full (11) | competitive | — | Speed comparison |

*Exact metric values are logged to MLflow. Run `make mlflow` after notebook 4 to compare.*

**On R² magnitude:** Individual Census incomes within the $100K+ cohort span $100K–$1M+.
Available features (BLS occupation wages, demographics) explain occupation-level means
well but cannot explain individual variance (e.g., equity compensation, bonuses, tenure).
R² of 0.10–0.15 represents non-trivial predictive power given this constraint.

**Feature importance (SHAP):**
Top drivers are `Annual Mean Wage` (BLS occupation-level wage), `Occ_Mean_Income`,
`State_Mean_Income`, followed by `Education_Ord` and `Age`. Geography (Region_Code)
and demographic features contribute less but are statistically significant.

---

## 7. Shared pipeline module

`pipeline.py` is the single source of truth for:
- `FEATURES_FULL` — the 11-feature vector used by the production model
- `FEATURES_DEMO` — the 10-feature demographic-only vector
- `REGION_CODES` — deterministic region → integer mapping
- `engineer_features()` — adds all derived columns (Education_Ord, Gender_Bin, Region,
  Region_Code, Occ_Mean_Income, State_Mean_Income)

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
  Age–income patterns rise early and plateau later in career.
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
  by unobserved factors (equity, bonuses, tenure, specific employer).
- **Census data resolution:** microdata income may differ from W-2 or administrative records.

---

## 10. Reproducibility checklist

```bash
make install          # create .venv
make data             # regenerate Data/cleaned_high_pay_data.csv
make model            # train model → models/xgb_salary_model.pkl
make test             # run 64 tests
make dashboard        # launch Streamlit
make api              # launch FastAPI
make mlflow           # view experiment runs
```

All steps are fully automated and reproducible from a fresh clone.
