# High-Paying Jobs in the US — End-to-End Data Science Case Study

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MarwaBS/High_pay_Analysis_us/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/High_pay_Analysis_us/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](streamlit_app.py)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](04_salary_prediction_model.ipynb)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)](04_salary_prediction_model.ipynb)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](04_salary_prediction_model.ipynb)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](api/main.py)
[![Tests](https://img.shields.io/badge/Tests-77%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-75%25%2B-green)](pyproject.toml)

## Key Findings

> **For senior interviewers and reviewers — findings first.**

| Finding | Detail |
|---|---|
| **Gender pay gap** | Male earners average ~$30K more than female peers within the same occupation and state (Cohen's *d* ≈ 0.27, statistically significant). Gap persists after controlling for education and region. |
| **Age is the top predictor** | Permutation importance: Age accounts for the largest unique drop in R² (ΔR²=0.112), outranking occupation and BLS wage signals. |
| **Education premium** | Each ordinal step in education (Bachelor's → Master's → Professional → Doctoral) yields a measurable income increment; the jump from Bachelor's to Doctoral is ~$45K in median terms. |
| **Regional disparity** | Northeast workers earn the most (highest R²=0.097 predictability); model explains substantially less variance in the South and Midwest. |
| **Log-transform unlocks model signal** | Switching the target to `log1p(Annual Income)` raised R² from 0.040 → 0.077 (+93%) by normalising the right-skewed income distribution. |
| **Target-encoding leakage fixed** | `Occ_Mean_Income` and `State_Mean_Income` are now computed from the training split only, eliminating the leakage present in earlier versions. |

---

A **production-grade, end-to-end data science pipeline** analyzing high-paying jobs (≥ $100K/yr) across all 50 US states, integrating **Bureau of Labor Statistics (BLS) OEWS** and **US Census** microdata. The project covers the complete ML lifecycle: raw data ingestion → cleaning → EDA → geospatial mapping → XGBoost salary prediction with SHAP explainability → MLflow experiment tracking → FastAPI serving → interactive Streamlit dashboard → Docker deployment.

**Keywords:** salary prediction, XGBoost, SHAP, MLflow, FastAPI, Streamlit, BLS OEWS, US Census, data science portfolio, income inequality analysis, feature engineering, CI/CD

---

### Dashboard preview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  💼 High-Paying Jobs in the US                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  Sidebar filters:          │  Overview  │  Geographic  │  Predictor  │  │
│  • Region(s)               │                                            │
│  • Education Level(s)      │  10,255 records  │  $168K avg  │  CA  │   │
│  • Income range ($)        │                                            │
│                            │  [Top 15 Occupations bar chart]            │
│                            │  [Avg Income by Education bar chart]       │
│                            │  [Gender distribution violin plot]         │
│                            │  [US choropleth — income / LQ / count]     │
│                            │  [Salary predictor form → live estimate]   │
│                            │  [SHAP feature importance + residuals]     │
└─────────────────────────────────────────────────────────────────────────┘
```

> **To see the live dashboard:** run `make dashboard` or `docker compose up --build`,
> then open [http://localhost:8501](http://localhost:8501).

---

## Pipeline overview

The project is organized across four notebooks and two deployable services:

| Notebook | Purpose |
|----------|---------|
| `high_pay_jobs_data_cleaning.ipynb` | Data integration & cleaning (BLS + Census → single dataset) |
| `high_paying_jobs_data_visualization.ipynb` | EDA: distributions, rankings, correlations |
| `us_high_income_jobs_mapping.ipynb` | Geospatial: choropleth maps by state |
| `04_salary_prediction_model.ipynb` | **ML: XGBoost + LightGBM + SHAP + statistical tests + MLflow** |

All figures are saved automatically to `Images/` at 300 DPI.

---

## Quickstart — one command

```bash
make install      # create .venv and install all dependencies
make test         # run the full 77-test suite (unit + integration)
make dashboard    # Streamlit dashboard → http://localhost:8501
make api          # FastAPI server    → http://localhost:8000
make docker       # build + start both services via Docker Compose
make mlflow       # MLflow tracking UI → http://localhost:5000
```

**All `make` targets:**

| Target | What it does |
|--------|-------------|
| `make install` | Create `.venv`, install `requirements.txt` |
| `make data` | Re-run cleaning notebook → `Data/cleaned_high_pay_data.csv` |
| `make model` | Train XGBoost model via `scripts/train_model.py` → `models/` |
| `make test` | Run all 64 pytest tests with `-v` |
| `make test-fast` | Same, quiet output |
| `make coverage` | Tests + HTML coverage report in `htmlcov/` |
| `make lint` | `ruff check` (fast linter, replaces flake8) |
| `make format` | `ruff format` (opinionated auto-formatter, Black-compatible) |
| `make type-check` | `mypy` static type checker on `api/`, `pipeline.py`, `scripts/` |
| `make dashboard` | Streamlit on port 8501 |
| `make api` | FastAPI + uvicorn on port 8000 |
| `make docker` | `docker compose up --build` |
| `make mlflow` | MLflow tracking UI on port 5000 |
| `make clean` | Remove `models/`, `__pycache__`, `.pytest_cache`, `htmlcov/` |
| `make clean-all` | `clean` + delete `.venv` |

---

## Manual setup (no make)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
pre-commit install                 # install git quality hooks
```

---

## Model performance

| Metric | Value | Notes |
|--------|-------|-------|
| Test R² | **0.077** | Dollar space after `expm1` back-transform; +93% vs. linear baseline (0.040) |
| CV R² (5-fold, log space) | **0.155 ± 0.020** | Stable; confirms no overfitting |
| RMSE | **$105,232** | On held-out 20% test set |
| MAE | **$52,279** | Median absolute error |
| 80% PI width | **~$128K** | Empirical from residual 10th/90th pct |
| Train / test | 8,204 / 2,051 | random_state=42 |

> **Why is R² moderate?** Census individual income within the $100K+ cohort has extremely high within-occupation variance driven by unobserved factors (equity compensation, bonuses, tenure, specific employer). The available BLS + demographic features explain occupation- and state-level income *patterns* reliably but cannot resolve individual-level variation. This is a data-ceiling effect, not a modelling failure. Switching to a `log1p` target (see below) and Optuna HPO pushed R² from 0.040 → 0.077 (+93%); 5-fold CV is stable (±0.020), confirming no overfitting.

> **Note on CV vs test R²:** CV R² (0.155) is computed in **log space** for training stability; test R² (0.077) is computed in **dollar space** after `expm1` back-transformation. They measure different things and are not directly comparable — both are reported for full transparency.

**Prediction intervals:** the `/predict` API endpoint and dashboard return an empirical 80% prediction interval derived from the 10th/90th percentiles of test-set residuals (dollar space). Offsets: −$49,920 (lower) / +$78,030 (upper). Intervals are wider than ±RMSE because income residuals are right-skewed.

---

## Portfolio highlights

- **Log-transform + Optuna HPO:** `log1p(Annual Income)` target with 30-trial TPE search (n_estimators=169, max_depth=3, lr=0.045, reg_lambda=9.88) pushed R² from 0.040 → 0.077 (+93%) and MAE from ~$59K → ~$52K.
- **Target-encoding leakage eliminated:** `Occ_Mean_Income` and `State_Mean_Income` computed from the **training split only**, saved as `models/group_means.json`, loaded at API startup — confirmed leak-free by a dedicated integration test.
- **Collinearity removal:** `Annual Mean Wage` dropped after VIF analysis (VIF = 5.44×10⁸, corr = 0.9999 with `Hourly Mean`). Feature set reduced 11 → 10 with cleaner importance scores.
- **Permutation importance:** 50-repeat permutation importance reveals Age (ΔR²=0.112) and Occ_Mean_Income (ΔR²=0.085) dominate — more trustworthy than gain-based importance for correlated features.
- **Subgroup fairness analysis:** R² and MAE computed for Gender (Male/Female) and Region (4 US Census regions) on the held-out test set. Male R²=0.071 vs Female R²=0.023; Northeast R²=0.097 vs South R²=0.067 — disparities documented and visualized.
- **Statistical rigor:** ANOVA (education × income), Welch t-test (gender pay gap Cohen's *d*=0.27, *p*<0.001), and Tukey HSD post-hoc tests validate EDA findings.
- **MLflow experiment tracking:** params, metrics (R², RMSE, MAE, CV, subgroup), and model artefact logged per run. Compare runs with `make mlflow`.
- **Production API:** FastAPI + Pydantic v2 with `/health`, `/meta`, `/predict` (returns salary + 80% PI + percentile + group benchmarks). Saved group means loaded at startup for consistent inference encoding. Sync route — no event-loop blocking.
- **Interactive dashboard:** Streamlit with 4 tabs — Overview, Geographic choropleth, Salary Predictor, and Model Insights (gain importance, permutation importance, subgroup R² charts, residual + actual-vs-predicted plots).
- **Shared pipeline module:** `pipeline.py` is the single source of truth — consumed by API, dashboard, training script, and all 77 tests. Zero duplication.
- **No pickle:** model stored as XGBoost native binary (`.ubj`); all other artefacts as plain JSON — portable, auditable, language-agnostic.
- **Full DevOps stack:** multi-stage Docker build (non-root user, health checks, resource limits), Docker Compose, GitHub Actions CI (pip-audit CVE scan + lint + test on Python 3.10 and 3.11), Dependabot, Makefile, `pyproject.toml`, pre-commit hooks.
- **77 tests** across unit (config, data schema, feature engineering, model prediction) and integration (leakage proof, round-trip group-means persistence, end-to-end R², PI sign check).

---

## Data

**Sources:**
- U.S. Bureau of Labor Statistics (BLS): state-level Occupational Employment and Wage Statistics (OEWS)
- U.S. Census Bureau: microdata — demographics, education, occupation

**Cleaned dataset:** `Data/cleaned_high_pay_data.csv` — 10,255 rows × 15 columns

**Key fields:** Occupation, Annual Income, Education Level, Gender, State Abbreviation, Hourly Mean, Location Quotient, Employment, Jobs per 1000.
(`Annual Mean Wage` is in the raw dataset but was dropped from model features — VIF = 5.44×10⁸ collinearity with `Hourly Mean`.)

Data are used for educational and analytical purposes only. Consult each provider's terms for reuse.

---

## Data cleaning and preparation

Implemented in `high_pay_jobs_data_cleaning.ipynb`:

**BLS cleaning:**
- Normalize strings; strip whitespace and title-case state/occupation names
- Remove hyphens from OCC_CODE; drop invalid codes
- Convert numerics; drop missing values; keep 50 US states only
- Define "high-paying" cohort: `A_MEAN ≥ 100,000` or `H_MEAN ≥ 48.08` (≈ $100K annualized)

**Census cleaning:**
- Extract 6-digit SOC from OCCSOC (zero-padded); decode SEX / STATEICP / EDUCD / DEGFIELDD
- Drop rows with missing key fields

**Integration:**
- Inner join on `[OCC_CODE, STATE]`
- Reorder and rename columns to a tidy schema; remove duplicates
- Final output: `Data/cleaned_high_pay_data.csv` (10,255 × 15)

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install git quality hooks
pre-commit install

# 3. Run EDA notebooks (open in Jupyter)
jupyter notebook

# 4. Train the model
python scripts/train_model.py

# 5. Launch the interactive dashboard
streamlit run streamlit_app.py

# 6. Run the test suite
pytest tests/ -v
```

### Run with Docker (one command, no Python setup needed)

```bash
docker compose up --build
# Dashboard → http://localhost:8501
# API docs  → http://localhost:8000/docs
docker compose down
```

### Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

**Key API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — model loaded, dataset rows |
| `GET` | `/meta` | Valid states, occupations, education levels |
| `POST` | `/predict` | Salary prediction + percentile + group benchmarks |
| `GET` | `/docs` | Auto-generated Swagger UI |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state":"CA","occupation":"Software Developers","education_level":"Bachelor'\''s degree","gender":"Female","age":32}'
```

### MLflow experiment tracking

After running notebook 4, compare all model runs:
```bash
make mlflow   # open http://localhost:5000
```
Logged per run: model type, hyperparameters, R², RMSE, MAE, CV R² mean ± std, and model artefact.

---

## Findings with figures

**Top occupations by average income**
![Top 10 Occupations by Average Income](./Images/Top_Occupations_Avg_Income.png)
Chief Executives, Physicians, and Lawyers lead. STEM software roles cluster just below — occupation choice is the strongest signal in the dataset, outranking state, education, and demographics for predicting whether someone earns above the cohort median.

**Average income by education level**
![Average Income by Education Level](./Images/Average_Income_by_Education_Level.png)
Each ordinal step adds income: the Bachelor's → Doctoral gap is ~$45K in medians. However, within-tier variance is high — a Bachelor's-degree Software Engineer often out-earns a Doctoral-degree academic, confirming that education alone is insufficient and occupation context is necessary.

**Salary distributions for top occupations**
![Salary Distribution for Top Occupations](./Images/Top_10_Salary_Distribution.png)
Right-skewed distributions with long upper tails in every role — the primary justification for the `log1p` target transform. Surgeons and CEOs show the widest spread, driven by equity compensation and bonuses not captured in the dataset.

**Correlation among numeric features**
![Correlation Heatmap](./Images/Correlation_Annual_Income.png)
`Hourly Mean` and `Annual Mean Wage` show near-perfect correlation (r ≈ 0.9999). Both cannot coexist in a model — VIF confirms multicollinearity (5.44×10⁸). `Annual Mean Wage` was removed; `Hourly Mean` was retained. Annual Income shows weak correlation with BLS headcount metrics, confirming that individual income is driven by within-occupation factors not captured at the aggregate BLS level.

**Age vs annual income**
![Age vs Income](./Images/Age_Annual_Income.png)
Age is the **single strongest predictor** (permutation ΔR²=0.112, ranking above occupation and BLS wage signals). Income rises steeply from 22–40, plateaus 40–65. Age acts as a proxy for seniority, negotiating experience, and accumulated tenure — unobserved variables that the model captures indirectly.

**Gender distribution across top occupations**
![Gender by Occupation](./Images/Gender_Distribution_Occupations.png)
Male representation dominates in most high-paying occupations, with the largest gaps in Engineering and Executive roles. The composition gap partly explains the observed pay gap but Welch t-test confirms it persists *within* occupation-state cells (Cohen's *d* = 0.27, *p* < 0.001).

**Gender distribution across top states**
![Gender by State](./Images/Gender_Distribution_States.png)
Female representation in the $100K+ cohort is highest in DC, MD, and VA — states with large government/healthcare/education sectors where gender-pay gaps tend to be smaller than private-sector tech and finance.

**Gender distribution by education (within $100K+)**
![Gender by Education](./Images/Gender_Education_Distribution.png)
At every education tier, male workers outnumber female within the $100K+ cohort. The gap is smallest at the Doctoral level — consistent with academic/research roles having narrower pay dispersion — and largest at the Professional degree level (law, medicine).

**Distribution of $100K+ individuals by state**
![High-Paid Individuals by State](./Images/High_Paid_Individuals_by_State.png)
CA, NY, TX, and FL lead in absolute headcount (large populations). MD, VA, and WA punch above their weight on a per-capita basis, reflecting federal contractor and tech cluster concentration.

**Average income by state (bar)**
![Average Income by State](./Images/Average_Highest_Income_state_Viz.png)
New England and Mid-Atlantic states dominate average income. The model captures this through `Region_Code` and `State_Mean_Income` — Northeast R² (0.097) is the highest of the four regions, confirming regional signal is real and learnable.

**Location Quotient by state**
![LQ by State](./Images/High_Paying_Jobs_LQ_Distribution_Viz.png)
MD, VA, DC and WA show LQ > 1.5, meaning high-paying jobs are over-represented relative to national employment share. These states, not the largest by population, are the densest clusters of premium roles — an insight job seekers optimizing for salary should weight over absolute headcount.

**Dominant education level by state**
![Dominant Education by State](./Images/Dominant_education_by_state_Viz.png)
Bachelor's degree dominates most of the contiguous US for $100K+ earners. Master's is modal in select Midwest states; Professional degrees lead in ND. This geographic clustering reflects industry mix (oil & gas, agriculture, manufacturing) rather than education ROI differences.

**Education–income premium by state**
![Education Premium by State](./Images/Education_Income_Premiums_by_State_Viz.png)
The education premium varies 2–3× across states. High-LQ tech states (WA, CA) show lower marginal returns to advanced degrees — top-tier individual contributors without graduate degrees still earn at or above the state median. High-premium states tend to be smaller markets with concentrated professional services.

**Market size vs education premium**
![Market Size vs Premium](./Images/Market_Size_Income_Premium_Analysis_Viz.png)
Larger labor markets show a mild *negative* correlation with education premium — supporting the hypothesis that large, competitive markets (NYC, SF) compress the education signal and reward occupation/skill specificity instead.

**Average income by US Census region**
![Regional Patterns](./Images/Regional_Patterns_Analysis_Viz.png)
Northeast leads in mean ± std income. South and Midwest show lower means with wider distributions. The model's subgroup analysis confirms this: Northeast test R²=0.097, South R²=0.067 — the Northeast is the most predictable region because industry mix is more homogeneous within occupation cells.

## Map gallery (choropleths)

Average income by state (map)
![Average Income by State (Map)](./Images/Average_Highest_Income_state.png)

High-paying jobs distribution (map)
![High-Paying Jobs Distribution (Map)](./Images/High_Paying_Jobs_Distribution.png)

Location Quotient by state (map)
![LQ by State (Map)](./Images/High_Paying_Jobs_LQ_Distribution.png)

Dominant education by state (map)
![Dominant Education by State (Map)](./Images/Dominant_education_by_state.png)

Gender share overlays (map)
![Male % (Map)](./Images/Male_Percentage_state.png)
![Female % (Map)](./Images/Female_Percentage_state.png)

---

## Case interpretation and results

- **Geographic:** Large economies (CA, NY, TX) lead in absolute headcount. Concentration (LQ) peaks in MD, VA, WA — specialized clusters drive premium roles.
- **Education ROI:** Bachelor's degrees dominate most states for $100K+ roles. Master's is dominant in SD, MT, NE, MO, WV; Professional in ND.
- **Demographic:** Gender participation is uneven across occupations and states. Age–income patterns plateau later in career.
- **Market dynamics:** Bigger markets often pair with higher education premiums, but industry composition (tech / finance / healthcare) matters more than market size alone.
- **Correlations:** Employment and jobs-per-1000 move together. Annual income shows weak correlation with headcount — reinforcing the primacy of occupation and geography.

### Recommendations

**Job seekers:** Target states with strong concentration (LQ) for your field, not just volume. Align degree investments with target regions.

**Employers:** Calibrate compensation to regional wage dynamics. Recruit across clusters where talent density is highest.

**Policymakers:** Direct workforce and education funding toward regional specializations.

### Limitations

- Analysis covers a single timeframe; nominal incomes (no cost-of-living adjustment applied).
- No causal inference; descriptive and exploratory focus.
- Industry deep-dives and longitudinal trends would refine signals.

See `reports/data_science_report.md` for the full analyst-oriented narrative.

---

## Repository structure

```
High_pay_Analysis_us/
│
├── pipeline.py                                # ★ Single source of truth: FEATURES + engineer_features
│
├── Notebooks
│   ├── high_pay_jobs_data_cleaning.ipynb      # Pipeline: BLS + Census → cleaned CSV
│   ├── high_paying_jobs_data_visualization.ipynb  # EDA: distributions, rankings, correlations
│   ├── us_high_income_jobs_mapping.ipynb      # Geospatial: choropleth maps
│   └── 04_salary_prediction_model.ipynb       # ★ ML: XGBoost + SHAP + statistical tests + MLflow
│
├── streamlit_app.py                           # ★ Interactive dashboard (run with streamlit)
├── config.yaml                                # ★ All thresholds, paths, color palettes
├── Dockerfile                                 # ★ Multi-stage build: dashboard + api (non-root)
├── docker-compose.yml                         # ★ Two services: dashboard (8501) + api (8000)
├── Makefile                                   # ★ install / data / model / test / lint / format / clean
├── pyproject.toml                             # ★ Ruff, mypy, pytest, coverage configuration
├── .pre-commit-config.yaml                    # ★ ruff, nbstripout, file hygiene hooks
│
├── api/
│   ├── main.py                                # ★ FastAPI app: /health /meta /predict (logging, env CORS)
│   └── schemas.py                             # ★ Pydantic v2 request/response models
│
├── scripts/
│   └── train_model.py                         # ★ Standalone model training script (replaces Makefile one-liner)
│
├── tests/
│   ├── conftest.py                            # ★ Shared session-scope fixtures (cfg, df, group_means, df_engineered)
│   ├── test_pipeline.py                       # ★ Config, schema, feature engineering, model prediction (45 tests)
│   ├── test_api.py                            # ★ API endpoints, validation, prediction (22 tests)
│   └── test_integration.py                    # ★ Full pipeline path: split → group_means → engineer → predict (10 tests)
│
├── Data/                                      # Processed datasets (single source of truth)
│   ├── cleaned_high_pay_data.csv              #   10,255 rows × 15 cols
│   ├── bls_data.csv
│   └── census_data.csv
│
├── Resources/                                 # Raw source data
│   ├── bls_state_data.xlsx
│   └── census_data.csv
│
├── models/                                    # Saved ML model artefacts (generated, no pickle)
│   ├── xgb_salary_model.ubj                   #   XGBoost native binary (portable)
│   ├── feature_names.json                     #   Feature list (10 features)
│   ├── group_means.json                       #   Training-set occ/state means (leakage-free inference)
│   └── model_metrics.json                     #   R², RMSE, MAE, CV R², PI offsets, subgroup metrics, permutation importance
│
├── Images/                                    # Auto-saved figures (300 DPI)
│
├── reports/
│   └── data_science_report.md                 # Analyst-oriented narrative and findings
│
├── .github/workflows/
│   └── ci.yml                                 # ★ GitHub Actions: lint + test on Python 3.10 & 3.11
│
├── requirements.txt                           # Pinned runtime + dev dependencies
├── requirements-lock.txt                      # pip freeze — exact transitive deps for full reproducibility
├── CONTRIBUTING.md                            # Contribution guide
└── LICENSE                                    # MIT
```

---

## Reproducibility

- **Single source of truth:** all notebooks and services consume `Data/cleaned_high_pay_data.csv` and `pipeline.py`.
- **Config-driven:** thresholds, paths, and palette live in `config.yaml` — never hardcoded.
- **77 tests:** unit (config, data schema, feature engineering, model prediction) + integration (leakage proof, group-means round-trip, end-to-end R², PI sign check).
- **CI/CD:** GitHub Actions runs `pip-audit` CVE scan + lint + tests on every push (Python 3.10 and 3.11).
- **Dependabot:** weekly automated dependency and GitHub Actions version updates.
- **Exact lock file:** `requirements-lock.txt` pins all 133 transitive dependencies.
- **Pre-commit hooks:** ruff linting/formatting and nbstripout run automatically on every commit.
