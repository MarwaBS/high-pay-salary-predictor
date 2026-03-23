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
make test         # run the full 64-test suite
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
| `make clean` | Remove `models/*.pkl`, `__pycache__`, `.pytest_cache`, `htmlcov/` |
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

## Portfolio highlights

- **XGBoost salary prediction:** cross-validated model with SHAP global + dependence explainability.
- **Statistical rigor:** ANOVA (education × income), Welch t-test (gender pay gap), and Tukey HSD post-hoc tests validate EDA findings.
- **MLflow experiment tracking:** all model runs (Ridge, XGBoost full, XGBoost demographic, LightGBM) logged with params, metrics, and model artefacts. Compare in the UI with `make mlflow`.
- **Production API:** FastAPI + Pydantic v2 with `/health`, `/meta`, `/predict` endpoints and Swagger docs.
- **Interactive dashboard:** Streamlit with Plotly choropleth, salary predictor widget, and model insights tab.
- **Shared pipeline module:** `pipeline.py` is the single source of truth for feature engineering and feature constants — consumed by the API, dashboard, training script, and all tests (zero duplication).
- **Full DevOps stack:** multi-stage Docker build, Docker Compose, GitHub Actions CI (lint + test on Python 3.10 and 3.11), Makefile, `pyproject.toml`, pre-commit hooks.
- **64 tests** across config validation, data schema, feature engineering, pipeline constants, and API endpoints.

---

## Data

**Sources:**
- U.S. Bureau of Labor Statistics (BLS): state-level Occupational Employment and Wage Statistics (OEWS)
- U.S. Census Bureau: microdata — demographics, education, occupation

**Cleaned dataset:** `Data/cleaned_high_pay_data.csv` — 10,255 rows × 15 columns

**Key fields:** Occupation, Annual Income, Education Level, Gender, State Abbreviation, Annual Mean Wage, Location Quotient, Employment, Jobs per 1000.

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

Top occupations by average income
![Top 10 Occupations by Average Income](./Images/Top_Occupations_Avg_Income.png)
- Specialized roles dominate the top earnings within the $100K+ cohort.

Average income by education level
![Average Income by Education Level](./Images/Average_Income_by_Education_Level.png)
- Higher education levels correlate with higher average incomes.

Salary distributions for top occupations
![Salary Distribution for Top Occupations](./Images/Top_10_Salary_Distribution.png)
- Some roles show tight pay clusters; others exhibit wider dispersion.

Correlation among numeric features
![Correlation Heatmap](./Images/Correlation_Annual_Income.png)
- Employment and jobs-per-1000 co-move; annual income has weak links to headcount metrics.

Age vs annual income
![Age vs Income](./Images/Age_Annual_Income.png)
- Income rises early and plateaus later.

Gender distribution across top occupations
![Gender by Occupation](./Images/Gender_Distribution_Occupations.png)

Gender distribution across top states
![Gender by State](./Images/Gender_Distribution_States.png)

Gender distribution by education (within $100K+)
![Gender by Education](./Images/Gender_Education_Distribution.png)

Distribution of $100K+ individuals by state
![High-Paid Individuals by State](./Images/High_Paid_Individuals_by_State.png)

Average income by state
![Average Income by State](./Images/Average_Highest_Income_state_Viz.png)

Location Quotient by state
![LQ by State](./Images/High_Paying_Jobs_LQ_Distribution_Viz.png)

Dominant education level by state
![Dominant Education by State](./Images/Dominant_education_by_state_Viz.png)

Education–income premium by state
![Education Premium by State](./Images/Education_Income_Premiums_by_State_Viz.png)

Market size vs education premium
![Market Size vs Premium](./Images/Market_Size_Income_Premium_Analysis_Viz.png)

Average income by US Census region
![Regional Patterns](./Images/Regional_Patterns_Analysis_Viz.png)

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
│   ├── conftest.py                            # ★ Shared session-scope fixtures (cfg, df, df_engineered)
│   ├── test_pipeline.py                       # ★ Config, schema, feature engineering, ML (48 tests)
│   └── test_api.py                            # ★ API endpoints, validation, prediction (16 tests)
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
├── models/                                    # Saved ML model artefacts (generated)
│   ├── xgb_salary_model.pkl
│   └── feature_names.pkl
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
- **64 tests:** config, data schema, feature engineering, pipeline constants, and API endpoints.
- **CI/CD:** GitHub Actions runs lint + tests on every push (Python 3.10 and 3.11).
- **Exact lock file:** `requirements-lock.txt` pins all 133 transitive dependencies.
- **Pre-commit hooks:** ruff linting/formatting and nbstripout run automatically on every commit.
