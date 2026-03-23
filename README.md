## High-Paying Jobs in the US — Complete Case Study

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MarwaBS/High_pay_Analysis_us/actions/workflows/ci.yml/badge.svg)](https://github.com/MarwaBS/High_pay_Analysis_us/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](streamlit_app.py)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](04_salary_prediction_model.ipynb)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)](04_salary_prediction_model.ipynb)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

An end-to-end data science pipeline analyzing high-paying jobs ($100K+) across the United States,
integrating **Bureau of Labor Statistics (BLS)** and **US Census** microdata.
Includes EDA, geospatial mapping, **XGBoost salary prediction**, **SHAP explainability**,
and an **interactive Streamlit dashboard**.

The pipeline is organized across four notebooks:

| Notebook | Purpose |
|----------|---------|
| `high_pay_jobs_data_cleaning.ipynb` | Data integration & cleaning (BLS + Census → single dataset) |
| `high_paying_jobs_data_visualization.ipynb` | EDA: distributions, rankings, correlations |
| `us_high_income_jobs_mapping.ipynb` | Geospatial: choropleth maps by state |
| `04_salary_prediction_model.ipynb` | **ML: XGBoost + SHAP + statistical tests** |

All figures are saved automatically to `Images/` at 300 DPI. Filenames are unique across notebooks.

### Quickstart

```bash
# Create and activate virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install all dependencies (ML + dashboard + testing)
pip install -r requirements.txt

# Optional: lock exact versions for reproducibility
pip freeze > requirements-lock.txt
```

### Run the interactive dashboard

```bash
streamlit run streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

## Portfolio highlights

- **ML salary prediction:** XGBoost model (R² reported in notebook) with cross-validation and SHAP explainability.
- **Statistical rigor:** ANOVA and Welch t-tests validate EDA findings (education, gender, regional income gaps).
- **Interactive dashboard:** Streamlit app with Plotly charts, geographic choropleth, salary predictor widget, and model insights.
- **End-to-end pipeline:** raw BLS/Census → cleaned dataset → EDA → ML model → dashboard.
- **Reproducible:** single `config.yaml` for all thresholds and paths; `pytest` test suite; GitHub Actions CI.
- **Geospatial fallback:** maps render with GeoPandas; graceful pyshp fallback ensures PNG export.

## Data

- Source file: Data/cleaned_high_pay_data.csv
- Key fields used: Occupation, Annual Income, Education Level, Gender, State Abbreviation/State, (optional) Annual Mean Wage, Location Quotient, Employment, Jobs per 1000.

## Data cleaning and preparation

This repository includes a full cleaning and integration pipeline in `high_pay_jobs_data_cleaning.ipynb`. Here’s a concise summary of what it does and the resulting dataset used by the analysis notebooks:

- Inputs
	- BLS (wages/employment by occupation and state): raw in `Resources/bls_state_data.xlsx`, cleaned export saved as `Data/bls_data.csv`.
	- Census (demographics, education, occupation microdata): raw in `Resources/census_data.csv`, cleaned export saved as `Data/census_data.csv`.

- BLS cleaning (occupation and wage stats)
	- Standardize strings: strip whitespace and title‑case `AREA_TITLE` (state name) and `OCC_TITLE` (occupation).
	- Normalize occupation codes: remove hyphens from `OCC_CODE` and drop invalid codes.
	- Convert numeric columns to proper dtypes; drop rows with missing values for a clean baseline.
	- Keep the 50 U.S. states only (exclude territories/aggregates) using `PRIM_STATE` and verify 50 unique states.
	- Define “high‑paying” cohort: `A_MEAN >= 100000` or `H_MEAN >= 48.08` (≈$100K annualized).
	- Rename `AREA_TITLE -> STATE`; save subset as `Data/bls_data.csv`.

- Census cleaning (demographics and education)
	- Read `Resources/census_data.csv`; extract the 6‑digit SOC from `OCCSOC` and zero‑pad.
	- Decode codes to labels: `SEX -> {Male,Female}`, `STATEICP ->` state abbreviation, education `EDUCD -> EDUCATION_LABEL`, degree field `DEGFIELDD -> DEGFIELDD_NAME`.
	- Drop rows with missing key fields; rename `OCCSOC -> OCC_CODE`; save as `Data/census_data.csv`.

- Integration
	- Inner join on `['OCC_CODE','STATE']` to align Census demographics with BLS wages by occupation and state.
	- Reorder and rename columns to a tidy schema; remove duplicate rows.
	- Final dataset saved to `Data/cleaned_high_pay_data.csv` with shape `(10255, 15)`.

- Final schema (columns)
	- State Abbreviation, State, Gender, Age, Education Code, Education Level, Degree Field, Occupation Code, Occupation, Annual Income, Employment, Location Quotient, Jobs per 1000, Hourly Mean, Annual Mean Wage.

For full implementation details and reproducibility steps, see `high_pay_jobs_data_cleaning.ipynb`.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA notebooks (open in Jupyter)
jupyter notebook

# 3. Run the ML notebook (trains model, saves to models/)
#    Open 04_salary_prediction_model.ipynb → Run All

# 4. Launch the interactive dashboard
streamlit run streamlit_app.py

# 5. Run the test suite
pytest tests/ -v

# Optional: full GIS stack for maps
pip install geopandas shapely fiona pyproj rtree
```

### Run with Docker (one command, no Python setup needed)

```bash
# Build image and start the dashboard
docker compose up --build

# Dashboard available at http://localhost:8501
# The trained model is saved to ./models/ and reused on subsequent starts

# Stop
docker compose down
```

### Run the cleaning pipeline (optional)

If you want to regenerate the cleaned dataset:

1. Ensure the raw files are present:
   - `Resources/bls_state_data.xlsx`
   - `Resources/census_data.csv`

2. Open `high_pay_jobs_data_cleaning.ipynb` and Run All. It will create:
   - `Data/bls_data.csv`
   - `Data/census_data.csv`
   - `Data/cleaned_high_pay_data.csv` (final; shape ~ 10K rows × 15 cols)

All notebooks consume `Data/cleaned_high_pay_data.csv` as the single source of truth.

## Findings with figures (non‑map visuals)

Top occupations by average income  
![Top 10 Occupations by Average Income](./Images/Top_Occupations_Avg_Income.png)
- Specialized roles dominate the top earnings within the $100K+ cohort.

Average income by education level  
![Average Income by Education Level](./Images/Average_Income_by_Education_Level.png)
- Higher education levels correlate with higher average incomes; returns appear incremental.

Salary distributions for top occupations  
![Salary Distribution for Top Occupations](./Images/Top_10_Salary_Distribution.png)
- Some roles show tight pay clusters; others exhibit wider dispersion.

Correlation among numeric features  
![Correlation Heatmap](./Images/Correlation_Annual_Income.png)
- Employment and jobs-per-1000 co-move; wages co-move as expected. Annual income has weak links to these, implying other drivers (role, geography).

Age vs annual income  
![Age vs Income](./Images/Age_Annual_Income.png)
- Income rises early and plateaus later; scatter suggests additional factors influence earnings.

Gender distribution across top occupations  
![Gender by Occupation](./Images/Gender_Distribution_Occupations.png)
- Mix varies by occupation; some balanced, others skewed.

Gender distribution across top states  
![Gender by State](./Images/Gender_Distribution_States.png)
- Large markets dominate headcount; balance differs by state reflecting industry mix.

Gender distribution by education (within $100K+)  
![Gender by Education](./Images/Gender_Education_Distribution.png)
- Gender mix differs by education pathways feeding into high-paying jobs.

Distribution of $100K+ individuals by state (bar)  
![High-Paid Individuals by State](./Images/High_Paid_Individuals_by_State.png)
- Headcount clusters in large economies and tech/finance hubs.

Average income by state (bar)  
![Average Income by State (Bar)](./Images/Average_Highest_Income_state_Viz.png)
- Higher averages appear in states with strong tech/finance presence and higher COL.

Location Quotient by state (bar)  
![LQ by State (Bar)](./Images/High_Paying_Jobs_LQ_Distribution_Viz.png)
- Above-average concentration indicates specialized clusters.

Dominant education level by state (count of states)  
![Dominant Education by State (Bar)](./Images/Dominant_education_by_state_Viz.png)
- Most states lean toward higher education levels; some show strong professional/specialized degree presence.

Education–income premium by state (bar)  
![Education Premium by State (Bar)](./Images/Education_Income_Premiums_by_State_Viz.png)
- Premiums vary and tend to be larger where specialized credentials are rewarded.

Market size vs education premium (scatter)  
![Market Size vs Premium](./Images/Market_Size_Income_Premium_Analysis_Viz.png)
- Larger markets often show higher premiums, but composition matters more than size.

Average income by US Census region (bar)  
![Regional Patterns (Bar)](./Images/Regional_Patterns_Analysis_Viz.png)
- Regions with dense tech/finance ecosystems show higher averages; market share reflects population/economic concentration.

## Map gallery (choropleths)

Average income by state (map)  
![Average Income by State (Map)](./Images/Average_Highest_Income_state.png)

High‑paying jobs distribution (map)  
![High‑Paying Jobs Distribution (Map)](./Images/High_Paying_Jobs_Distribution.png)

Location Quotient by state (map)  
![LQ by State (Map)](./Images/High_Paying_Jobs_LQ_Distribution.png)

Dominant education by state (map)  
![Dominant Education by State (Map)](./Images/Dominant_education_by_state.png)

Gender share overlays (map)  
![Male % (Map)](./Images/Male_Percentage_state.png)  
![Female % (Map)](./Images/Female_Percentage_state.png)

Education premium and market dynamics (panels)  
![Market Size vs Premium (Panels)](./Images/Market_Size_Income_Premium_Analysis.png)

Regional patterns (panels)  
![Regional Patterns (Panels)](./Images/Regional_Patterns_Analysis.png)

Note: Maps render when GeoPandas+GIS stack is installed. The non-map notebook remains fully runnable without it.

## Case interpretation and results

Synthesis across all visuals and maps:

- Geographic opportunity vs. concentration
	- Large economies (CA, NY, TX) lead in absolute high‑income headcount.
	- Concentration (LQ) is strongest in MD, VA, WA—specialized clusters drive premium roles.

- Education ROI signals
	- Bachelor’s degrees dominate most states for $100K+ roles.
	- Master’s is dominant in a handful (SD, MT, NE, MO, WV); Professional dominates ND; Doctoral is not dominant in any state.
	- Premiums rise with credential intensity, but vary regionally with industry mix.

- Demographic dynamics
	- Gender participation is uneven across occupations and states; large markets show mixed balance.
	- Age–income patterns plateau later in career, suggesting role/region drive additional upside.

- Market size vs. premium
	- Bigger markets often pair with higher education premiums, but composition (tech/finance/healthcare) matters more than just size.

- Correlation overview
	- Employment and jobs‑per‑1000 move together; wage measures correlate as expected.
	- Annual income shows weak links to headcount metrics—reinforcing the role of occupation and geography.

These patterns align with the Business Intelligence report’s narrative: opportunity is not uniform; specialty clusters and education alignment amplify outcomes.

### Recommendations

- Job seekers
	- Target states with strong concentration (LQ) for your field, not just volume.
	- Align degree investments with target regions and sectors; consider timing to avoid oversaturated markets.

- Employers
	- Calibrate compensation to regional wage dynamics; recruit across clusters where talent density is highest.
	- Use targeted outreach to improve diversity where gaps exist.

- Policymakers
	- Direct workforce and education funding toward regional specializations.
	- Foster mobility pipelines and upskilling for underrepresented demographics.

### Limitations and next steps

- The analysis focuses on a single timeframe and nominal incomes (no cost‑of‑living adjustment).
- Industry deep‑dives and longitudinal trends can refine signals.
- Future work: add COLA normalization, sector‑specific cuts, and predictive modeling.

See `reports/data_science_report.md` for a concise, analyst‑oriented narrative, methods, and next steps.

## Data sources & attribution

- U.S. Bureau of Labor Statistics (BLS): state-level Occupational Employment and Wage Statistics (OEWS).
- U.S. Census Bureau: microdata used for demographics and education fields.

Data are used for educational/analytical purposes. Please consult each provider’s terms for reuse and citation.

## Repository structure

```
High_pay_Analysis_us/
├── Notebooks
│   ├── high_pay_jobs_data_cleaning.ipynb          # Pipeline: BLS + Census → cleaned CSV
│   ├── high_paying_jobs_data_visualization.ipynb  # EDA: distributions, rankings, correlations
│   ├── us_high_income_jobs_mapping.ipynb          # Geospatial: choropleth maps
│   └── 04_salary_prediction_model.ipynb           # ★ ML: XGBoost + SHAP + statistical tests
│
├── streamlit_app.py                               # ★ Interactive dashboard (run with streamlit)
├── config.yaml                                    # ★ All thresholds, paths, color palettes
├── Dockerfile                                     # ★ Multi-stage Docker build (python:3.11-slim)
├── docker-compose.yml                             # ★ One-command deployment
│
├── Data/                                          # Processed datasets (single source of truth)
│   ├── cleaned_high_pay_data.csv                  #   10,255 rows × 15 cols
│   ├── bls_data.csv
│   └── census_data.csv
│
├── Resources/                                     # Raw source data
│   ├── bls_state_data.xlsx
│   └── census_data.csv
│
├── models/                                        # Saved ML model artefacts
│   ├── xgb_salary_model.pkl                       #   (generated by notebook 4)
│   └── feature_names.pkl
│
├── Images/                                        # Auto-saved figures (300 DPI, unique names)
│   ├── [26 EDA/map PNGs from notebooks 2 & 3]
│   ├── SHAP_Feature_Importance_Bar.png            #   (generated by notebook 4)
│   ├── SHAP_Beeswarm.png
│   ├── SHAP_Dependence_Plots.png
│   └── Residual_Analysis.png
│
├── tests/
│   └── test_pipeline.py                           # ★ pytest unit tests (config, schema, ML)
│
├── .github/workflows/
│   └── ci.yml                                     # ★ GitHub Actions: lint + test on every push
│
├── reports/
│   └── data_science_report.md
│
├── us_state/                                      # Shapefiles for mapping (optional)
├── requirements.txt
├── config.yaml
└── LICENSE
```

## Reproducibility and notes

- **Single source of truth:** all four notebooks consume `Data/cleaned_high_pay_data.csv`.
- **Config-driven:** thresholds, paths, and color palettes live in `config.yaml` — never hardcoded.
- **Tested:** `pytest tests/` covers data schema, feature engineering, and model performance.
- **CI/CD:** GitHub Actions runs lint + tests on every push.
- **Lock file:** run `pip freeze > requirements-lock.txt` for exact version pinning.
- For narrative details, see `reports/data_science_report.md`.
