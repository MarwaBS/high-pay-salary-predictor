## High-Paying Jobs in the US — Complete Case Study

This study analyzes high-paying jobs ($100K+) across the United States, combining non-map visuals and choropleth maps. The analysis is split across two notebooks to avoid duplication and keep a clean workflow:

- high_paying_jobs_data_visualization.ipynb — Non-map visuals (distributions, rankings, correlations)
- us_high_income_jobs_mapping.ipynb — Choropleth maps only (optional)

All figures are saved automatically to Images/ at 300 DPI. Filenames are unique across notebooks.

### Quickstart (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: full GIS stack for maps
pip install geopandas shapely fiona pyproj rtree
```

## Portfolio highlights

- End-to-end pipeline: raw BLS/Census ➜ cleaned, integrated dataset ➜ reproducible visuals and maps.
- Robust notebooks: single import/load, helper-based saving, idempotent and unique figure filenames.
- Geospatial fallback: maps render with GeoPandas; graceful pyshp fallback ensures PNG export.
- Clear storytelling: README + data-science report with insights, limitations, and next steps.

## Data

- Source file: Data/cleaned_high_pay_data.csv
- Key fields used: Occupation, Annual Income, Education Level, Gender, State Abbreviation/State, (optional) Annual Mean Wage, Location Quotient, Employment, Jobs per 1000.

## Data cleaning and preparation

This repository includes a full cleaning and integration pipeline in `high_pay_jobs_data_cleaning.ipynb`. Here’s a concise summary of what it does and the resulting dataset used by the analysis notebooks:

- Inputs
	- BLS (wages/employment by occupation and state): raw in `Ressources/bls_state_data.xlsx`, cleaned export saved as `Data/bls_data.csv`.
	- Census (demographics, education, occupation microdata): raw in `Ressources/census_data.csv`, cleaned export saved as `Data/census_data.csv`.

- BLS cleaning (occupation and wage stats)
	- Standardize strings: strip whitespace and title‑case `AREA_TITLE` (state name) and `OCC_TITLE` (occupation).
	- Normalize occupation codes: remove hyphens from `OCC_CODE` and drop invalid codes.
	- Convert numeric columns to proper dtypes; drop rows with missing values for a clean baseline.
	- Keep the 50 U.S. states only (exclude territories/aggregates) using `PRIM_STATE` and verify 50 unique states.
	- Define “high‑paying” cohort: `A_MEAN >= 100000` or `H_MEAN >= 48.08` (≈$100K annualized).
	- Rename `AREA_TITLE -> STATE`; save subset as `Data/bls_data.csv`.

- Census cleaning (demographics and education)
	- Read `Ressources/census_data.csv`; extract the 6‑digit SOC from `OCCSOC` and zero‑pad.
	- Decode codes to labels: `SEX -> {Male,Female}`, `STATEICP ->` state abbreviation, education `EDUCD -> EDUCATION_LABEL`, degree field `DEGFIELDD -> DEGFIELDD_NAME`.
	- Drop rows with missing key fields; rename `OCCSOC -> OCC_CODE`; save as `Data/census_data.csv`.

- Integration
	- Inner join on `['OCC_CODE','STATE']` to align Census demographics with BLS wages by occupation and state.
	- Reorder and rename columns to a tidy schema; remove duplicate rows.
	- Final dataset saved to `Data/cleaned_high_pay_data.csv` with shape `(10255, 15)`.

- Final schema (columns)
	- State Abbreviation, State, Gender, Age, Education Code, Education Level, Degree Field, Occupation Code, Occupation, Annual Income, Employment, Location Quotient, Jobs per 1000, Hourly Mean, Annual Mean Wage.

For full implementation details and reproducibility steps, see `high_pay_jobs_data_cleaning.ipynb`.

## How to run (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: install mapping stack if you want to render maps
pip install geopandas shapely fiona pyproj rtree
```

- Then open and Run All:
- high_paying_jobs_data_visualization.ipynb (non-map visuals)
- us_high_income_jobs_mapping.ipynb (maps; will gracefully skip if GeoPandas is unavailable)

### Run the cleaning pipeline (optional)

If you want to regenerate the cleaned dataset used by both notebooks:

1) Ensure the raw files are present:
	- `Ressources/bls_state_data.xlsx`
	- `Ressources/census_data.csv`

2) Open `high_pay_jobs_data_cleaning.ipynb` and Run All. It will create:
	- `Data/bls_data.csv`
	- `Data/census_data.csv`
	- `Data/cleaned_high_pay_data.csv` (final; shape ~ (10k+, 15 cols))

Note: Both notebooks consume `Data/cleaned_high_pay_data.csv` as the single source of truth.

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
├── high_paying_jobs_data_visualization.ipynb  # Non-map analysis and figures
├── high_pay_jobs_data_cleaning.ipynb          # Data cleaning and integration pipeline
├── us_high_income_jobs_mapping.ipynb  # Choropleth/maps only (optional)
├── Data/
│   ├── cleaned_high_pay_data.csv
│   ├── bls_data.csv
│   └── census_data.csv
├── Images/                            # Auto-saved figures (300 DPI, unique names)
│   ├── Top_Occupations_Avg_Income.png
│   ├── Average_Income_by_Education_Level.png
│   ├── Gender_Distribution_Occupations.png
│   ├── Gender_Distribution_States.png
│   ├── Gender_Education_Distribution.png
│   ├── High_Paid_Individuals_by_State.png
│   ├── Top_10_Salary_Distribution.png
│   ├── Correlation_Annual_Income.png
│   ├── Average_Highest_Income_state_Viz.png
│   ├── High_Paying_Jobs_LQ_Distribution_Viz.png
│   ├── Dominant_education_by_state_Viz.png
│   ├── Education_Income_Premiums_by_State_Viz.png
│   ├── Market_Size_Income_Premium_Analysis_Viz.png
│   ├── Regional_Patterns_Analysis_Viz.png
│   ├── Average_Highest_Income_state.png
│   ├── High_Paying_Jobs_Distribution.png
│   ├── High_Paying_Jobs_LQ_Distribution.png
│   ├── Dominant_education_by_state.png
│   ├── Male_Percentage_state.png
│   ├── Female_Percentage_state.png
│   ├── Market_Size_Income_Premium_Analysis.png
│   └── Regional_Patterns_Analysis.png
├── reports/
│   └── data_science_report.md
└── us_state/                          # Shapefiles for mapping (optional)
```

## Reproducibility and notes

- Figures are generated directly from the notebooks and saved to Images/. Names are unique across notebooks.
- Non-map visuals are complementary to maps, not duplicates.
- Requirements are listed in requirements.txt (pandas, numpy, matplotlib, seaborn, jupyter, optional: geopandas, pyshp, etc.).
- For more narrative detail, see reports/data_science_report.md.
