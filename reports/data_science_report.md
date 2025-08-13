# High-Paying Jobs in the US — Data Analysis Report

## 1. Problem statement and goals

Analyze the landscape of high-paying jobs (≥ $100K) across US states to answer:
- Where are high-paying roles most prevalent and most concentrated?
- How does education level relate to high-income participation and premiums?
- What demographic patterns (gender, age) are present within the $100K+ cohort?
- How do market size and specialization interact with income outcomes?

Deliverables: clean dataset, reproducible notebooks, figures, and a concise set of insights with technical notes and next steps.

## 2. Data sources and scope

- Unified dataset: Data/cleaned_high_pay_data.csv (shape ~10k+ × 15)
- Sources integrated:
  - Bureau of Labor Statistics (BLS): wages, employment, location quotient (LQ) by occupation/state
  - US Census: demographics, education, occupation microdata
- Unit of analysis: individual records aligned with occupation-state wage context

Final schema (selected):
State Abbreviation, State, Gender, Age, Education Code, Education Level, Degree Field, Occupation Code, Occupation, Annual Income, Employment, Location Quotient, Jobs per 1000, Hourly Mean, Annual Mean Wage.

## 3. Data preparation (summary)

Implemented in high_pay_jobs_data_cleaning.ipynb:
- BLS: string normalization; OCC_CODE de-hyphenation; numeric casting; exclude territories; filter high-pay cohort (A_MEAN ≥ 100000 or H_MEAN ≥ 48.08); rename AREA_TITLE → STATE.
- Census: extract zero-padded SOC from OCCSOC; decode SEX/STATEICP/EDUCD/DEGFIELDD; drop missing key fields; rename OCCSOC → OCC_CODE.
- Integration: inner join on [OCC_CODE, STATE]; reorder/rename columns; drop duplicates; save to Data/cleaned_high_pay_data.csv.

Reproducibility: see the cleaning notebook for exact code and outputs.

## 4. Methods

- Exploratory data analysis (EDA): descriptive statistics, grouped aggregations.
- Visual analytics:
  - Non-map figures (bars, distributions, correlations) in high_paying_jobs_data_visualization.ipynb
  - Choropleths and state-level maps in us_high_income_jobs_mapping.ipynb (optional GeoPandas)
- Geospatial: choropleth rendering with GeoPandas; fallback static export if GIS stack is missing.

## 5. Key findings (evidence-backed)

- Geography: Large economies (CA, NY, TX) lead in absolute high-income headcount; concentration (LQ) peaks in MD, VA, WA indicating specialized clusters.
- Education: Bachelor’s degree dominates most states in the $100K+ cohort; Master’s dominates a handful (SD, MT, NE, MO, WV); Professional dominates ND; Doctoral is not dominant in any state.
- Demographics: Gender participation varies across occupations and states; age–income relationship rises early and plateaus later.
- Market dynamics: Bigger markets often show higher education premiums, but composition (tech/finance/healthcare) matters more than size alone.
- Correlations: Employment and jobs-per-1000 co-move; wage measures are internally consistent; annual income has weak correlation with headcount metrics—role and geography are stronger drivers.

Figures referenced (see Images/):
- Top_Occupations_Avg_Income.png, Average_Income_by_Education_Level.png
- High_Paid_Individuals_by_State.png, Top_10_Salary_Distribution.png
- Correlation_Annual_Income.png, Age_Annual_Income.png
- Average_Highest_Income_state[_Viz].png, High_Paying_Jobs_LQ_Distribution[_Viz].png
- Dominant_education_by_state[_Viz].png, Education_Income_Premiums_by_State_Viz.png
- Market_Size_Income_Premium_Analysis[_Viz].png, Regional_Patterns_Analysis[_Viz].png

## 6. Technical notes

- Environment: requirements.txt (pandas, numpy, seaborn, matplotlib; optional geopandas/pyshp for maps)
- Output management: figures saved at 300 DPI to Images/ with unique filenames across notebooks
- Single source of truth: Data/cleaned_high_pay_data.csv consumed by both notebooks

## 7. Limitations

- Snapshot analysis (single timeframe); nominal incomes (no cost-of-living adjustment)
- Potential confounding by occupation mix and regional industry structure
- No causal inference; descriptive and exploratory focus

## 8. Next steps (data-science oriented)

- Normalization: adjust incomes for regional price parity or COLA
- Statistical testing: ANOVA/OLS for education premiums by region; robust SE to address heteroskedasticity
- Modeling: predictive modeling of income by occupation, education, and geography; SHAP for interpretability
- Temporal: multi-year trend analysis and cluster dynamics over time
- Industry cuts: sector-specific deep-dives (tech, healthcare, finance, engineering)

## 9. Reproducibility checklist

- Clean the data: run high_pay_jobs_data_cleaning.ipynb (generates Data/cleaned_high_pay_data.csv)
- Generate figures: run high_paying_jobs_data_visualization.ipynb and us_high_income_jobs_mapping.ipynb
- Verify outputs: ensure Images/ contains all referenced figures

This report is designed for a data scientist/data analyst portfolio: concise problem framing, transparent pipeline, reproducible analysis, and actionable next steps.
