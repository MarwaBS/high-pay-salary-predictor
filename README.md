# High-Paying Jobs Analysis: BLS and Census Data  

## Introduction  
This project investigates high-paying jobs (annual salaries of $100K+) in the U.S. by integrating data from the Bureau of Labor Statistics (BLS) and the U.S. Census Bureau. The goal is to uncover trends, geographic patterns, and demographic insights into these occupations.  

---

## Data Sources  

### 1. Bureau of Labor Statistics (BLS)  
- **Dataset**: Occupational Employment and Wage Statistics (OEWS)  
- **Content**:  
  - Employment and wage estimates for various occupations.  
  - Geographic and industry-specific data.  
- **Source**: [BLS OEWS Tables](https://www.bls.gov/oes/tables.htm)  

### 2. U.S. Census Bureau  
- **Dataset**: Educational Attainment and Demographics  
- **Content**:  
  - Individual-level demographic, education, and occupation data.  
- **Source**: [Census Bureau](https://www.census.gov/)  

---

## Data Cleaning  

### Bureau of Labor Statistics (BLS)  
- Filtered columns relevant to the analysis (e.g., `OCC_CODE`, `AREA_TITLE`, `A_MEAN`).  
- Standardized `OCC_CODE` for consistency (removed hyphens and invalid entries).  
- Retained only occupations with annual mean salaries ≥ $100K.  
- Excluded data from non-mainland U.S. regions.  

### Census Data  
- Reformatted `OCCSOC` to match BLS's `OCC_CODE` structure.  
- Decoded categorical columns like `SEX` and `EDUCD` into descriptive labels.  
- Standardized state and region codes for compatibility with BLS data.  
- Removed rows with missing or incomplete entries.  

---

## Data Merging  

### Process  
- **Objective**: Combine datasets to include all relevant columns for matched rows where both `PRIM_STATE` (state abbreviation) and `OCC_CODE` (occupation code) align.  
- **Steps**:  
  1. Merged using an inner join on `PRIM_STATE` and `OCC_CODE`.  
  2. Checked for missing values and verified data integrity.  
  3. Renamed columns for clarity using a mapping dictionary.  
  4. Dropped redundant columns.  

### Output  
- **File**: `cleaned_high_pay_data.csv`  
- **Content**: A unified dataset containing:  
  - **Geographic Details**: State and area names.  
  - **Occupation Details**: Codes, titles, and employment numbers.  
  - **Wage Data**: Hourly and annual wages (mean and median).  
  - **Demographics**: Gender, age, education levels, and degree fields.  

---

## Key Features of Cleaned Data  

1. **Geographic Analysis**:  
   - Data categorized by state and region.  

2. **Wage Analysis**:  
   - Both hourly and annual salaries for each occupation.  

3. **Demographics**:  
   - Variables like age, gender, and education level included.  

4. **Consistency**:  
   - Standardized codes for seamless integration and analysis.  

---

## Objectives  

1. **National Trends**:  
   - Identify which occupations dominate the $100K+ category.  

2. **Geographic Insights**:  
   - Discover regions with the highest concentrations of high-paying jobs.  

3. **Demographic Analysis**:  
   - Examine the role of gender, age, and education in earning potential.  

4. **Educational Impact**:  
   - Assess how educational attainment correlates with high-paying occupations.  

---

## Next Steps  

1. **Analysis**:  
   - Explore trends and correlations using the merged dataset.  

2. **Visualization**:  
   - Create graphs to illustrate key findings.  

3. **Insights**:  
   - Summarize observations to inform policies or career planning.  

---

## Conclusion  
This project successfully integrates and prepares two comprehensive datasets for a detailed exploration of high-paying jobs in the U.S. The cleaned and merged data provide a strong foundation for analyzing trends, demographics, and geographic distributions of $100K+ occupations.  
