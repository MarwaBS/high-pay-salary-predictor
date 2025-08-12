# =============================================================================
# 1. SETUP AND CONFIGURATION
# =============================================================================
# Core imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Try to import GeoPandas, fallback if unavailable
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
    print("✅ GeoPandas available - Full geographic analysis enabled")
except ImportError:
    HAS_GEOPANDAS = False
    print("⚠️ GeoPandas not available - Using alternative visualization approach")

# Set professional style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'

# Ensure output directory exists
os.makedirs("Images", exist_ok=True)

print("\n📊 Environment ready for geographic analysis")

# =============================================================================
# 2. DATA LOADING
# =============================================================================
# File paths
high_pay_data_path = './Data/cleaned_high_pay_data.csv'
shapefile_path = './us_state/us_state.shp'

# Load the high-pay data
print("📊 Loading high-paying jobs data...")
try:
    high_pay_data = pd.read_csv(high_pay_data_path)
    print(f"✅ Data loaded: {len(high_pay_data):,} records")
except FileNotFoundError:
    print(f"❌ Error: File not found at {high_pay_data_path}")
    high_pay_data = None

# Load geographic data (with fallback for missing GeoPandas)
if HAS_GEOPANDAS:
    try:
        us_states = gpd.read_file(shapefile_path)
        print("🗺️  Geographic shapefile loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Shapefile not found at {shapefile_path}")
        us_states = None
else:
    print("🗺️  Using alternative geographic approach...")
    us_states = None

# Ensure 'STUSPS' column exists in geographic data
if HAS_GEOPANDAS and us_states is not None:
    if 'STUSPS' not in us_states.columns:
        print("⚠️ 'STUSPS' column is missing in the geographic data. Attempting to fix...")
        if 'STATE_ABBR' in us_states.columns:
            us_states.rename(columns={'STATE_ABBR': 'STUSPS'}, inplace=True)
            print("✅ Renamed 'STATE_ABBR' to 'STUSPS'.")
        else:
            print("❌ 'STUSPS' column not found. Adding manually...")
            state_abbreviation_mapping = {
                'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
                'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
                'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
                'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
                'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
                'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
                'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
                'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
                'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
                'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
                'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
                'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
                'Wisconsin': 'WI', 'Wyoming': 'WY'
            }
            us_states['STUSPS'] = us_states['STATE_NAME'].map(state_abbreviation_mapping)
            print("✅ Added 'STUSPS' column manually.")

# =============================================================================
# 3. DATA PREPARATION
# =============================================================================
# Aggregate high-paying jobs and total jobs by state
print("📊 Aggregating high-paying jobs and total jobs by state...")
try:
    job_data = (
        high_pay_data.groupby('State Abbreviation')
        .agg(
            High_Paying_Jobs=('Annual Income', 'size'),
            Total_Jobs=('Employment', 'size'),
            Location_Quotient=('Location Quotient', 'mean')  # Average LQ for better representation
        )
        .reset_index()
    )
    print(f"✅ Aggregation completed: {len(job_data)} states analyzed")
except KeyError as e:
    print(f"❌ Error during aggregation: {e}")
    job_data = pd.DataFrame()

# Merge with GeoDataFrame for mapping
if HAS_GEOPANDAS and us_states is not None:
    try:
        job_data_geo = pd.merge(us_states, job_data, left_on='STUSPS', right_on='State Abbreviation', how='left')
        job_data_geo = gpd.GeoDataFrame(job_data_geo, geometry=job_data_geo.geometry)
        print("🗺️  Geographic data merged successfully")
    except KeyError as e:
        print(f"❌ Error during GeoDataFrame merge: {e}")
        job_data_geo = None
else:
    print("⚠️ GeoPandas not available or geographic data missing. Using alternative approach...")
    job_data_geo = job_data.copy()

# Exclude Alaska (AK) and Hawaii (HI) for mapping clarity
if job_data_geo is not None:
    try:
        job_data_geo = job_data_geo.loc[~job_data_geo['STUSPS'].isin(['AK', 'HI'])]
        print(f"✅ Excluded Alaska and Hawaii for mapping clarity: {len(job_data_geo)} states remaining")
    except KeyError as e:
        print(f"❌ Error during exclusion: {e}")

# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================
# Define reusable functions for plotting maps
def StatesPlot(df, column_to_plot, cmap='viridis', label_color='black', label_size=6,
               title='United States Map', filename='us_map.png', min_value=None, max_value=None,
               edge_color='black', edge_linewidth=0.5):
    """
    Enhanced function to plot US data - works with or without geopandas.
    Creates professional bar charts and heatmaps for state-level analysis.
    """
    # ...existing code for StatesPlot...

def Education_State(df, column, title, filename, cmap='tab20'):
    """
    Enhanced function to visualize dominant education levels by state.
    """
    # ...existing code for Education_State...

def plot_gender_distribution(df):
    """
    Enhanced function to plot gender distribution for high-paying jobs.
    """
    # ...existing code for plot_gender_distribution...

# =============================================================================
# 5. ANALYSIS AND VISUALIZATION
# =============================================================================
# Question 1: Income Distribution by State
# Question 2: High-Paying Jobs Distribution
# Question 3: Dominant Education Levels
# Question 4: Gender Distribution

# Ensure all maps are preserved and enhanced
StatesPlot(
    df=job_data_geo,
    column_to_plot='High_Paying_Jobs',
    cmap='YlGnBu',
    label_color='black',
    label_size=8,
    title='High-Paying Jobs Distribution Across the United States',
    filename='High-Paying_Jobs_Distribution.png',
    edge_color='black'
)

StatesPlot(
    df=job_data_geo,
    column_to_plot='Location_Quotient',
    cmap='YlGnBu',
    label_color='black',
    label_size=8,
    title='Location Quotient Distribution Across the United States',
    filename='High_Paying_Jobs_LQ_Distribution.png',
    edge_color='black'
)

Education_State(
    df=job_data_geo,
    column='Education Level',
    title='Dominant Education Level in High-Paying States ($100K+)',
    filename='Dominant_education_by_state.png'
)

plot_gender_distribution(job_data_geo)