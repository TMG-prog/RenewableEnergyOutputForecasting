"""
Renewable Energy Grid Management Dashboard
Run this app using: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import types

# ==========================================
# 0. PATH RESOLUTION
# ==========================================
# Anchor model paths to this script's location
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="GridOps: Renewable Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalist CSS for a cleaner UI
st.markdown("""
    <style>
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        max-width: 1200px;
    }
    h1 { 
        color: #1E293B; 
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #E2E8F0; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stMetric label {
        color: #64748B !important;
        font-weight: 500;
        font-size: 1rem;
    }
    .stMetric div { 
        color: #0F172A !important; 
    }
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
def cube_root(values):
    return np.cbrt(values)

def cube_values(values):
    return np.power(values, 3)

_fake_main = types.ModuleType("main")
_fake_main.cube_root = cube_root
_fake_main.cube_values = cube_values
sys.modules["main"] = _fake_main

@st.cache_resource
def load_models():
    """Load the trained pipelines."""
    try:
        wind_path = os.path.join(MODELS_DIR, "random_forest", "random_forest_wind.pkl")
        solar_path = os.path.join(MODELS_DIR, "random_forest", "random_forest_solar.pkl")

        wind_model = joblib.load(wind_path)
        solar_model = joblib.load(solar_path)
        return wind_model, solar_model
    except Exception as e:
        st.warning(f"Models not found or failed to load. Running in simulation mode. (Error: {e})")
        return None, None

wind_model, solar_model = load_models()

# ==========================================
# 3. SIDEBAR: SCENARIOS & INPUTS
# ==========================================
with st.sidebar:
    st.title(" Grid Controls")
    st.markdown("---")

    st.subheader("Scenario Planning")
    scenario = st.radio(
        "Select Weather Forecast",
        ["Normal Operations", "Severe Storm", "Heatwave"],
        captions=["Standard day", "High Wind, Low Solar", "Low Wind, High Solar"]
    )

    # Dynamic Defaults
    if scenario == "Normal Operations":
        def_wind, def_cloud, def_uv = 25.0, 40.0, 6.0
    elif scenario == "Severe Storm":
        def_wind, def_cloud, def_uv = 65.0, 95.0, 1.0
    else:
        def_wind, def_cloud, def_uv = 10.0, 10.0, 10.0

    st.markdown("---")
    st.subheader("Fine-Tune Parameters")
    base_wind = st.slider("Base Wind Speed (kph)", 0.0, 100.0, def_wind, 1.0)
    base_cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, def_cloud, 1.0)
    base_uv = st.slider("Peak UV Index", 0.0, 11.0, def_uv, 0.5)

    st.markdown("---")
    target_demand = st.number_input("Grid Demand Threshold (MW)", value=150000, step=5000)

# ==========================================
# 4. DATA SIMULATION & FEATURE EXPANSION
# ==========================================
def generate_40_feature_forecast(wind_base, cloud_base, uv_base):
    hours = np.arange(24)
    uv_curve = np.where((hours > 6) & (hours < 18), uv_base * np.sin(np.pi * (hours - 6) / 12), 0)
    
    np.random.seed(42) 
    wind_curve = np.clip(wind_base + np.random.normal(0, wind_base * 0.1, 24), 0, None)
    
    temp_base = 22.0
    temp_curve = temp_base + 5 * np.sin(np.pi * (hours - 8) / 12)
    
    df = pd.DataFrame()
    df['latitude'] = -1.58
    df['longitude'] = 36.96
    df['country_daily_mean_temp'] = temp_base
    df['country_daily_mean_wind'] = wind_base
    df['temperature_celsius'] = temp_curve
    df['wind_kph'] = wind_curve
    df['cloud_cover'] = cloud_base
    df['uv_index'] = uv_curve
    df['wind_degree'] = 180.0
    df['pressure_mb'] = 1012.0
    df['precip_mm'] = 0.0
    df['humidity'] = 50.0
    df['feels_like_celsius'] = temp_curve
    df['visibility_km'] = 10.0
    df['gust_kph'] = wind_curve * 1.3
    df['moon_illumination'] = 50.0
    
    aq_cols = ['air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide', 
               'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10', 
               'air_quality_us-epa-index', 'air_quality_gb-defra-index']
    for col in aq_cols:
        df[col] = 1.0 
        
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * 180 / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * 180 / 365.25)
    
    df['wind_kph_lag_1d'] = wind_base
    df['wind_kph_lag_2d'] = wind_base
    df['uv_index_lag_1d'] = uv_base
    df['uv_index_lag_2d'] = uv_base
    df['cloud_cover_lag_1d'] = cloud_base
    df['cloud_cover_lag_2d'] = cloud_base
    df['temperature_celsius_lag_1d'] = temp_base
    df['temperature_celsius_lag_2d'] = temp_base
    df['wind_kph_roll_mean_3d'] = wind_base
    df['wind_kph_roll_std_3d'] = 2.0
    df['temperature_celsius_roll_mean_3d'] = temp_base
    df['temperature_celsius_roll_std_3d'] = 1.0

    return df, hours

forecast_features_df, forecast_hours = generate_40_feature_forecast(base_wind, base_cloud, base_uv)

# ==========================================
# 5. GENERATE LIVE PREDICTIONS
# ==========================================
forecast_results = pd.DataFrame({"hour": forecast_hours})

if wind_model and solar_model:
    wind_expected = wind_model.regressor_.feature_names_in_ if hasattr(wind_model, 'regressor_') else wind_model.feature_names_in_
    solar_expected = solar_model.regressor_.feature_names_in_ if hasattr(solar_model, 'regressor_') else solar_model.feature_names_in_
    
    aligned_wind_df = pd.DataFrame()
    for col in wind_expected:
        aligned_wind_df[col] = forecast_features_df.get(col, 0.0)
            
    aligned_solar_df = pd.DataFrame()
    for col in solar_expected:
        aligned_solar_df[col] = forecast_features_df.get(col, 0.0)

    wind_pred_raw = wind_model.predict(aligned_wind_df)
    solar_pred = solar_model.predict(aligned_solar_df)
    
    forecast_results["wind_power"] = np.clip(wind_pred_raw, 0, None)
    forecast_results["solar_power"] = np.clip(solar_pred, 0, None)
else:
    forecast_results["wind_power"] = forecast_features_df["wind_kph"] ** 3
    forecast_results["solar_power"] = forecast_features_df["uv_index"] * (1 - forecast_features_df["cloud_cover"]/100) * 10000

forecast_results["total_power"] = forecast_results["wind_power"] + forecast_results["solar_power"]

# ==========================================
# 6. MAIN DASHBOARD UI
# ==========================================
st.title("Next-Day Grid Dispatch Forecast")
st.markdown("Monitor forecasted renewable supply against baseline grid demand in real-time.")
st.write("") # Spacer

total_supply_peak = forecast_results["total_power"].max()
total_deficit_hours = len(forecast_results[forecast_results["total_power"] < target_demand])
minimum_margin = (forecast_results["total_power"] - target_demand).min()

# Clean Metric Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Peak Forecasted Supply", f"{total_supply_peak:,.0f} MW")
with col2:
    st.metric("Target Grid Demand", f"{target_demand:,.0f} MW")
with col3:
    if total_deficit_hours > 0:
        st.error(f"⚠️ Deficit Warning: {total_deficit_hours} Hours")
    else:
        st.success(f" Grid Stable (Min Margin: {minimum_margin:,.0f} MW)")

st.write("")
st.write("")

# ==========================================
# 7. HOURLY DATA TABLE
# ==========================================
# Displaying the data table prominently as the primary visual output
st.subheader("Hourly Forecast Breakdown")
st.markdown("Detailed tabular view of forecasted weather conditions and expected power output over the next 24 hours.")

display_df = forecast_results.copy()
display_df["Wind Spd (kph)"] = forecast_features_df["wind_kph"]
display_df["Cloud (%)"] = forecast_features_df["cloud_cover"]
display_df["UV"] = forecast_features_df["uv_index"]

# Reorder columns for readability
display_df = display_df[["hour", "Wind Spd (kph)", "Cloud (%)", "UV", "wind_power", "solar_power", "total_power"]]
display_df.columns = ["Hour", "Wind Spd (kph)", "Cloud (%)", "UV", "Wind Power (MW)", "Solar Power (MW)", "Total Power (MW)"]

st.dataframe(
    display_df.style.format("{:.1f}").background_gradient(cmap='Blues', subset=['Total Power (MW)']), 
    use_container_width=True,
    height=600
)