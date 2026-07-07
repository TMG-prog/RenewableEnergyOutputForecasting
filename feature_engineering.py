import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find the input file at: {os.path.abspath(input_path)}\n"
            "Please verify Member 1's cleaned file name matches perfectly!"
        )
    print("⚡ Loading preprocessed weather data from nested directory...")
    df = pd.read_csv(input_path)
    
    # ─── COLUMN MAPPING ───────────────────────────────────────────────────
    # Map initialcolumn names to what our feature engineering math expects
    df = df.rename(columns={
        'last_updated': 'datetime',
        'cloud': 'cloud_cover'
    })
    
    # Sort by location AND time so rows aren't a mixed-up deck of countries
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['location_name', 'datetime']).reset_index(drop=True)
    
    print("Engineering target power proxies (today's realized output)")
    # Wind Power Proxy: Proportional to the cube of wind speed
    df['wind_power_proxy'] = df['wind_kph'] ** 3
    
    # Solar Power Proxy: UV Index scaled down by Cloud Cover percentage
    clear_factor = 1.0 - (df['cloud_cover'] / 100.0)
    df['solar_power_proxy'] = df['uv_index'] * clear_factor
    
    print("Grouping data for country-level representation")
    # Calculates the regional daily baseline for that country on that specific date
    df['country_daily_mean_temp'] = df.groupby(['country', 'datetime'])['temperature_celsius'].transform('mean')
    df['country_daily_mean_wind'] = df.groupby(['country', 'datetime'])['wind_kph'].transform('mean')
    
    print("Encoding cyclical time coordinates")
    # Convert hours and days into sine/cosine wave coordinates
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    print("Building isolated daily lag features")
    # Grouping by location_name prevents tracking errors across different regional timelines
    target_lags = ['wind_kph', 'uv_index', 'cloud_cover', 'temperature_celsius']
    for col in target_lags:
        df[f'{col}_lag_1d'] = df.groupby('location_name')[col].shift(1)
        df[f'{col}_lag_2d'] = df.groupby('location_name')[col].shift(2)
        
    print("Calculating isolated 3-day rolling metrics")
    # Calculates rolling updates locally so Location A doesn't roll into Location B's rows
    for col in ['wind_kph', 'temperature_celsius']:
        df[f'{col}_roll_mean_3d'] = df.groupby('location_name')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        # Apply standard deviation computation first, then safely resolve initial NaN edge cases to 0
        df[f'{col}_roll_std_3d'] = df.groupby('location_name')[col].transform(lambda x: x.rolling(window=3, min_periods=1).std()).fillna(0)
        
    print("Shifting targets forward so the model forecasts NEXT-DAY output")
    # Everything computed above (raw weather, lag_1d/lag_2d, rolling means/stds,
    # country-level daily means) reflects information known as of "today" for
    # each row. Up to this point, wind_power_proxy / solar_power_proxy are
    # TODAY's realized output — training on them directly would just teach the
    # model to re-derive wind_kph**3 and uv_index*clear_factor, which is
    # leakage, not forecasting.
    #
    # Shifting each target back by one row *within its own location's
    # timeline* re-labels each row with TOMORROW's power proxy instead, while
    # every feature column stays "today's" value. That turns this into a
    # genuine next-day forecasting problem: predict tomorrow's wind/solar
    # power potential using only information available today (current +
    # lagged/rolling weather). The last day on record for each location has
    # no "tomorrow" to attach, so it becomes NaN here and is dropped below —
    # same mechanism already used to drop the lag warm-up rows.
    df['wind_power_proxy'] = df.groupby('location_name')['wind_power_proxy'].shift(-1)
    df['solar_power_proxy'] = df.groupby('location_name')['solar_power_proxy'].shift(-1)

    # Clean up empty boundary rows introduced by shifting operations
    # (lag warm-up rows at the start of each location's timeline, and the
    # final day of each location's timeline, which has no next-day target)
    df = df.dropna().reset_index(drop=True)
    
    # Drop unneeded raw helper columns
    df = df.drop(columns=['hour', 'day_of_year'])
    
    # Save the file into the cleaned data directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Processed dataset shape: {df.shape}")
    print(f"📂 Final feature dataset saved safely at: {output_path}")

if __name__ == "__main__":
    INPUT = r"Project\data\cleaned\GlobalWeatherRepository_Cleaned_Optimized.csv"
    OUTPUT = r"Project\data\cleaned\processed_features.csv"
    
    engineer_features(INPUT, OUTPUT)