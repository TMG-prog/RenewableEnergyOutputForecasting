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
    # Map Member 1's column names to what our feature engineering math expects
    df = df.rename(columns={
        'last_updated': 'datetime',
        'cloud': 'cloud_cover'
    })
    
    # Force chronological sorting to ensure lag and rolling features align perfectly
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print("📈 Engineering target power proxies...")
    # Wind Power Proxy: Proportional to the cube of wind speed
    df['wind_power_proxy'] = df['wind_kph'] ** 3
    
    # Solar Power Proxy: UV Index scaled down by Cloud Cover percentage
    clear_factor = 1.0 - (df['cloud_cover'] / 100.0)
    df['solar_power_proxy'] = df['uv_index'] * clear_factor
    
    print("🔄 Encoding cyclical time coordinates...")
    # Convert hours and days into sine/cosine wave coordinates
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    print("⏳ Building historical lag features (1h, 6h, 24h)...")
    target_lags = ['wind_kph', 'uv_index', 'cloud_cover', 'temperature_celsius']
    for col in target_lags:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_6h'] = df[col].shift(6)
        df[f'{col}_lag_24h'] = df[col].shift(24)
        
    print("📊 Calculating 24-hour rolling metrics...")
    for col in ['wind_kph', 'temperature_celsius']:
        df[f'{col}_roll_mean_24h'] = df[col].rolling(window=24).mean()
        df[f'{col}_roll_std_24h'] = df[col].rolling(window=24).std()
        
    # Clean up empty boundary rows introduced by rolling operations
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