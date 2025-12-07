# src/features.py
"""
Create lag/rolling/time features and save processed dataset for modeling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Adjusting DATA_DIR and MODELS_DIR for Colab environment where __file__ is not defined
DATA_DIR = os.path.abspath(os.path.join('.', 'data')) # This will point to '/content/data'
MODELS_DIR = os.path.abspath(os.path.join('.', 'models')) # This will point to '/content/models'
os.makedirs(MODELS_DIR, exist_ok=True)

LAGS = 14
ROLL_WINDOWS = [7,14,30]

def feature_factory(df, lags=LAGS, rolls=ROLL_WINDOWS):
    df = df.copy()
    df['return_1d'] = df['price'].pct_change()
    df['log_price'] = np.log1p(df['price'])
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['price'].shift(i)
    for w in rolls:
        df[f'roll_mean_{w}'] = df['price'].rolling(window=w).mean()
        df[f'roll_std_{w}'] = df['price'].rolling(window=w).std()
    df['dow'] = df.index.dayofweek
    df['dom'] = df.index.day
    df['month'] = df.index.month
    df = df.dropna().copy()
    return df

def fit_scaler_and_save(df, feature_cols, scaler_path):
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    joblib.dump(scaler, scaler_path)
    return scaler

if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if file.endswith('_daily.parquet'):
            coin = file.replace('_daily.parquet','')
            path = os.path.join(DATA_DIR, file)
            print("Feature-engineering:", coin)
            df = pd.read_parquet(path)
            df_feat = feature_factory(df)
            feature_cols = [c for c in df_feat.columns if c not in ['price','volume','market_cap']]
            scaler_path = os.path.join(MODELS_DIR, f'{coin}_lstm_scaler.pkl')
            fit_scaler_and_save(df_feat, feature_cols, scaler_path)
            out = os.path.join(DATA_DIR, f'{coin}_features.parquet')
            df_feat.to_parquet(out)
            print("Saved features:", out, "rows:", len(df_feat))
