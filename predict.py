# src/predict.py
"""
Unified predict API:
- can load XGBoost or LSTM model and produce next-N day predictions (iterative)
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model

# Adjusting DATA_DIR and MODELS_DIR for Colab environment where __file__ is not defined
DATA_DIR = os.path.abspath(os.path.join('.', 'data')) # This will point to '/content/data'
MODELS_DIR = os.path.abspath(os.path.join('.', 'models')) # This will point to '/content/models'

def predict_next_days_xg(coin='bitcoin', n_days=7):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    features = joblib.load(os.path.join(MODELS_DIR, f'{coin}_xg_features.pkl'))
    model = joblib.load(os.path.join(MODELS_DIR, f'{coin}_xg_model.pkl'))
    last_row = df.iloc[-1:].copy()
    preds = []
    cur = last_row.copy()
    for i in range(n_days):
        Xcur = cur[features]
        dX = xgb.DMatrix(Xcur)
        p = model.predict(dX)[0]
        preds.append(p)
        for lag in range(14,1,-1):
            cur[f'lag_{lag}'] = cur[f'lag_{lag-1}']
        cur['lag_1'] = p
        for w in [7,14,30]:
            cur[f'roll_mean_{w}'] = (cur[f'roll_mean_{w}'] * (w-1) + p) / w
    start = df.index[-1] + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n_days, freq='D')
    return pd.DataFrame({'predicted_price': preds}, index=idx)

def predict_next_days_lstm(coin='bitcoin', n_days=7, seq_len=30):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, f'{coin}_lstm_features.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{coin}_lstm_scaler.pkl'))
    model = load_model(os.path.join(MODELS_DIR, f'{coin}_lstm_final.h5'))
    X_raw = df[feature_cols].values
    X_scaled = scaler.transform(X_raw)
    seq = X_scaled[-seq_len:].copy()
    preds = []
    cur_seq = seq.copy()
    for i in range(n_days):
        X_in = cur_seq.reshape(1, cur_seq.shape[0], cur_seq.shape[1])
        p = model.predict(X_in)[0,0]
        preds.append(p)
        new_row = cur_seq[-1].copy()
        cur_seq = np.vstack([cur_seq[1:], new_row])
    start = df.index[-1] + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n_days, freq='D')
    return pd.DataFrame({'predicted_price': preds}, index=idx)
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model # Not used in predict_next_days_xg, but good to keep if predict_next_days_lstm is intended later

# Adjusting DATA_DIR and MODELS_DIR for Colab environment
DATA_DIR = os.path.abspath(os.path.join('.', 'data'))
MODELS_DIR = os.path.abspath(os.path.join('.', 'models'))

def predict_next_days_xg(coin='bitcoin', n_days=7):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    features = joblib.load(os.path.join(MODELS_DIR, f'{coin}_xg_features.pkl'))
    model = joblib.load(os.path.join(MODELS_DIR, f'{coin}_xg_model.pkl'))
    last_row = df.iloc[-1:].copy()
    preds = []
    cur = last_row.copy()
    for i in range(n_days):
        Xcur = cur[features]
        dX = xgb.DMatrix(Xcur)
        p = model.predict(dX)[0]
        preds.append(p)
        for lag in range(14,1,-1):
            cur[f'lag_{lag}'] = cur[f'lag_{lag-1}']
        cur['lag_1'] = p
        for w in [7,14,30]:
            cur[f'roll_mean_{w}'] = (cur[f'roll_mean_{w}'] * (w-1) + p) / w
    start = df.index[-1] + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n_days, freq='D')
    return pd.DataFrame({'predicted_price': preds}, index=idx)

# The predict_next_days_lstm function from src/predict.py is not included here 
# as it was not part of the current import statement causing the error.
# If you intend to use it, you would need to embed it similarly or create actual Python files.

print(predict_next_days_xg('bitcoin', 5))
