# src/train_xg.py
"""
Train XGBoost regressor using lag & rolling features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Adjusting DATA_DIR and MODELS_DIR for Colab environment where __file__ is not defined
DATA_DIR = os.path.abspath(os.path.join('.', 'data')) # This will point to '/content/data'
MODELS_DIR = os.path.abspath(os.path.join('.', 'models')) # This will point to '/content/models'
os.makedirs(MODELS_DIR, exist_ok=True)

def train_xg_for(coin='bitcoin'):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    X = df.drop(columns=['price','volume','market_cap'])
    y = df['price'].values
    n = len(X)
    train_end = int(0.8*n)
    val_end = int(0.9*n)
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective':'reg:squarederror',
        'learning_rate':0.05,
        'max_depth':6,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'seed':42,
        'verbosity':0
    }
    evallist = [(dtrain,'train'),(dval,'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=50, evals=evallist, verbose_eval=50)
    preds = bst.predict(xgb.DMatrix(X_test))
    mae = mean_absolute_error(y_test, preds)
    # Removed 'squared=False' as it's not supported in older sklearn versions and computed RMSE manually.
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"XGBoost {coin} MAE={mae:.4f} RMSE={rmse:.4f}")
    joblib.dump(bst, os.path.join(MODELS_DIR, f'{coin}_xg_model.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, f'{coin}_xg_features.pkl'))

if __name__ == "__main__":
    for c in ['bitcoin','ethereum']:
        print("Training XG on", c)
        train_xg_for(c)
