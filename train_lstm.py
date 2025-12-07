# src/train_lstm.py
"""
Train a simple LSTM on sequences of price with multivariate features.
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Adjusting DATA_DIR and MODELS_DIR for Colab environment where __file__ is not defined
DATA_DIR = os.path.abspath(os.path.join('.', 'data')) # This will point to '/content/data'
MODELS_DIR = os.path.abspath(os.path.join('.', 'models')) # This will point to '/content/models'
os.makedirs(MODELS_DIR, exist_ok=True)

SEQ_LEN = 30

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_lstm_for(coin='bitcoin'):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    feature_cols = [c for c in df.columns if c not in ['price','volume','market_cap']]
    X_raw = df[feature_cols].values
    y_raw = df['price'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{coin}_lstm_scaler.pkl'))
    X_seq, y_seq = prepare_sequences(X_scaled, y_raw, SEQ_LEN)
    n = len(X_seq)
    train_end = int(0.8*n)
    val_end = int(0.9*n)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    ckpt_path = os.path.join(MODELS_DIR, f'{coin}_lstm_best.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss')
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks, verbose=2)
    preds = model.predict(X_test).squeeze()
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds)) # Removed 'squared=False' as it's not supported in older sklearn versions and computed RMSE manually.
    print(f"LSTM {coin} MAE={mae:.4f} RMSE={rmse:.4f}")
    model.save(os.path.join(MODELS_DIR, f'{coin}_lstm_final.h5'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, f'{coin}_lstm_features.pkl'))

if __name__ == "__main__":
    for c in ['bitcoin','ethereum']:
        print("Training LSTM on", c)
        train_lstm_for(c)
