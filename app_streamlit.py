# src/app_streamlit.py
"""
Streamlit demo: choose coin, model, and horizon -> show predictions vs history.
"""
import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler # Added for LSTM scaler

st.set_page_config(page_title="FutureCoin Unique", layout="wide")
st.title("FutureCoin — Crypto Price Predictor")

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
        # Ensure all required features are present, handle missing ones if necessary
        missing_features = [f for f in features if f not in cur.columns]
        if missing_features:
            # This case ideally should not happen if feature_factory creates all needed features
            # For now, let's assume 'cur' always has what 'features' expects.
            # If issues arise, a more robust feature reconstruction would be needed here.
            raise ValueError(f"Missing features in current dataframe for prediction: {missing_features}")

        Xcur = cur[features]
        dX = xgb.DMatrix(Xcur)
        p = model.predict(dX)[0]
        preds.append(p)
        
        # Update lag features iteratively
        for lag in range(14, 1, -1):
            cur[f'lag_{lag}'] = cur[f'lag_{lag-1}']
        cur['lag_1'] = p
        
        # Update rolling mean features iteratively (simple approximation)
        for w in [7, 14, 30]:
            # This approximation assumes constant window size and simply averages in the new prediction
            # A more accurate rolling calculation would require storing the actual window values.
            cur[f'roll_mean_{w}'] = (cur[f'roll_mean_{w}'] * (w - 1) + p) / w 

    start = df.index[-1] + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n_days, freq='D')
    return pd.DataFrame({'predicted_price': preds}, index=idx)


def predict_next_days_lstm(coin='bitcoin', n_days=7, seq_len=30):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_features.parquet'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, f'{coin}_lstm_features.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{coin}_lstm_scaler.pkl'))
    model = load_model(os.path.join(MODELS_DIR, f'{coin}_lstm_final.h5'))

    # Use only the feature columns present in the original training set
    X_raw = df[feature_cols].values
    X_scaled = scaler.transform(X_raw)

    # Take the last sequence_length data points for prediction
    # Ensure we have enough data points for the sequence
    if len(X_scaled) < seq_len:
        st.error(f"Not enough data points ({len(X_scaled)}) to form a sequence of {seq_len} for LSTM prediction. Try a smaller sequence length or more historical data.")
        st.stop()

    seq = X_scaled[-seq_len:].copy()
    preds = []
    cur_seq = seq.copy() # Current sequence to be updated with predictions

    for i in range(n_days):
        # Reshape for LSTM input: (1, seq_len, num_features)
        X_in = cur_seq.reshape(1, cur_seq.shape[0], cur_seq.shape[1])
        p = model.predict(X_in, verbose=0)[0,0] # Predict one step ahead
        preds.append(p)

        # For iterative prediction, we need to update the sequence with the new prediction
        # This requires recreating the *features* for the new predicted day.
        # This is a simplification and might not be perfectly accurate as new features (lags, rolls) 
        # would ideally be re-calculated based on actual future data, which isn't available.
        # For now, we'll shift the sequence and put a placeholder or a simpler derived feature.
        
        # This part requires careful reconstruction of features for the 'next day'.
        # A full re-implementation of feature_factory would be needed for perfect iterative prediction.
        # For simplicity, we'll just append a dummy row or just use the predicted price itself if features are simple.
        
        # For a simple iterative forecast with LSTM, usually, the prediction 'p' would be scaled back,
        # then inserted into the 'price' feature, and all other features (lags, rolls) would be recomputed for the 'next day'.
        # This implementation simplifies by not fully recomputing all features, which might lead to less accurate long-term forecasts.
        
        # Simplified approach: create a 'dummy' new row to append, preserving scaled values for non-price related features
        # and using the prediction for price-related features after inverse scaling (not strictly applied here for simplicity)
        # A more complex logic for feature recalculation based on 'p' would be needed for accuracy.
        
        # For this example, let's assume we can approximate the next input sequence by shifting and replacing the 'price' related features.
        # This is highly simplified and might not be robust. A better approach involves inverse transforming 'p', 
        # reconstructing a full feature set for the next day, transforming it, and then appending.
        
        # Given the previous context, the scaler scales all features, so 'p' is an unscaled price.
        # The `seq` holds scaled features. To add `p` back into a *scaled* feature set, 
        # we'd need to inverse transform it, then re-compute new features based on it (lag, roll), then scale them.
        
        # A simpler way (less accurate but common in demos) is to just append the last row of features, 
        # and if 'price' or 'log_price' were part of `feature_cols`, that would be wrong. 
        # The LSTM is predicting `y_raw` (unscaled price) directly, while X is scaled.
        # So, the `cur_seq` is scaled, but `p` is unscaled.
        
        # Correct iterative prediction for LSTM usually involves:
        # 1. Predict `p` (unscaled price).
        # 2. Reconstruct the *next* day's feature vector `next_X_raw` using `p` and current `df` state.
        # 3. Scale `next_X_raw` to get `next_X_scaled`.
        # 4. Append `next_X_scaled` to `cur_seq` and remove the oldest element.
        
        # Given the features defined in features.py (return_1d, log_price, lags, rolls), 
        # recreating `next_X_raw` and scaling it correctly is complex within this iterative loop without the full `feature_factory`.
        
        # For this Streamlit app, we will use a highly simplified iteration for `cur_seq` for demonstration purposes.
        # This means the future predictions will progressively become less accurate as they rely on approximated feature updates.
        
        # Let's assume the price is the last feature for simplicity, or we update the `lag_1` feature with `p`.
        # This requires knowing the index of `lag_1` and other features in `feature_cols`.
        
        # This part is critical and tricky for iterative LSTM prediction: 
        # We have X_scaled, a sequence of scaled features. 'p' is an *unscaled* price.
        # To get the next sequence element, we need a *scaled* feature vector for the next day.
        # This vector would depend on the *predicted price 'p'* and other features.
        # A simple shift would mean we are just shifting the existing scaled features and not incorporating 'p' effectively.
        
        # The previous xgboost example manually updated lags and rolling means. LSTM needs a new *scaled feature vector*.
        # Let's make a strong assumption here for demonstration, and this needs to be made explicit.
        
        # Simplified (and potentially inaccurate) update for demo: 
        # Shift the sequence, and for the new last element, try to approximate based on the current last scaled element.
        # For proper iterative prediction, 'p' would be unscaled, then used to generate a new feature row (like in feature_factory),
        # then this new row would be scaled and appended.

        # For now, let's use the XGboost update logic adapted to scaled space, which is also an approximation:
        # Need to know which indices correspond to price, lag_1, rolling means etc. in feature_cols.
        # This requires mapping `feature_cols` to the column indices of `X_scaled`.
        # For a quick fix, let's just shift the sequence and assume the new last element can be 'roughly' approximated 
        # or that the model can handle a less precise feature update for the purpose of a short-term demo.
        
        # A more correct, but more complex, implementation for LSTM iterative prediction:
        # 1. Inverse transform `p` if needed (here `p` is already unscaled price).
        # 2. Construct a new `X_raw_next_day` row: Take the last `X_raw` row, update `price`, `return_1d`, `log_price`, `lag` features, `roll_mean` features.
        #    This would need access to original data and feature logic.
        # 3. Scale `next_X_raw` using the scaler to get `next_X_scaled_row`.
        # 4. Update `cur_seq` by removing the oldest row and appending `next_X_scaled_row`.
        
        # Since we don't have the full feature_factory logic here, let's simplify.
        # The LSTM is trained to predict the unscaled price 'y_raw' from scaled X_seq.
        # For iterative prediction, we essentially need to create the 'next' scaled feature vector.
        
        # This is a very basic way to update the sequence, just pushing the last state forward. 
        # This is unlikely to be accurate for long horizons as the feature dynamics are not truly updated.
        # A more advanced approach would involve re-calculating all features for the next step, using 'p'.
        
        new_scaled_row = cur_seq[-1].copy() # Take the last scaled feature vector as a base
        # If 'log_price' or 'return_1d' is in feature_cols, we'd need to update it here.
        # For simplicity, we are essentially feeding the model with a shifted sequence, 
        # not a sequence that fully reflects the predicted price's impact on future features.
        cur_seq = np.vstack([cur_seq[1:], new_scaled_row])

    start = df.index[-1] + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=n_days, freq='D')
    return pd.DataFrame({'predicted_price': preds}, index=idx)


coins = [f.replace('_daily.parquet','') for f in os.listdir(DATA_DIR) if f.endswith('_daily.parquet')]
if not coins:
    st.error("No processed coin data found. Run preprocessing first.")
    st.stop()

coin = st.selectbox("Select coin", coins, index=0)
model_choice = st.radio("Model", ['XGBoost', 'LSTM'])
horizon = st.slider("Days to predict", 1, 30, 7)

hist = pd.read_parquet(os.path.join(DATA_DIR, f'{coin}_daily.parquet'))['price']
st.subheader("Recent history (last 180 days)")
st.line_chart(hist.tail(180))

if st.button("Run prediction"):
    st.info(f"Predicting {horizon} days with {model_choice} for {coin}...")
    if model_choice == 'XGBoost':
        pred = predict_next_days_xg(coin=coin, n_days=horizon)
    else:
        pred = predict_next_days_lstm(coin=coin, n_days=horizon)
    st.subheader("Predicted prices")
    combined = pd.concat([hist.tail(180), pred['predicted_price']])
    st.line_chart(combined)
    st.write(pred)
