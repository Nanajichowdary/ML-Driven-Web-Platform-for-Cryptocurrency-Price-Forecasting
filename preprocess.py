# src/preprocess.py
"""
Preprocess pipeline:
- load raw CSV
- resample to daily close
- fill missing days
- save processed parquet
"""

import os
import pandas as pd

# Adjusting DATA_DIR for Colab environment where __file__ is not defined
DATA_DIR = os.path.abspath(os.path.join('.', 'data')) # This will point to '/content/data'
RAW_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_raw.csv')]

def futurecoin_cleanser(path, freq='D'):
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    df = df.resample(freq).agg({'price':'last','volume':'last','market_cap':'last'})
    # Updated to avoid FutureWarning regarding inplace operations on chained assignments
    df['price'] = df['price'].ffill()
    df['volume'] = df['volume'].ffill()
    df['market_cap'] = df['market_cap'].ffill()
    # Updated to avoid FutureWarning regarding DataFrame.fillna with 'method'
    df.bfill(inplace=True)
    q1, q99 = df['price'].quantile(0.01), df['price'].quantile(0.99)
    df['price'] = df['price'].clip(lower=q1, upper=q99)
    return df

if __name__ == "__main__":
    for raw in RAW_FILES:
        coin = os.path.basename(raw).replace('_raw.csv','')
        print("Preprocessing", coin)
        df = futurecoin_cleanser(raw)
        out = os.path.join(DATA_DIR, f'{coin}_daily.parquet')
        df.to_parquet(out)
        print("Saved", out, "rows:", len(df))
