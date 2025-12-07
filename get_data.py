# src/get_data.py
"""
Fetch historical crypto market data from CoinGecko and save CSV.
"""


import os
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime, timezone # Import timezone

# Adjusting OUT_DIR for Colab environment where __file__ is not defined
OUT_DIR = os.path.abspath(os.path.join('.', 'data')) # This will create 'data' in the current working directory (/content)
os.makedirs(OUT_DIR, exist_ok=True)

def futurecoin_fetcher(coin_id='bitcoin', vs_currency='usd', days='max'):
    cg = CoinGeckoAPI()
    raw = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    df_p = pd.DataFrame(raw['prices'], columns=['timestamp','price'])
    df_v = pd.DataFrame(raw['total_volumes'], columns=['timestamp','volume'])
    df_m = pd.DataFrame(raw['market_caps'], columns=['timestamp','market_cap'])
    df = df_p.merge(df_v, on='timestamp').merge(df_m, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp').sort_index()
    return df

if __name__ == "__main__":
    for coin in ['bitcoin', 'ethereum']:
        print(f"Fetching {coin} ...")
        df = futurecoin_fetcher(coin_id=coin, days='365') # Changed from '3650' to '365'
        out = os.path.join(OUT_DIR, f'{coin}_raw.csv')
        df.to_csv(out)
        print(f"Saved {out} rows={len(df)} at {datetime.now(timezone.utc).isoformat()}Z") # Updated to use datetime.now(timezone.utc)
