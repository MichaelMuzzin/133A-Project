import sys
import os
import pandas as pd
import numpy as np
import talib  # Technical Indicators
from alpaca_trade_api.rest import REST
import time
import matplotlib.pyplot as plt

#make sure python see's correct packages
site_packages_path = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

print(f"Using Python version: {sys.version}")
print(f"Python path: {sys.path}")

#ALPACA API
ALPACA_API_KEY = "PKEISJ9Y9ALRA3K143GD"
ALPACA_SECRET_KEY = "zWUYvjUbsk0n8CkZffeSJ2QuSaOcW0K4RyZU2ZoV"
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

#Get at LEAST 10,000 rows
raw_rows_to_fetch = 20000  
batch_size = 1000  
interval = "1Min"
bars_list = []

for i in range(0, raw_rows_to_fetch, batch_size):
    print(f"Fetching batch {i // batch_size + 1}...")
    batch_bars = api.get_crypto_bars(["SOL/USD"], timeframe=interval, limit=batch_size).df
    bars_list.append(batch_bars)
    time.sleep(1)  # Prevent hitting API rate limits

bars = pd.concat(bars_list, ignore_index=True)

# Ensure timestamp is available
bars = bars.reset_index()
if 'index' in bars.columns:
    bars['timestamp'] = bars['index']
    bars = bars.drop(columns=['index'])

print(f"Columns in the raw data: {bars.columns.tolist()}")

# Filter for SOL/USD and keep the symbol column (to have these base columns: timestamp, symbol, open, high, low, close, volume)
if "symbol" in bars.columns:
    sol_data = bars[bars["symbol"] == "SOL/USD"].copy()
else:
    print("Error: 'symbol' column not found.")
    exit()

if 'timestamp' not in sol_data.columns:
    print("Error: 'timestamp' column not found.")
    exit()

# Reorder columns and convert timestamp to datetime
sol_data = sol_data[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
sol_data['timestamp'] = pd.to_datetime(sol_data['timestamp'])
sol_data = sol_data.sort_values(by="timestamp").reset_index(drop=True)

# Compute 30 Technical Indicators
sol_data['SMA_10']        = talib.SMA(sol_data['close'], timeperiod=10)
sol_data['SMA_50']        = talib.SMA(sol_data['close'], timeperiod=50)
sol_data['EMA_10']        = talib.EMA(sol_data['close'], timeperiod=10)
sol_data['EMA_50']        = talib.EMA(sol_data['close'], timeperiod=50)
sol_data['MACD'], sol_data['MACD_Signal'], _ = talib.MACD(sol_data['close'])
sol_data['CCI_14']        = talib.CCI(sol_data['high'], sol_data['low'], sol_data['close'], timeperiod=14)
sol_data['BB_upper'], _, sol_data['BB_lower'] = talib.BBANDS(sol_data['close'], timeperiod=20)
sol_data['ATR_14']        = talib.ATR(sol_data['high'], sol_data['low'], sol_data['close'], timeperiod=14)
sol_data['STDDEV_14']     = talib.STDDEV(sol_data['close'], timeperiod=14)
sol_data['RSI_14']        = talib.RSI(sol_data['close'], timeperiod=14)
sol_data['WILLR_14']      = talib.WILLR(sol_data['high'], sol_data['low'], sol_data['close'], timeperiod=14)
sol_data['STOCH_K'], sol_data['STOCH_D'] = talib.STOCH(sol_data['high'], sol_data['low'], sol_data['close'])
sol_data['ROC_14']        = talib.ROC(sol_data['close'], timeperiod=14)
sol_data['MOM_14']        = talib.MOM(sol_data['close'], timeperiod=14)
sol_data['OBV']           = talib.OBV(sol_data['close'], sol_data['volume'])
sol_data['AD_Line']       = talib.AD(sol_data['high'], sol_data['low'], sol_data['close'], sol_data['volume'])
sol_data['MFI_14']        = talib.MFI(sol_data['high'], sol_data['low'], sol_data['close'], sol_data['volume'], timeperiod=14)
sol_data['Log_Returns']   = np.log(sol_data['close'] / sol_data['close'].shift(1))
sol_data['Pct_Change_1']  = sol_data['close'].pct_change(1)
sol_data['Pct_Change_5']  = sol_data['close'].pct_change(5)

# Drop rows with NaN values (from rolling window calculations)
sol_data.dropna(inplace=True)

print(f"Final dataframe shape: {sol_data.shape}")
if sol_data.shape[0] < 10000:
    print(f"Warning: Final dataframe has only {sol_data.shape[0]} rows, which is less than 10,000.")
else:
    print(" Dataframe has at least 10,000 rows.")

# Save the dataframe to CSV
sol_data.to_csv("solana_trading_features_10000_1Min.csv", index=False)
print("Data saved to 'solana_trading_features_10000_1Min.csv'.")

#Plotting with matplotlib ( if you want to )
plt.figure(figsize=(14, 7))
plt.plot(sol_data['timestamp'], sol_data['close'], label='Close Price', color='blue', alpha=0.7)
plt.plot(sol_data['timestamp'], sol_data['SMA_10'], label='SMA 10', color='orange', linewidth=1.5)
plt.plot(sol_data['timestamp'], sol_data['SMA_50'], label='SMA 50', color='green', linewidth=1.5)
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.title("SOL/USD Close Price with SMA 10 & SMA 50")
plt.legend()
plt.tight_layout()
plt.show()
