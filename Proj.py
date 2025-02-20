import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST

# Alpaca API Keys (Replace with your own keys)
ALPACA_API_KEY = "PKEISJ9Y9ALRA3K143GD"
ALPACA_SECRET_KEY = "zWUYvjUbsk0n8CkZffeSJ2QuSaOcW0K4RyZU2ZoV"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# Fetch Solana (SOL/USD) historical data (past 7 days, 1-minute interval)
bars = api.get_crypto_bars(["SOL/USD"], timeframe="1Min").df

# Debugging: Print DataFrame preview
print("Alpaca API Response Preview:")
print(bars.head())

# Check if data was received
if bars.empty:
    print("Error: No data received from Alpaca API. Check your API key or symbol.")
    exit()

# Reset index to ensure "symbol" is accessible
bars = bars.reset_index()

# Ensure "symbol" column exists before filtering
if "symbol" in bars.columns:
    sol_data = bars[bars["symbol"] == "SOL/USD"].copy()
    sol_data = sol_data.drop(columns=["symbol"])  # Drop 'symbol' column if unnecessary
else:
    print("Error: 'symbol' column not found. Check API response format.")
    exit()

# Select relevant columns
sol_data = sol_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Convert timestamp to datetime & sort
sol_data['timestamp'] = pd.to_datetime(sol_data['timestamp'])
sol_data = sol_data.sort_values(by="timestamp").reset_index(drop=True)

# Print dataset shape and first few rows
print(f"Dataset shape: {sol_data.shape}")
print(sol_data.head())

# Save dataset
sol_data.to_csv("raw_solana_data.csv", index=False)
