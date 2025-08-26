import pandas as pd
import yfinance as yf
import ta # Technical Analysis library
from datetime import datetime, timedelta
import numpy as np # Explicitly import numpy

# List of some SENSEX blue-chip stock tickers (adjust as needed)
# You might need to verify the exact ticker symbols for yfinance for Indian stocks (e.g., .NS for NSE)
# Example: Reliance Industries -> RELIANCE.NS
sensex_tickers = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",       # Tata Consultancy Services
    # "HDFCBANK.NS",  # HDFC Bank - Removed for brevity in demonstration
    # "ICICIBANK.NS", # ICICI Bank - Removed for brevity in demonstration
    # "INFY.NS",      # Infosys - Removed for brevity in demonstration
    # "HINDUNILVR.NS" # Hindustan Unilever - Removed for brevity in demonstration
]

def fetch_historical_data(ticker, period="1y", interval="1d"):
    """Fetches historical data for a given ticker."""
    print(f"[DEBUG] Inside fetch_historical_data for {ticker}")
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        if not stock_data.empty:
            print(f"[DEBUG] Successfully downloaded {len(stock_data)} data points for {ticker}")
            return stock_data
        else:
            print(f"[DEBUG] No data fetched for {ticker}. Check ticker symbol or period.")
            return None
    except Exception as e:
        print(f"[ERROR] Error fetching data for {ticker}: {e}")
        return None

def generate_features(df):
    """Generates a set of technical indicators and other features from raw stock data."""
    print("[DEBUG] Starting feature generation...")
    # Ensure the DataFrame has the correct column names (Open, High, Low, Close, Volume)
    # yfinance typically provides these.

    # Ensure columns are Series by using .squeeze() if needed (though direct access usually returns Series)
    close_prices = df['Close'].squeeze()
    high_prices = df['High'].squeeze()
    low_prices = df['Low'].squeeze()
    volumes = df['Volume'].squeeze()
    
    print(f"[DEBUG] Type of close_prices after squeeze: {type(close_prices)}")
    print(f"[DEBUG] Shape of close_prices after squeeze: {close_prices.shape}")

    # 1. Moving Averages
    df['SMA_10'] = ta.trend.sma_indicator(close_prices, window=10)
    df['EMA_10'] = ta.trend.ema_indicator(close_prices, window=10)
    df['SMA_30'] = ta.trend.sma_indicator(close_prices, window=30)
    df['EMA_30'] = ta.trend.ema_indicator(close_prices, window=30)

    # 2. Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(close_prices, window=14)

    # 3. Moving Average Convergence Divergence (MACD)
    df['MACD'] = ta.trend.macd(close_prices)
    df['MACD_Signal'] = ta.trend.macd_signal(close_prices)
    df['MACD_Diff'] = ta.trend.macd_diff(close_prices)

    # 4. Bollinger Bands
    df['BBL'] = ta.volatility.bollinger_lband(close_prices)
    df['BBM'] = ta.volatility.bollinger_mavg(close_prices)
    df['BBH'] = ta.volatility.bollinger_hband(close_prices)
    df['BB_Width'] = ta.volatility.bollinger_wband(close_prices)

    # 5. Stochastic Oscillator
    df['STOCH_K'] = ta.momentum.stoch(high_prices, low_prices, close_prices, window=14, smooth_window=3)
    df['STOCH_D'] = ta.momentum.stoch_signal(high_prices, low_prices, close_prices, window=14, smooth_window=3)

    # 6. Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(high_prices, low_prices, close_prices, window=14)

    # 7. Volume-Based Indicators
    df['OBV'] = ta.volume.on_balance_volume(close_prices, volumes)

    # 8. Price-Based Features
    df['Daily_Return'] = close_prices.pct_change()
    df['Lag_Close_1'] = close_prices.shift(1)
    df['Lag_Return_1'] = df['Daily_Return'].shift(1)

    # 9. Date/Time Features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Drop any rows with NaN values that result from feature calculation
    original_rows = len(df)
    df = df.dropna()
    print(f"[DEBUG] Dropped {original_rows - len(df)} rows with NaN values after feature generation.")

    print("[DEBUG] Feature generation finished.")
    return df

if __name__ == "__main__":
    print("[DEBUG] Starting feature_engineering.py script")
    all_stocks_features = {}
    for ticker in sensex_tickers:
        print(f"Processing {ticker}...")
        raw_data = fetch_historical_data(ticker, period="5y", interval="1d") # Re-using fetch function
        if raw_data is not None:
            processed_data = generate_features(raw_data.copy()) # Use a copy to avoid SettingWithCopyWarning
            if not processed_data.empty:
                all_stocks_features[ticker] = processed_data
                print(f"[DEBUG] Features generated for {ticker}: {len(processed_data)} rows.")
            else:
                print(f"[DEBUG] No features generated for {ticker}.")
        else:
            print(f"[DEBUG] No raw data to generate features for {ticker}.")
    
    if all_stocks_features:
        print("\n--- Sample Features (first 5 rows) ---")
        for ticker, data in all_stocks_features.items():
            print(f"\n{ticker}:")
            print(data.head())
        
        # You can now save this data, or proceed to data preparation
        # Example: Save to CSV
        # for ticker, data in all_stocks_features.items():
        #     data.to_csv(f"{ticker}_features.csv")
        # print("\nFeature-engineered data saved to CSV files.")
    else:
        print("No features generated for any of the specified tickers.")
    print("[DEBUG] feature_engineering.py script finished")