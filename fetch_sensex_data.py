import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# List of some SENSEX blue-chip stock tickers (adjust as needed)
# You might need to verify the exact ticker symbols for yfinance for Indian stocks (e.g., .NS for NSE)
# Example: Reliance Industries -> RELIANCE.NS
sensex_tickers = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",       # Tata Consultancy Services
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS", # ICICI Bank
    "INFY.NS",      # Infosys
    "HINDUNILVR.NS" # Hindustan Unilever
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

if __name__ == "__main__":
    print("[DEBUG] Starting fetch_sensex_data.py script")
    all_stocks_data = {}
    for ticker in sensex_tickers:
        print(f"Fetching data for {ticker}...")
        data = fetch_historical_data(ticker, period="5y", interval="1d") # Fetch last 5 years daily data
        if data is not None:
            all_stocks_data[ticker] = data
    
    if all_stocks_data:
        print("\n--- Sample Data (first 5 rows) ---")
        for ticker, data in all_stocks_data.items():
            print(f"\n{ticker}:")
            print(data.head())
        
        # You can now save this data, or proceed to feature engineering/model training
        # Example: Save to CSV
        # for ticker, data in all_stocks_data.items():
        #     data.to_csv(f"{ticker}_historical_data.csv")
        # print("\nHistorical data saved to CSV files.")
    else:
        print("No data fetched for any of the specified tickers.")
    print("[DEBUG] fetch_sensex_data.py script finished")


# For real-time data, yfinance is not ideal. We would need a dedicated real-time API. 
# Options include Alpha Vantage, Polygon.io, or specific brokerage APIs (like Zerodha Kite Connect, if we can get it working).
