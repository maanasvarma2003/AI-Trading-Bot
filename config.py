# config_template.py

# --- API Keys and Secrets ---
# Rename this file to `config.py` and fill in your actual API credentials.

# Zerodha Kite Connect API credentials
ZERODHA_API_KEY = "YOUR_ZERODHA_API_KEY"
ZERODHA_API_SECRET = "YOUR_ZERODHA_API_SECRET"
# For Zerodha, you will also need to generate a request token and then an access token
# Refer to Zerodha Kite Connect documentation for the authentication flow.
# Example: ACCESS_TOKEN = "YOUR_GENERATED_ACCESS_TOKEN"

# Example for other data providers (uncomment and fill as needed)
# ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
# POLYGON_IO_API_KEY = "YOUR_POLYGON_IO_API_KEY"

# --- Trading Parameters ---
INITIAL_CAPITAL = 100000 # Initial virtual capital for backtesting and simulated trading
TRANSACTION_COST_PER_TRADE = 0.001 # 0.1% transaction cost per trade
RISK_TOLERANCE = "aggressive" # Can be "aggressive", "moderate", "conservative"

# --- Data Parameters ---
HISTORICAL_DATA_PERIOD = "5y" # e.g., "1y", "5y", "max"
HISTORICAL_DATA_INTERVAL = "1d" # e.g., "1d", "60m", "1m"

# --- Model Training Parameters ---
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Real-time Trading Parameters ---
REALTIME_TRADE_FREQUENCY_SECONDS = 10 # How often to check for new data and place trades
REALTIME_TRADE_DURATION_MINUTES = 60 # How long to run the simulated live trading

# --- Monitoring and Retraining Parameters ---
METRICS_INTERVAL_SECONDS = 300 # How often to log performance and health metrics
RETRAINING_INTERVAL_DAYS = 7 # How often to check for model retraining
