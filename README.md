# Fully Automated SENSEX Trading Bot

This project aims to develop a fully automated trading bot that uses machine learning (Random Forest, XGBoost, Deep Reinforcement Learning) to execute buy/sell orders on SENSEX stocks, improving trade accuracy and speed while reducing human error. The bot is designed with an **aggressive risk tolerance**.

## Project Structure

-   `requirements.txt`: Lists all Python dependencies.
-   `config_template.py`: A template for securely storing API keys and sensitive configurations.
-   `fetch_sensex_data.py`: Handles fetching historical stock data using `yfinance` (can be extended for real-time APIs).
-   `feature_engineering.py`: Contains functions to generate technical indicators and other features from raw stock data.
-   `data_preparation.py`: Prepares the feature-engineered data for model training (target variable creation, chronological splitting, scaling).
-   `model_training.py`: Implements training, hyperparameter tuning, and evaluation of Random Forest and XGBoost models.
-   `drl_trading_bot.py`: Conceptual design for a Deep Reinforcement Learning environment and agent.
-   `backtesting_framework.py`: Provides a framework for simulating trading strategies on historical data and evaluating performance.
-   `realtime_trading_bot.py`: Outlines the real-time execution logic, including API integration, prediction, and order management.
-   `monitoring_and_retraining.py`: Sets up conceptual monitoring for bot performance and an automated retraining pipeline.
-   `main.py`: The orchestrator script to run different parts of the trading bot.
-   `all_prepared_data.pkl`: (Generated after running `data_preparation.py`) Stores the processed historical data to speed up model training.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd stocks
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**

    *   Rename `config_template.py` to `config.py`.
    *   Open `config.py` and replace placeholder values with your actual API keys and secrets.

        ```python
        # config.py
        ZERODHA_API_KEY = "YOUR_ZERODHA_API_KEY"
        ZERODHA_API_SECRET = "YOUR_ZERODHA_API_SECRET"
        # ... other API keys as needed ...
        ```

    *   **Zerodha Kite Connect (Recommended for Indian Markets):**
        *   You'll need a developer account with Zerodha to get API keys. Refer to the [official Kite Connect documentation](https://kite.trade/docs/connect/v3/) for detailed setup and authentication instructions.
        *   The `kiteconnect` library will handle API interactions.

    *   **Alternative Data Providers (e.g., Alpha Vantage, Polygon.io):**
        *   For historical and real-time data beyond `yfinance` (which is often sufficient for historical but less reliable for real-time), consider dedicated data providers. You'll need to sign up for their services and integrate their Python clients if available.

## Usage

Run the `main.py` script to orchestrate different parts of the bot.

```bash
python main.py
```

`main.py` will allow you to:

1.  **Run Data Pipeline:** Fetch raw data, engineer features, and prepare data for model training.
2.  **Train Models:** Train Random Forest, XGBoost, and conceptually define DRL agents.
3.  **Backtest Strategies:** Simulate trading using historical data and evaluate performance.
4.  **Simulate Live Trading:** Run the bot in a simulated real-time environment.
5.  **Monitor and Retrain (Conceptual):** Outline how monitoring and retraining would work.

## Project Stages (Conceptual Implementation Details)

### 1. Data Acquisition

-   **Tool:** `yfinance` is used for historical data. For real-time, consider integrating specific brokerage APIs (e.g., Zerodha Kite Connect, Upstox) or data providers.
-   **Process:** `fetch_sensex_data.py` will fetch historical data for specified SENSEX tickers.

### 2. Feature Engineering

-   **Tool:** `ta` (Technical Analysis library) is used.
-   **Process:** `feature_engineering.py` generates a rich set of technical indicators (Moving Averages, RSI, MACD, Bollinger Bands, Stochastic Oscillator, ATR, OBV), price-based features (returns, lagged prices), and date/time features.

### 3. Data Preparation

-   **Tool:** `pandas`, `scikit-learn` (`MinMaxScaler`).
-   **Process:** `data_preparation.py` creates a binary target variable (e.g., predicting next day's price movement), performs a chronological split into training, validation, and test sets, and scales numerical features. The prepared data is saved to `all_prepared_data.pkl`.

### 4. Model Selection and Training

-   **Tools:** `scikit-learn` (RandomForestClassifier), `xgboost` (XGBClassifier), and conceptually, `stable-baselines3` or `tensorflow/pytorch` for DRL.
-   **Process:** `model_training.py` trains these models on the prepared data, performs hyperparameter tuning (conceptually outlined), and evaluates them using metrics like accuracy, precision, recall, F1-score, and ROC AUC.

### 5. Backtesting

-   **Tool:** `backtesting_framework.py` (custom implementation).
-   **Process:** Simulates trading on historical test data using the trained models, accounting for transaction costs. Evaluates strategies using financial metrics such as Cumulative Returns, Annualized Returns, Annualized Volatility, Sharpe Ratio, and Maximum Drawdown.

### 6. Real-time Execution

-   **Tool:** `realtime_trading_bot.py` (conceptual, requires live API integration).
-   **Process:** Integrates with real-time data feeds, generates features on-the-fly, uses the best-performing model to generate trading signals, and places orders via a brokerage API. Implements aggressive risk management (e.g., position sizing, stop-losses).

### 7. Monitoring and Retraining

-   **Tool:** `monitoring_and_retraining.py` (conceptual, requires logging, alerting, and MLOps tools).
-   **Process:** Sets up continuous monitoring for bot performance, system health, and model drift. Implements an automated retraining pipeline to periodically re-train models with new data and deploy improved versions.

## Further Development

-   **Live API Integration:** Replace placeholder API clients with actual `kiteconnect` or other brokerage APIs.
-   **Sophisticated Risk Management:** Implement more advanced risk models (e.g., VaR, CVaR).
-   **DRL Implementation:** Fully implement the DRL environment and agent using libraries like `stable-baselines3`.
-   **Cloud Deployment:** Deploy the bot on a cloud platform (AWS, GCP, Azure) for robust 24/7 operation.
-   **Dashboarding and Alerting:** Integrate with tools like Grafana, Prometheus, or custom dashboards for real-time visualization and alerts.
-   **Portfolio Management:** Extend to handle multiple assets and portfolio optimization.


