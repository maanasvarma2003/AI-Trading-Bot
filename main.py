import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import all conceptual modules
from fetch_sensex_data import sensex_tickers, fetch_historical_data # Re-import original sensex_tickers
from feature_engineering import generate_features
from data_preparation import create_target_variable, split_time_series_data, scale_features, PREPARED_DATA_FILENAME
from model_training import train_and_evaluate_model, DummyDRLAgentPolicy
from backtesting_framework import BacktestingFramework
from realtime_trading_bot import RealtimeDataAPIClient, BrokerageAPIClient, RealtimeTradingBot, DummyTrainedModel
from monitoring_and_retraining import MonitoringSystem, RetrainingPipeline, DummyDataSourceConnector, DummyModelTrainer, DummyModelEvaluator, DummyDeploymentManager

# Load configuration from config.py (rename config_template.py to config.py first)
try:
    import config
except ImportError:
    print("Error: config.py not found. Please rename config_template.py to config.py and fill in your API keys.")
    exit()

def run_data_pipeline(tickers, period, interval):
    """Runs the data acquisition, feature engineering, and data preparation pipeline."""
    print("\n--- Running Data Pipeline ---")
    all_prepared_data = {}

    if os.path.exists(PREPARED_DATA_FILENAME):
        print(f"Loading prepared data from {PREPARED_DATA_FILENAME}...")
        try:
            with open(PREPARED_DATA_FILENAME, 'rb') as f:
                all_prepared_data = pickle.load(f)
            # Check if all expected tickers are in the loaded data and have the 'original_test_close' key
            if all(ticker in all_prepared_data and 'original_test_close' in all_prepared_data[ticker] for ticker in tickers):
                print("Prepared data loaded successfully with all required keys.")
            else:
                print("Prepared data is incomplete or outdated. Please run `python data_preparation.py` to regenerate.")
                all_prepared_data = {} # Clear incomplete data
        except Exception as e:
            print(f"Error loading prepared data: {e}. Please run `python data_preparation.py` to regenerate.")
            all_prepared_data = {}
    else:
        print("Prepared data file not found. Please run `python data_preparation.py` to generate data.")

    if not all_prepared_data:
        print("Data Pipeline finished (no data loaded/generated).")
    return all_prepared_data

def process_single_ticker_data(ticker, period, interval):
    """Helper function to process data for a single ticker."""
    raw_data = fetch_historical_data(ticker, period=period, interval=interval)
    if raw_data is not None and not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = ['_'.join(col).strip() for col in raw_data.columns.values]
            raw_data = raw_data.rename(columns=lambda x: x.replace(f"_{ticker}", "").strip('_'))
        if 'Close' not in raw_data.columns and 'close' in raw_data.columns: raw_data = raw_data.rename(columns={'close': 'Close'})
        if 'Volume' not in raw_data.columns and 'volume' in raw_data.columns: raw_data = raw_data.rename(columns={'volume': 'Volume'})

        feature_engineered_df = generate_features(raw_data.copy())
        if not feature_engineered_df.empty:
            df_with_target = create_target_variable(feature_engineered_df.copy(), num_days=1)
            features = [col for col in df_with_target.columns if col not in ['Close', 'Volume', 'Target']]
            target = 'Target'
            X = df_with_target[features]
            y = df_with_target[target]
            train_X, val_X, test_X = split_time_series_data(X, train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO)
            train_y, val_y, test_y = split_time_series_data(y, train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO)
            features_to_scale = [f for f in features if f not in ['Day_of_Week', 'Month', 'Year']]
            train_X_scaled, val_X_scaled, test_X_scaled, scaler = scale_features(train_X.copy(), val_X.copy(), test_X.copy(), features_to_scale)
            return {
                'train_X': train_X_scaled, 'train_y': train_y,
                'val_X': val_X_scaled, 'val_y': val_y,
                'test_X': test_X_scaled, 'test_y': test_y,
                'scaler': scaler,
                'original_test_close': raw_data['Close'].iloc[-len(test_X_scaled):].values # Store original Close for backtesting
            }
    return None

def run_model_training(prepared_data):
    """Trains and evaluates ML models using the prepared data."""
    print("\n--- Running Model Training and Evaluation ---")
    all_trained_models = {}

    if not prepared_data:
        print("No prepared data available for model training. Please run data pipeline first.")
        return {}

    for ticker in sensex_tickers:
        print(f"\nTraining models for {ticker}...")
        if ticker in prepared_data:
            data_for_ticker = prepared_data[ticker]
            train_X_scaled = data_for_ticker['train_X']
            train_y = data_for_ticker['train_y']
            val_X_scaled = data_for_ticker['val_X']
            val_y = data_for_ticker['val_y']
            test_X_scaled = data_for_ticker['test_X']
            test_y = data_for_ticker['test_y']

            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            trained_rf_model = train_and_evaluate_model(rf_model, "Random Forest", train_X_scaled, train_y, val_X_scaled, val_y, test_X_scaled, test_y)
            all_trained_models[f'{ticker}_rf'] = trained_rf_model

            # XGBoost
            xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            trained_xgb_model = train_and_evaluate_model(xgb_model, "XGBoost", train_X_scaled, train_y, val_X_scaled, val_y, test_X_scaled, test_y)
            all_trained_models[f'{ticker}_xgb'] = trained_xgb_model

            # DRL Agent (Conceptual)
            drl_agent_policy = DummyDRLAgentPolicy() # In a real scenario, this would be a trained DRL policy
            all_trained_models[f'{ticker}_drl_policy'] = drl_agent_policy
            print(f"Models trained and evaluated for {ticker}.")
        else:
            print(f"No prepared data for {ticker}. Skipping model training.")
    print("Model Training and Evaluation finished.")
    return all_trained_models

def run_backtesting(prepared_data, trained_models):
    """Runs backtesting for all trained models."""
    print("\n--- Running Backtesting ---")
    if not prepared_data or not trained_models:
        print("Prepared data or trained models not available for backtesting. Please run data pipeline and model training first.")
        return

    for ticker in sensex_tickers:
        if ticker in prepared_data:
            print(f"\nRunning backtests for {ticker}...")
            data_for_backtest = prepared_data[ticker]
            test_X_for_backtest = data_for_backtest['test_X']
            test_y_for_backtest = data_for_backtest['test_y']
            original_test_close = data_for_backtest['original_test_close']

            # Create a DataFrame for backtesting that includes the 'Close' column and is properly indexed
            # Ensure indices align correctly when combining
            test_df_for_backtest = pd.DataFrame(test_X_for_backtest.copy())
            test_df_for_backtest['Close'] = original_test_close.reindex(test_df_for_backtest.index) # Ensure index alignment
            test_df_for_backtest['Target'] = test_y_for_backtest.reindex(test_df_for_backtest.index) # Ensure index alignment
            
            # Drop any potential NaN values that might arise from reindexing if there were mismatches
            test_df_for_backtest = test_df_for_backtest.dropna()

            # Backtest Random Forest
            rf_model = trained_models.get(f'{ticker}_rf')
            if rf_model:
                backtester_rf = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=config.INITIAL_CAPITAL, transaction_cost_per_trade=config.TRANSACTION_COST_PER_TRADE)
                rf_portfolio_history = backtester_rf.run_backtest(rf_model, strategy_name=f"{ticker} Random Forest Strategy")
                rf_metrics = backtester_rf.calculate_metrics(rf_portfolio_history)
                print(f"\n{ticker} Random Forest Backtest Metrics:")
                for metric, value in rf_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")

            # Backtest XGBoost
            xgb_model = trained_models.get(f'{ticker}_xgb')
            if xgb_model:
                backtester_xgb = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=config.INITIAL_CAPITAL, transaction_cost_per_trade=config.TRANSACTION_COST_PER_TRADE)
                xgb_portfolio_history = backtester_xgb.run_backtest(xgb_model, strategy_name=f"{ticker} XGBoost Strategy")
                xgb_metrics = backtester_xgb.calculate_metrics(xgb_portfolio_history)
                print(f"\n{ticker} XGBoost Backtest Metrics:")
                for metric, value in xgb_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")

            # Backtest DRL Agent (Conceptual)
            drl_agent_policy = trained_models.get(f'{ticker}_drl_policy')
            if drl_agent_policy:
                backtester_drl = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=config.INITIAL_CAPITAL, transaction_cost_per_trade=config.TRANSACTION_COST_PER_TRADE)
                drl_portfolio_history = backtester_drl.run_backtest(drl_agent_policy, strategy_name=f"{ticker} DRL Agent Strategy")
                drl_metrics = backtester_drl.calculate_metrics(drl_portfolio_history)
                print(f"\n{ticker} DRL Agent Backtest Metrics (Conceptual):")
                for metric, value in drl_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")
        else:
            print(f"No prepared data for {ticker}. Skipping backtesting.")
    print("Backtesting finished.")

def run_simulated_live_trading(trained_models):
    """Runs a simulated live trading session."""
    print("\n--- Running Simulated Live Trading ---")
    if not trained_models:
        print("No trained models available for simulated live trading. Please run model training first.")
        return

    # Initialize conceptual API clients with config values
    data_api_client = RealtimeDataAPIClient(api_key=config.ZERODHA_API_KEY, api_secret=config.ZERODHA_API_SECRET) # Using Zerodha as example
    brokerage_api_client = BrokerageAPIClient(api_key=config.ZERODHA_API_KEY, api_secret=config.ZERODHA_API_SECRET)

    # For simulation, pick one of the trained models (e.g., Random Forest for RELIANCE.NS)
    # In a real scenario, you'd choose the best model for each ticker or a composite strategy
    simulated_model_key = f'{sensex_tickers[0]}_rf' if sensex_tickers else None
    if simulated_model_key and simulated_model_key in trained_models:
        best_model_for_live = trained_models[simulated_model_key]
        bot = RealtimeTradingBot(
            data_api_client, brokerage_api_client, best_model_for_live,
            initial_capital=config.INITIAL_CAPITAL, risk_tolerance=config.RISK_TOLERANCE
        )
        bot.run_live_trading(sensex_tickers, frequency_seconds=config.REALTIME_TRADE_FREQUENCY_SECONDS, duration_minutes=config.REALTIME_TRADE_DURATION_MINUTES)
    else:
        print("No suitable trained model found for simulated live trading.")
    print("Simulated Live Trading finished.")

def run_monitoring_and_retraining():
    """Conceptually runs monitoring and retraining cycles."""
    print("\n--- Running Conceptual Monitoring and Retraining ---")
    # For this conceptual demonstration, we need a dummy bot instance
    class DummyRealtimeTradingBotForMonitor:
        def __init__(self):
            self.portfolio_value = config.INITIAL_CAPITAL + np.random.uniform(-5000, 5000)
            self.capital = config.INITIAL_CAPITAL * 0.8 + np.random.uniform(-2000, 2000)
            self.shares_held = {"RELIANCE.NS": 10, "TCS.NS": 5} # Example
            self.data_api = True # Simulate API availability
            self.brokerage_api = True # Simulate API availability
            self.model = "CurrentBestModel" # Placeholder for the active model

    dummy_bot_instance_for_monitor = DummyRealtimeTradingBotForMonitor()

    monitor = MonitoringSystem(dummy_bot_instance_for_monitor, metrics_interval_seconds=config.METRICS_INTERVAL_SECONDS)
    retrainer = RetrainingPipeline(
        DummyDataSourceConnector(), DummyModelTrainer(), DummyModelEvaluator(), DummyDeploymentManager(),
        retraining_interval_days=config.RETRAINING_INTERVAL_DAYS
    )

    print("Starting conceptual continuous monitoring and retraining simulation. (Press Ctrl+C to stop in a real scenario)")
    
    # Simulate a few cycles for conceptual demo
    for i in range(3):
        print(f"\n--- Monitoring/Retraining Cycle {i+1} ---")
        monitor.run_monitoring_cycle()
        retrainer.run_retraining_cycle()
        # time.sleep(1) # In a real scenario, this would be a longer sleep
    print("Conceptual Monitoring and Retraining finished.")

def main():
    parser = argparse.ArgumentParser(description="Fully Automated SENSEX Trading Bot")
    parser.add_argument('--stage', type=str, default='all', help="Which stage to run: data_pipeline, train_models, backtest, live_trade, monitor_retrain, or all")
    args = parser.parse_args()

    prepared_data = {}
    trained_models = {}

    if args.stage == 'data_pipeline' or args.stage == 'all':
        prepared_data = run_data_pipeline(sensex_tickers, config.HISTORICAL_DATA_PERIOD, config.HISTORICAL_DATA_INTERVAL)

    if args.stage == 'train_models' or args.stage == 'all':
        if not prepared_data and os.path.exists(PREPARED_DATA_FILENAME):
            print(f"Loading prepared data from {PREPARED_DATA_FILENAME} for model training...")
            with open(PREPARED_DATA_FILENAME, 'rb') as f:
                prepared_data = pickle.load(f)
            print("Prepared data loaded successfully.")
        trained_models = run_model_training(prepared_data)

    if args.stage == 'backtest' or args.stage == 'all':
        if not prepared_data and os.path.exists(PREPARED_DATA_FILENAME):
            print(f"Loading prepared data from {PREPARED_DATA_FILENAME} for backtesting...")
            with open(PREPARED_DATA_FILENAME, 'rb') as f:
                prepared_data = pickle.load(f)
            print("Prepared data loaded successfully.")
        if not trained_models:
            print("Loading trained models for backtesting (assuming they were saved and can be loaded)...")
            # In a real system, you would load your saved models here
            # For conceptual demo, we'll re-run training if not already done
            if prepared_data: # Only retrain if data is available
                trained_models = run_model_training(prepared_data)

        run_backtesting(prepared_data, trained_models)

    if args.stage == 'live_trade' or args.stage == 'all':
        if not trained_models:
            print("Loading trained models for simulated live trading (assuming they were saved and can be loaded)...")
            # For conceptual demo, we'll re-run training if not already done
            if prepared_data: # Only retrain if data is available
                trained_models = run_model_training(prepared_data)
            else: # If no prepared data, create dummy for quick demo
                print("No prepared data, creating dummy for live trade simulation.")
                dummy_data = {t: {'test_X': pd.DataFrame(np.random.rand(100, 25), columns=[f'feature_{i}' for i in range(25)], index=pd.date_range(start='2022-01-01', periods=100, freq='D'))} for t in sensex_tickers}
                # Add a 'Close' column to dummy test_X for backtesting
                for t in sensex_tickers:
                    dummy_data[t]['test_X']['Close'] = np.random.rand(100) * 100 + 100
                trained_models = {f'{t}_rf': DummyTrainedModel() for t in sensex_tickers} # Dummy models
                prepared_data = dummy_data # Assign dummy data to prepared_data for consistency

        run_simulated_live_trading(trained_models)

    if args.stage == 'monitor_retrain' or args.stage == 'all':
        run_monitoring_and_retraining()

    print("\nProject execution finished.")

if __name__ == "__main__":
    main()
