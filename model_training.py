import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV # Conceptual for hyperparameter tuning
import numpy as np # Added for np.random.rand
import pickle # For loading prepared data

# Assuming these would be loaded from data_preparation.py outputs
from backtesting_framework import BacktestingFramework # Import the backtesting framework
from data_preparation import sensex_tickers, PREPARED_DATA_FILENAME # Import necessary constants
# Removed imports for data_preparation functions as we'll load prepared data

# Removed load_dummy_prepared_data as we'll use actual prepared data

def train_and_evaluate_model(model, model_name, train_X, train_y, val_X, val_y, test_X=None, test_y=None):
    """Trains a given model and evaluates its performance on the validation set, and optionally on the test set."""
    print(f"\n--- Training {model_name} Model ---")
    # Ensure features for training are consistent with what was used for scaling
    # We drop Close, Volume, Target here because they are not features for the model.
    features_for_model = [col for col in train_X.columns if col not in ['Close', 'Volume', 'Target']]
    model.fit(train_X[features_for_model], train_y)
    
    # Make predictions on the validation set
    val_predictions = model.predict(val_X[features_for_model])
    val_probabilities = model.predict_proba(val_X[features_for_model])[:, 1] # Probability of positive class
    
    # Evaluate performance on validation set
    accuracy = accuracy_score(val_y, val_predictions)
    precision = precision_score(val_y, val_predictions)
    recall = recall_score(val_y, val_predictions)
    f1 = f1_score(val_y, val_predictions)
    roc_auc = roc_auc_score(val_y, val_probabilities)
    
    print(f"{model_name} Validation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    if test_X is not None and test_y is not None:
        test_predictions = model.predict(test_X[features_for_model])
        test_probabilities = model.predict_proba(test_X[features_for_model])[:, 1]
        
        test_accuracy = accuracy_score(test_y, test_predictions)
        test_precision = precision_score(test_y, test_predictions)
        test_recall = recall_score(test_y, test_predictions)
        test_f1 = f1_score(test_y, test_predictions)
        test_roc_auc = roc_auc_score(test_y, test_probabilities)
        
        print(f"{model_name} Test Metrics (Final Evaluation):")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  ROC AUC: {test_roc_auc:.4f}")
    
    return model

# Dummy DRL Agent Policy for backtesting
class DummyDRLAgentPolicy:
    def choose_action(self, state):
        # Random action for conceptual DRL agent during backtesting
        return np.random.choice([0, 1, 2]) # 0: Sell, 1: Hold, 2: Buy


if __name__ == "__main__":
    print("[DEBUG] Starting model_training.py script")
    
    # Load all prepared data once
    try:
        with open(PREPARED_DATA_FILENAME, 'rb') as f:
            all_prepared_data = pickle.load(f)
        print(f"[DEBUG] Loaded prepared data from {PREPARED_DATA_FILENAME}.")
    except FileNotFoundError:
        print(f"[ERROR] Prepared data file {PREPARED_DATA_FILENAME} not found. Please run data_preparation.py first.")
        exit()

    all_trained_models = {}

    for ticker in sensex_tickers:
        print(f"\nProcessing data for {ticker} for model training...")
        if ticker in all_prepared_data:
            data_for_ticker = all_prepared_data[ticker]
            train_X_scaled = data_for_ticker['train_X']
            train_y = data_for_ticker['train_y']
            val_X_scaled = data_for_ticker['val_X']
            val_y = data_for_ticker['val_y']
            test_X_scaled = data_for_ticker['test_X']
            test_y = data_for_ticker['test_y']

            print(f"[DEBUG] Before training models for {ticker}:")
            print(f"  train_X_scaled shape: {train_X_scaled.shape}, columns: {train_X_scaled.columns.tolist()}")
            print(f"  val_X_scaled shape: {val_X_scaled.shape}, columns: {val_X_scaled.columns.tolist()}")
            print(f"  test_X_scaled shape: {test_X_scaled.shape}, columns: {test_X_scaled.columns.tolist()}")

            # --- Random Forest Classifier ---
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            trained_rf_model = train_and_evaluate_model(rf_model, "Random Forest", train_X_scaled, train_y, val_X_scaled, val_y, test_X_scaled, test_y)
            all_trained_models[f'{ticker}_rf'] = trained_rf_model

            # --- XGBoost Classifier ---
            xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            trained_xgb_model = train_and_evaluate_model(xgb_model, "XGBoost", train_X_scaled, train_y, val_X_scaled, val_y, test_X_scaled, test_y)
            all_trained_models[f'{ticker}_xgb'] = trained_xgb_model

            # --- DRL Agent (Conceptual) ---
            drl_agent_policy = DummyDRLAgentPolicy() # Using a dummy policy for conceptual backtesting
            all_trained_models[f'{ticker}_drl_policy'] = drl_agent_policy

            print(f"[DEBUG] Models trained and evaluated for {ticker}.")
        else:
            print(f"[DEBUG] No prepared data found for {ticker}. Skipping model training.")

    print("\n--- Evaluating Models with Backtesting Framework ---")
    if all_trained_models:
        for ticker in sensex_tickers:
            if ticker in all_prepared_data:
                print(f"\nRunning backtests for {ticker}...")
                data_for_backtest = all_prepared_data[ticker]
                test_X_for_backtest = data_for_backtest['test_X']
                test_y_for_backtest = data_for_backtest['test_y'] # Not directly used by backtester, but good to keep consistent

                # Recreate test_df with original Close prices and correct index for backtesting
                # This requires getting the original (unscaled) Close prices for the test set
                # We need to fetch raw_data and feature_engineer it up to the point of creating df_with_target for the test set portion
                # This is slightly complex because the scaler only applies to features, not Close/Volume.
                # A simpler approach for conceptual backtesting is to assume test_X_scaled also contains original Close and Volume.
                # However, for correct backtesting, we need the actual prices.
                # Let's assume, for now, that test_X_scaled contains an 'Original_Close' column or similar for backtesting purposes.
                # For this conceptual implementation, we will use the 'Close' column from the test_X_scaled for backtesting prices.

                # Create a DataFrame for backtesting that includes the 'Close' column and is properly indexed
                test_df_for_backtest = test_X_for_backtest.copy()
                test_df_for_backtest['Close'] = test_X_for_backtest['Close'] # Assuming 'Close' is still there, even if scaled
                test_df_for_backtest['Target'] = test_y_for_backtest # Add target for consistency, though backtester doesn't use it directly
                test_df_for_backtest.index = test_y_for_backtest.index # Ensure index is correct datetime index

                # Backtest Random Forest
                rf_model = all_trained_models[f'{ticker}_rf']
                backtester_rf = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=100000)
                rf_portfolio_history = backtester_rf.run_backtest(rf_model, strategy_name=f"{ticker} Random Forest Strategy")
                rf_metrics = backtester_rf.calculate_metrics(rf_portfolio_history)
                print(f"\n{ticker} Random Forest Backtest Metrics:")
                for metric, value in rf_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")

                # Backtest XGBoost
                xgb_model = all_trained_models[f'{ticker}_xgb']
                backtester_xgb = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=100000)
                xgb_portfolio_history = backtester_xgb.run_backtest(xgb_model, strategy_name=f"{ticker} XGBoost Strategy")
                xgb_metrics = backtester_xgb.calculate_metrics(xgb_portfolio_history)
                print(f"\n{ticker} XGBoost Backtest Metrics:")
                for metric, value in xgb_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")

                # Backtest DRL Agent (Conceptual)
                drl_agent_policy = all_trained_models[f'{ticker}_drl_policy']
                backtester_drl = BacktestingFramework(test_df_for_backtest.drop(['Target'], axis=1), initial_capital=100000)
                drl_portfolio_history = backtester_drl.run_backtest(drl_agent_policy, strategy_name=f"{ticker} DRL Agent Strategy")
                drl_metrics = backtester_drl.calculate_metrics(drl_portfolio_history)
                print(f"\n{ticker} DRL Agent Backtest Metrics (Conceptual):")
                for metric, value in drl_metrics.items():
                    if isinstance(value, (int, float)): print(f"  {metric}: {value:.4f}")
                    else: print(f"  {metric}: {value}")
            else:
                print(f"[DEBUG] No prepared data found for {ticker} for backtesting. Skipping backtesting.")
    else:
        print("No models trained or prepared data found for backtesting.")

    print("[DEBUG] model_training.py script finished")
