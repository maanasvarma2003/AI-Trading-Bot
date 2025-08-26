import pandas as pd
# from sklearn.model_selection import train_test_split # No longer needed for chronological split
from sklearn.preprocessing import MinMaxScaler
import numpy as np # Added for potential numpy operations if needed
import pickle # For saving/loading prepared data
import os # For checking file existence

# Import the feature_engineering script to get our processed data
from feature_engineering import sensex_tickers, fetch_historical_data, generate_features # Re-import original sensex_tickers list

# Define a filename for saving/loading prepared data
PREPARED_DATA_FILENAME = "all_prepared_data.pkl"

def create_target_variable(df, price_col='Close', num_days=1):
    """Creates a binary target variable: 1 if price goes up, 0 otherwise."""
    print(f"[DEBUG] Creating target variable...")
    # Ensure price_col exists before creating target
    if price_col not in df.columns:
        print(f"[ERROR] Price column '{price_col}' not found in DataFrame for target variable creation.")
        return df.copy() # Return copy to avoid modifying original df if error

    df['Target'] = (df[price_col].shift(-num_days) > df[price_col]).astype(int)
    original_rows = len(df)
    df = df.dropna() # Drop the last `num_days` rows where target is NaN
    print(f"[DEBUG] Dropped {original_rows - len(df)} rows after creating target variable.")
    return df

def split_time_series_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits time series data chronologically into training, validation, and test sets."""
    print("[DEBUG] Splitting data chronologically...")
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    print(f"[DEBUG] Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    return train_df, val_df, test_df

def scale_features(train_df, val_df, test_df, features_to_scale):
    """Scales numerical features using MinMaxScaler based on the training set."""
    print("[DEBUG] Scaling features...")
    scaler = MinMaxScaler()
    
    # Ensure features_to_scale are actually in the DataFrames before scaling
    train_cols_exist = [col for col in features_to_scale if col in train_df.columns]
    val_cols_exist = [col for col in features_to_scale if col in val_df.columns]
    test_cols_exist = [col for col in features_to_scale if col in test_df.columns]

    if not train_cols_exist or not val_cols_exist or not test_cols_exist:
        print("[ERROR] One or more features to scale not found in DataFrames. Skipping scaling for missing columns.")
        # Proceed with scaling only existing columns
        train_df[train_cols_exist] = scaler.fit_transform(train_df[train_cols_exist])
        val_df[val_cols_exist] = scaler.transform(val_df[val_cols_exist])
        test_df[test_cols_exist] = scaler.transform(test_df[test_cols_exist])
    else:
        train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
        val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
        test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    
    print("[DEBUG] Feature scaling finished.")
    return train_df, val_df, test_df, scaler

if __name__ == "__main__":
    print("[DEBUG] Starting data_preparation.py script")

    if os.path.exists(PREPARED_DATA_FILENAME):
        print(f"[DEBUG] Loading prepared data from {PREPARED_DATA_FILENAME}...")
        with open(PREPARED_DATA_FILENAME, 'rb') as f:
            all_prepared_data = pickle.load(f)
        print("[DEBUG] Prepared data loaded successfully.")
    else:
        print("[DEBUG] Prepared data file not found. Generating data...")
        all_prepared_data = {}
        for ticker in sensex_tickers:
            print(f"\nProcessing data for {ticker} for preparation...")
            raw_data = fetch_historical_data(ticker, period="5y", interval="1d") # Reverted to 5y period for full data
            if raw_data is not None and not raw_data.empty:
                # Flatten MultiIndex columns if they exist (from yfinance)
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.columns = ['_'.join(col).strip() for col in raw_data.columns.values] # Flatten
                    # Rename 'Close_' to 'Close' etc. if they become 'Close_RELIANCE.NS'
                    raw_data = raw_data.rename(columns=lambda x: x.replace(f"_{ticker}", "").strip('_'))
                    print(f"[DEBUG] Flattened MultiIndex columns for {ticker}.")
                
                # Ensure 'Close' and 'Volume' columns are present and properly named after flattening
                if 'Close' not in raw_data.columns and 'close' in raw_data.columns: raw_data = raw_data.rename(columns={'close': 'Close'})
                if 'Volume' not in raw_data.columns and 'volume' in raw_data.columns: raw_data = raw_data.rename(columns={'volume': 'Volume'})
                
                feature_engineered_df = generate_features(raw_data.copy())
                
                if not feature_engineered_df.empty:
                    # 2. Create Target Variable
                    df_with_target = create_target_variable(feature_engineered_df.copy(), num_days=1)
                    
                    # Define features and target (now with flattened column names)
                    features = [col for col in df_with_target.columns if col not in ['Close', 'Volume', 'Target']]
                    target = 'Target'
                    
                    X = df_with_target[features]
                    y = df_with_target[target]
                    
                    # 3. Split Data Chronologically
                    train_X, val_X, test_X = split_time_series_data(X, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
                    train_y, val_y, test_y = split_time_series_data(y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
                    
                    # Also split the original Close prices for backtesting before scaling
                    original_close_prices_test = split_time_series_data(df_with_target['Close'], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)[2] # [2] to get test set

                    # 4. Scale Features
                    # Identify numerical features to scale (all features are numerical here, except date-related)
                    features_to_scale = [f for f in features if f not in ['Day_of_Week', 'Month', 'Year']]
                    train_X_scaled, val_X_scaled, test_X_scaled, scaler = scale_features(train_X.copy(), val_X.copy(), test_X.copy(), features_to_scale)
                    
                    all_prepared_data[ticker] = {
                        'train_X': train_X_scaled,
                        'train_y': train_y,
                        'val_X': val_X_scaled,
                        'val_y': val_y,
                        'test_X': test_X_scaled,
                        'test_y': test_y,
                        'scaler': scaler, # Keep the scaler for inverse transform if needed later
                        'original_test_close': original_close_prices_test # Store original Close prices for backtesting
                    }
                    print(f"[DEBUG] Data prepared for {ticker}.")
                else:
                    print(f"[DEBUG] No feature-engineered data for {ticker} to prepare.")
            else:
                print(f"[DEBUG] No raw data for {ticker} to generate features for preparation.")

        if all_prepared_data:
            # Save the prepared data
            with open(PREPARED_DATA_FILENAME, 'wb') as f:
                pickle.dump(all_prepared_data, f)
            print(f"[DEBUG] All prepared data saved to {PREPARED_DATA_FILENAME}.")
        else:
            print("No data prepared for any of the specified tickers.")

    if all_prepared_data:
        print("\n--- Sample Prepared Data (first 5 rows of scaled train_X) ---")
        for ticker, data in all_prepared_data.items():
            print(f"\n{ticker} Train X Scaled:")
            print(data['train_X'].head())
        
        print("\nData preparation complete for all tickers. Ready for model training.")
    else:
        print("No data prepared for any of the specified tickers.")
    print("[DEBUG] data_preparation.py script finished")
