import numpy as np
import pandas as pd
from pycaret.regression import *
import os
from datetime import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(os.path.join(MODEL_DIR, 'models')):
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)

def preprocess_data(df):
    """Convert raw transaction data to yearly spending features with robust cleaning"""
    
    # Drop rows where essential columns are missing
    df = df.dropna(subset=['Date Submitted', 'Status', 'Price(RM)'])
    
    # Convert to datetime safely, coerce errors to NaT and drop them
    df['Date Submitted'] = pd.to_datetime(df['Date Submitted'], errors='coerce')
    df = df.dropna(subset=['Date Submitted'])
    
    # Ensure Price(RM) is numeric, coerce errors to NaN and drop them
    df['Price(RM)'] = pd.to_numeric(df['Price(RM)'], errors='coerce')
    df = df.dropna(subset=['Price(RM)'])
    
    # Filter only collected rewards
    df = df[df['Status'] == 'Reward Collected']
    
    # Extract year and calculate yearly totals
    df['Year'] = df['Date Submitted'].dt.year
    yearly_data = df.groupby('Year').agg({
        'Price(RM)': ['sum', 'count', 'mean'],
    })
    yearly_data.columns = ['TotalSpent', 'TransactionCount', 'AverageSpent']
    yearly_data = yearly_data.reset_index()
    
    # Calculate year-over-year growth rates
    yearly_data['YoY_Growth'] = yearly_data['TotalSpent'].pct_change()
    
    return yearly_data

def train_and_predict(df):
    # Preprocess the raw data
    processed_df = preprocess_data(df)
    
    # Ensure we have enough historical data (at least 3 years)
    if len(processed_df) < 3:
        raise ValueError("Insufficient historical data (need at least 3 years)")
    
    # Setup PyCaret experiment
    reg = setup(
        data=processed_df,
        target='TotalSpent',
        train_size=0.8 if len(processed_df) > 5 else 0.7,
        session_id=42,
        verbose=False,
        numeric_features=['TransactionCount', 'AverageSpent', 'YoY_Growth'],
        fold=min(5, len(processed_df)-1),
        remove_outliers=True,
        normalize=True
    )


    
    # Compare models and select the best one based on MAE
    best_model = compare_models(sort='MAE', n_select=1, exclude=['catboost'])  # Exclude slow models
    
    # Tune the best model
    tuned_model = tune_model(best_model, optimize='MAE')
    
    # Finalize the model (train on entire dataset)
    final_model = finalize_model(tuned_model)
    
    # Generate future predictions (next 4 years)
    last_year = processed_df['Year'].max()
    future_years = list(range(last_year + 1, last_year + 5))
    
    # Create future data with conservative growth assumptions
    last_trans_count = processed_df['TransactionCount'].iloc[-1]
    last_avg_spent = processed_df['AverageSpent'].iloc[-1]
    last_yoy = processed_df['YoY_Growth'].iloc[-1] if not np.isnan(processed_df['YoY_Growth'].iloc[-1]) else 0.05
    
    future_data = pd.DataFrame({
        'Year': future_years,
        'TransactionCount': np.round(np.linspace(
            last_trans_count,
            last_trans_count * (1 + min(last_yoy, 0.1)),  # Cap growth at 10%
            4
        )),
        'AverageSpent': np.linspace(
            last_avg_spent,
            last_avg_spent * (1 + min(last_yoy/2, 0.05)),  # More conservative growth for avg
            4
        ),
        'YoY_Growth': [min(last_yoy * 0.9, 0.08)] * 4  # Decaying growth rate
    })
    
    # Make predictions
    future_predictions = predict_model(final_model, data=future_data)
    
    # Calculate error metrics on the training data
    predictions = predict_model(final_model, data=processed_df)
    mae = mean_absolute_error(processed_df['TotalSpent'], predictions['prediction_label'])
    mape = mean_absolute_percentage_error(processed_df['TotalSpent'], predictions['prediction_label'])
    
    # Save model
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"voucher_model_{timestamp}.pkl"
    save_model(final_model, os.path.join(MODEL_DIR, 'models', model_filename))
    
    return {
        "model_type": type(final_model).__name__,
        "mae": mae,
        "mape": mape,
        "future_years": future_years,
        "future_predictions": future_predictions['prediction_label'].tolist(),
        "historical_data": processed_df.to_dict('records'),
        "last_year": last_year,
        "last_year_spent": processed_df['TotalSpent'].iloc[-1],
        "growth_rate": last_yoy,
        "timestamp": timestamp
    }
