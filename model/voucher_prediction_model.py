import numpy as np
import pandas as pd
from pycaret.regression import *
from sklearn.model_selection import train_test_split
import joblib
import datetime
import os

# creates directory for saved models
MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(MODEL_DIR + '/models'):
    os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_predict(df):
    if 'Year' not in df.columns or 'TotalSpent(RM)' not in df.columns:
        raise ValueError("Missing 'Year' or 'TotalSpent(RM)' column")   

    # Prepare data
    X = df[['Year']]
    y = df['TotalSpent(RM)']
    
    # Initialize PyCaret
    setup_data = pd.concat([X, y], axis=1)
    setup_data.columns = ['Year', 'TotalSpent(RM)']  # Ensure proper column names
    
    # Initialize PyCaret regression
    reg = setup(
        data=setup_data,
        target='TotalSpent(RM)',
        train_size=0.8,
        session_id=42,
        silent=True,  # Set to False if you want to see progress
        verbose=False
    )
    
    # Compare models and select the best one
    best_model = compare_models(sort='R2', n_select=1)
    
    # Finalize the best model (trains on entire dataset)
    final_model = finalize_model(best_model)
    
    # Make predictions on test set (held back during setup)
    predictions = predict_model(final_model)
    
    # Get metrics
    metrics = pull()  # Gets the metrics dataframe
    
    # Future predictions
    future_years = np.array([2024, 2025, 2026, 2027]).reshape(-1, 1)
    future_df = pd.DataFrame(future_years, columns=['Year'])
    future_predictions = predict_model(final_model, data=future_df)
    
    # Save the trained model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"voucher_prediction_model_{timestamp}.pkl"
    save_model(final_model, os.path.join(MODEL_DIR, model_filename))
    
    return {
        "model_type": type(final_model).__name__,
        "r2": metrics.loc['Mean']['R2'],  # Mean R2 from CV
        "mse": metrics.loc['Mean']['MSE'],  # Mean MSE from CV
        "future_years": future_years.flatten().tolist(),
        "future_predictions": future_predictions['prediction_label'].tolist(),
        "timestamp": timestamp,
        "model_filename": model_filename
    }