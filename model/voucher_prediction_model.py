import numpy as np
import pandas as pd
import os
from datetime import datetime as dt
from sklearn.linear_model import Lars
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(os.path.join(MODEL_DIR, 'models')):
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)

def preprocess_yearly_data(df):
    df = df.dropna(subset=['Year', 'Status', 'Price(RM)'])
    df['Price(RM)'] = pd.to_numeric(df['Price(RM)'], errors='coerce')
    df = df.dropna(subset=['Price(RM)'])
    df = df[df['Status'] == 'Reward Collected']

    # Ensure Year is numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    yearly_data = df.groupby('Year').agg({
        'Price(RM)': ['sum', 'count', 'mean'],
    })
    yearly_data.columns = ['TotalSpent', 'TransactionCount', 'AverageSpent']
    yearly_data = yearly_data.reset_index()
    yearly_data = yearly_data.sort_values('Year')
    return yearly_data

def train_and_predict(df):
    processed_df = preprocess_yearly_data(df)

    # Prepare features and target
    X = processed_df[['TransactionCount', 'AverageSpent']].values
    y = processed_df['TotalSpent'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Least Angle Regression model
    model = Lars()
    model.fit(X_train, y_train)

    # Evaluate on full data for consistency with previous code
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    r2_accuracy = max(0, min(1, r2)) * 100

    # Generate future data for next 4 years
    last_year = processed_df['Year'].max()
    future_years = list(range(last_year + 1, last_year + 5))

    last_trans_count = processed_df['TransactionCount'].iloc[-1]
    last_avg_spent = processed_df['AverageSpent'].iloc[-1]

    future_data = pd.DataFrame({
        'Year': future_years,
        'TransactionCount': np.round(np.linspace(
            last_trans_count,
            last_trans_count * 1.10,
            4
        )),
        'AverageSpent': np.linspace(
            last_avg_spent,
            last_avg_spent * 1.05,
            4
        )
    })

    # Scale future features
    X_future = future_data[['TransactionCount', 'AverageSpent']].values
    X_future_scaled = scaler.transform(X_future)

    # Predict future spending
    future_predictions = model.predict(X_future_scaled)
    future_data['PredictedSpent'] = future_predictions

    # Save model
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"voucher_yearly_lars_model_{timestamp}.pkl"

    import joblib
    joblib.dump({
        'model': model,
        'scaler': scaler
    }, os.path.join(MODEL_DIR, 'models', model_filename))

    return {
        "model_type": type(model).__name__,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "r2_accuracy": r2_accuracy,
        "future_years": future_data['Year'].tolist(),
        "future_predictions": future_data['PredictedSpent'].tolist(),
        "historical_data": processed_df.to_dict('records'),
        "all_models_results": [],  # not applicable as we no longer compare multiple models
        "timestamp": timestamp
    }
