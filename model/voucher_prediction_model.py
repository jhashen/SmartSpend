import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
import os

# creates directory for saved models
MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(MODEL_DIR + '/models'):
    os.makedirs(MODEL_DIR, exist_ok=True)

# main functions, data training and prediction
def train_and_predict(df):
    if 'Year' not in df.columns or 'TotalSpent(RM)' not in df.columns:
        raise ValueError("Missing 'Year' or 'TotalSpent(RM)' column")   

# data preprocessing, prepare input features and target
    X = df[['Year']]
    y = df['TotalSpent(RM)']

# split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

#predict test set, evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    future_years = np.array([[2024], [2025], [2026], [2027]])
    future_predictions = model.predict(future_years)

# save the trained model 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = "voucher_prediction_model.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_filename))

    return {
        
        "r2": r2,
        "mse": mse,
        "future_years": future_years,
        "future_predictions": future_predictions,
        "timestamp": timestamp
    }