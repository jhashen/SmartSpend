import numpy as np
import pandas as pd
from pycaret.regression import *
import os
import glob
from datetime import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
FOLDER_PATH = r"C:\Users\acer\Desktop\Coding Projects\Machine Learning\SmartSpend\uploads"
FILENAME_PATTERN = "dummy_data"

MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(os.path.join(MODEL_DIR, 'models')):
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)

# ========== FUNCTIONS ==========

def get_file_by_name_pattern(folder_path, pattern, extension="xlsx"):
    files = glob.glob(os.path.join(folder_path, f"*{pattern}*.{extension}"))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {folder_path}")
    return files[0]

def preprocess_monthly_data(df):
    df = df.dropna(subset=['Date Submitted', 'Status', 'Price(RM)'])
    df['Date Submitted'] = pd.to_datetime(df['Date Submitted'], errors='coerce')
    df = df.dropna(subset=['Date Submitted'])
    df['Price(RM)'] = pd.to_numeric(df['Price(RM)'], errors='coerce')
    df = df.dropna(subset=['Price(RM)'])
    df = df[df['Status'] == 'Reward Collected']
    df['YearMonth'] = df['Date Submitted'].dt.to_period('M').astype(str)
    monthly_data = df.groupby('YearMonth').agg({
        'Price(RM)': ['sum', 'count', 'mean'],
    })
    monthly_data.columns = ['TotalSpent', 'TransactionCount', 'AverageSpent']
    monthly_data = monthly_data.reset_index()
    monthly_data['MonthIndex'] = range(1, len(monthly_data)+1)  # For potential trend modelling
    return monthly_data

def train_and_predict_monthly(df):
    processed_df = preprocess_monthly_data(df)

    # Setup PyCaret regression
    reg = setup(
        data=processed_df,
        target='TotalSpent',
        session_id=42,
        verbose=False,
        numeric_features=['TransactionCount', 'AverageSpent', 'MonthIndex'],
        fold=3,
        train_size=0.8,
        remove_outliers=True,
        normalize=True
    )

    # Compare models
    best_model = compare_models(sort='MAE', exclude=['catboost'])
    compare_df = pull()

    # Tune and finalise
    tuned_model = tune_model(best_model, optimize='MAE')
    final_model = finalize_model(tuned_model)

    # Generate future data (next 4 months)
    last_month_index = processed_df['MonthIndex'].max()
    future_months = list(range(last_month_index + 1, last_month_index + 5))

    last_trans_count = processed_df['TransactionCount'].iloc[-1]
    last_avg_spent = processed_df['AverageSpent'].iloc[-1]

    future_data = pd.DataFrame({
        'YearMonth': [f"Future_{i}" for i in future_months],
        'TransactionCount': np.round(np.linspace(
            last_trans_count,
            last_trans_count * 1.05,
            4
        )),
        'AverageSpent': np.linspace(
            last_avg_spent,
            last_avg_spent * 1.03,
            4
        ),
        'MonthIndex': future_months
    })

    # ========== Debug prints ==========
    print("\n🔍 Future data before cleaning:")
    print(future_data)
    print(future_data.dtypes)

    # ========== Data cleaning ==========
    future_data['TransactionCount'] = pd.to_numeric(future_data['TransactionCount'], errors='coerce')
    future_data['AverageSpent'] = pd.to_numeric(future_data['AverageSpent'], errors='coerce')
    future_data['MonthIndex'] = pd.to_numeric(future_data['MonthIndex'], errors='coerce')

    # Drop rows with missing or invalid data
    future_data_cleaned = future_data.dropna()
    print("\n✅ Future data after cleaning (used for prediction):")
    print(future_data_cleaned)

    # ========== Predict ==========
    future_predictions = predict_model(final_model, data=future_data_cleaned)

    # ========== Check prediction count ==========
    print("\n🔍 Number of rows in cleaned future data:", len(future_data_cleaned))
    print("🔍 Number of predictions generated:", len(future_predictions))

    # Calculate training errors
    predictions = predict_model(final_model, data=processed_df)
    mae = mean_absolute_error(processed_df['TotalSpent'], predictions['prediction_label'])
    mape = mean_absolute_percentage_error(processed_df['TotalSpent'], predictions['prediction_label'])

    # Save model
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"voucher_monthly_model_{timestamp}.pkl"
    save_model(final_model, os.path.join(MODEL_DIR, 'models', model_filename))

    return {
        "model_type": type(final_model).__name__,
        "mae": mae,
        "mape": mape,
        "future_months": future_data_cleaned['MonthIndex'].tolist(),
        "future_predictions": future_predictions['prediction_label'].tolist(),
        "monthly_data": processed_df.to_dict('records'),
        "all_models_results": compare_df.to_dict('records'),
        "timestamp": timestamp
    }

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    try:
        file_path = get_file_by_name_pattern(FOLDER_PATH, pattern=FILENAME_PATTERN)
        print("Loaded file:", file_path)

        df = pd.read_excel(file_path)
        result = train_and_predict_monthly(df)

        # ========== PRINT RESULTS ==========
        print("\n✅ Best Model:", result["model_type"])
        print("MAE:", result["mae"])
        print("MAPE:", result["mape"])

        
        print("\n📊 All Models Tried and Their Metrics:")
        for model in result["all_models_results"]:
            print(model)

        print("\n🔮 Future Predictions (Next 4 Months):")
        for month, pred in zip(result["future_months"], result["future_predictions"]):
            print(f"Month {month}: RM {pred:,.2f}")

        # ========== VISUALISATIONS ==========

        # 1. Historical monthly spending
        monthly_df = pd.DataFrame(result["monthly_data"])
        plt.figure(figsize=(10,5))
        plt.bar(monthly_df['YearMonth'], monthly_df['TotalSpent'], color='skyblue')
        plt.title('Monthly Voucher Spending')
        plt.xlabel('Month')
        plt.ylabel('Total Spent (RM)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 2. Future predicted spending
        future_months = [f"Future_{i}" for i in result["future_months"]]
        future_preds = result["future_predictions"]

        plt.figure(figsize=(10,5))
        plt.bar(future_months, future_preds, color='lightgreen')
        plt.title('Predicted Voucher Spending for Upcoming Months')
        plt.xlabel('Future Month')
        plt.ylabel('Predicted Total Spent (RM)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("❌ Error:", e)
