import numpy as np
import pandas as pd
from pycaret.regression import *
import os
import glob
from datetime import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
FOLDER_PATH = r"C:\Users\acer\Desktop\Coding Projects\Machine Learning\SmartSpend\uploads"
FILENAME_PATTERN = "ML Test Data"
SHEET_NAME = "Table1"  #reads only the needed sheet in excel, and not the others


MODEL_DIR = os.path.dirname(__file__)
if not os.path.exists(os.path.join(MODEL_DIR, 'models')):
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)

# ========== FUNCTIONS ==========

def get_file_by_name_pattern(folder_path, pattern, extension="xlsx"):
    files = glob.glob(os.path.join(folder_path, f"*{pattern}*.{extension}"))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {folder_path}")
    return files[0]

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


def train_and_predict_yearly(df):
    processed_df = preprocess_yearly_data(df)

    # Setup PyCaret regression
    reg = setup(
        data=processed_df,
        target='TotalSpent',
        session_id=42,
        verbose=False,
        numeric_features=['TransactionCount', 'AverageSpent'],
        fold=3,
        train_size=0.8,
        remove_outliers=False,
        normalize=True
    )

    # Compare models
    best_model = compare_models(sort='MAE', exclude=['catboost', 'lightgbm'])
    compare_df = pull()

    # Tune and finalise
    tuned_model = tune_model(best_model, optimize='MAE')
    final_model = finalize_model(tuned_model)

    # ========== Generate future data for next 4 years ==========
    last_year = processed_df['Year'].max()
    future_years = list(range(last_year + 1, last_year + 5))

    last_trans_count = processed_df['TransactionCount'].iloc[-1]
    last_avg_spent = processed_df['AverageSpent'].iloc[-1]

    future_data = pd.DataFrame({
        'Year': future_years,
        'TransactionCount': np.round(np.linspace(
            last_trans_count,
            last_trans_count * 1.10,  # Slight growth over 4 years
            4
        )),
        'AverageSpent': np.linspace(
            last_avg_spent,
            last_avg_spent * 1.05,  # Slight growth over 4 years
            4
        )
    })

    # Clean future data
    future_data_cleaned = future_data.dropna()

    # Predict future spending
    future_predictions = predict_model(final_model, data=future_data_cleaned)
    future_data_cleaned['PredictedSpent'] = future_predictions['prediction_label']

    # Print future yearly predictions
    print("\n🔮 Predicted Spending for Next 4 Years:")
    for _, row in future_data_cleaned.iterrows():
        print(f"Year {int(row['Year'])}: RM {row['PredictedSpent']:,.2f}")

    # Calculate training errors
    predictions = predict_model(final_model, data=processed_df)
    mae = mean_absolute_error(processed_df['TotalSpent'], predictions['prediction_label'])
    mape = mean_absolute_percentage_error(processed_df['TotalSpent'], predictions['prediction_label'])
    r2 = r2_score(processed_df['TotalSpent'], predictions['prediction_label'])
    r2_accuracy = max(0, min(1, r2)) * 100

    print(f"\n📈 Model R²: {r2:.4f} (Accuracy: {r2_accuracy:.2f}%)")

    # Save model
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"voucher_yearly_model_{timestamp}.pkl"
    save_model(final_model, os.path.join(MODEL_DIR, 'models', model_filename))

    return {
        "model_type": type(final_model).__name__,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "r2_accuracy": r2_accuracy,
        "future_data": future_data_cleaned,
        "yearly_data": processed_df,
        "all_models_results": compare_df,
        "timestamp": timestamp
    }


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    try:
        file_path = get_file_by_name_pattern(FOLDER_PATH, pattern=FILENAME_PATTERN)
        print("Loaded file:", file_path) 
        SHEET_NAME = "Table1"

        df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
        print("✅ Loaded sheet columns:", df.columns)

        df = pd.read_excel(file_path, sheet_name=SHEET_NAME) 
        result = train_and_predict_yearly(df)

        # ========== PRINT RESULTS ==========
        print("\n✅ Best Model:", result["model_type"])
        print("MAE:", result["mae"])
        print("MAPE:", result["mape"])
        print(f"R²: {result['r2']:.4f}")
        print("🎯 Model Accuracy (based on R²): {:.2f}%".format(result["r2_accuracy"]))

        print("\n📊 All Models Tried and Their Metrics:")
        for model in result["all_models_results"].to_dict('records'):
            print(model)

        # ========== VISUALISATIONS ==========

        # 1. Historical yearly spending
        yearly_df = result["yearly_data"]
        plt.figure(figsize=(8,5))
        plt.bar(yearly_df['Year'].astype(str), yearly_df['TotalSpent'], color='skyblue')
        
        #Add data labels
        for i, value in enumerate(yearly_df['TotalSpent']):
            plt.text(yearly_df['Year'].astype(str).iloc[i], value, f'{value:,.0f}', ha='center', va='bottom', fontsize=12)
        
        plt.title('Yearly Voucher Spending (RM)')
        plt.xlabel('Year')
        plt.ylabel('Total Spent (RM)')
        plt.tight_layout()
        plt.show()

        # 2. Future predicted yearly spending
        future_df = result["future_data"]
        plt.figure(figsize=(8,5))
        plt.plot(future_df['Year'].astype(str), future_df['PredictedSpent'], marker='o', color='mediumseagreen')

        # Add data labels
        for i, value in enumerate(future_df['PredictedSpent']):
            plt.text(future_df['Year'].astype(str).iloc[i], value, f'{value:,.0f}', 
                    ha='center', va='bottom', fontsize=9)

        plt.title('Predicted Spending for Next 4 Years')
        plt.xlabel('Future Year')
        plt.ylabel('Predicted Total Spent (RM)')
        plt.grid(True, which='both', linestyle='-', alpha=0.6)
        plt.tight_layout()
        plt.show()


    except Exception as e:
        print("❌ Error:", e)
