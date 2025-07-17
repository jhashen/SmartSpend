from flask import Flask, render_template, request, url_for
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from model.voucher_prediction_model import train_and_predict  # Your integrated ML function

app = Flask(__name__, static_folder='static')

# Ensure temp_uploads folder exists
TEMP_UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

#loads home page
@app.route('/')
def home():
    return render_template('frontend_design.html')

#app flow
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('frontend_design.html', prediction="❌ No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('frontend_design.html', prediction="❌ Empty file uploaded")

    try:
        # Save file temporarily with unique name
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_file_path = os.path.join(TEMP_UPLOAD_FOLDER, unique_filename)
        file.save(temp_file_path)

        # Read available sheets
        xls = pd.ExcelFile(temp_file_path)
        sheet_names = xls.sheet_names

        # Required columns for ML model
        required_columns = {'Year', 'Status', 'Price(RM)'}

        # If only one sheet, proceed directly
        if len(sheet_names) == 1:
            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])

            # Validate columns
            if not required_columns.issubset(df.columns):
                os.remove(temp_file_path)
                return render_template('frontend_design.html', prediction=f"⚠️ Selected sheet does not contain required columns: {', '.join(required_columns)}.")

            result = train_and_predict(df)
            os.remove(temp_file_path)  # Clean up

            return render_prediction(result)

        else:
            # Render sheet selection page
            return render_template('select_sheet.html', sheet_names=sheet_names, temp_file=unique_filename)

    except Exception as e:
        return render_template('frontend_design.html', prediction=f"⚠️ Error: {str(e)}")

@app.route('/predict_sheet', methods=['POST'])
def predict_sheet():
    selected_sheet = request.form['sheet']
    temp_file = request.form['temp_file']
    temp_file_path = os.path.join(TEMP_UPLOAD_FOLDER, temp_file)

    try:
        # Load selected worksheet
        df = pd.read_excel(temp_file_path, sheet_name=selected_sheet)

        # Required columns for ML model
        required_columns = {'Year', 'Status', 'Price(RM)'}
        if not required_columns.issubset(df.columns):
            os.remove(temp_file_path)
            return render_template('frontend_design.html', prediction=f"⚠️ Selected sheet does not contain required columns: {', '.join(required_columns)}.")

        result = train_and_predict(df)
        os.remove(temp_file_path)  # Clean up

        return render_prediction(result)

    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return render_template('frontend_design.html', prediction=f"⚠️ Error: {str(e)}")

def render_prediction(result):
    # Plotting
    plot_filename = 'prediction_plot.png'
    plot_path = os.path.join('static', plot_filename)

    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Load historical processed data from result
    if 'historical_data' not in result or len(result['historical_data']) == 0:
        return render_template('frontend_design.html', prediction="⚠️ No historical data available to plot.")

    processed_df = pd.DataFrame(result['historical_data'])

    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Year'], processed_df['TotalSpent'], marker='o', label='Actual Spending', color='blue')
    plt.plot(result['future_years'], result['future_predictions'], marker='o', linestyle='--', label='Future Predictions', color='orange')

    plt.title(f"Voucher Spending Prediction (Best Model: {result['model_type']})")
    plt.xlabel("Year")
    plt.ylabel("Total Spent (RM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    plot_url = url_for('static', filename=plot_filename) + f"?v={datetime.datetime.now().timestamp()}"

    prediction_text = (
        f"✅ Best Model: {result['model_type']}<br>"
        f"📅 Future Predictions:<br>"
        f"&nbsp;&nbsp;{result['future_years'][0]}: RM {result['future_predictions'][0]:.2f}<br>"
        f"&nbsp;&nbsp;{result['future_years'][1]}: RM {result['future_predictions'][1]:.2f}<br>"
        f"&nbsp;&nbsp;{result['future_years'][2]}: RM {result['future_predictions'][2]:.2f}<br>"
        f"&nbsp;&nbsp;{result['future_years'][3]}: RM {result['future_predictions'][3]:.2f}<br>"
        f"📊 Model Metrics - MAE: {result['mae']:.2f}, MAPE: {result['mape']:.2%}"
    )

    return render_template('frontend_design.html', prediction=prediction_text, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
