from flask import Flask, render_template, request, url_for
import os
import matplotlib.pyplot as plt
from model.voucher_prediction_model import train_and_predict  # Correct import
import pandas as pd

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('frontend_design.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('frontend_design.html', prediction="❌ No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('frontend_design.html', prediction="❌ Empty file uploaded")

    try:
        df = pd.read_excel(file)
        result = train_and_predict(df)

        # Clean old plot
        plot_filename = 'prediction_plot.png'
        plot_path = os.path.join('static', plot_filename)
        plot_url = url_for('static', filename=plot_filename) + f"?v={os.path.getmtime(plot_path)}"
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], df['TotalSpent(RM)'], marker='o', label='Actual Spending', color='blue')
        plt.plot(result['future_years'].flatten(), result['future_predictions'], marker='o', linestyle='--', label='Future Predictions', color='orange')
        plt.title("Voucher Spending Prediction")
        plt.xlabel("Year")
        plt.ylabel("Total Spent (RM)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return render_template(
            'frontend_design.html',
            prediction=(
                f"✅ Suggested spend for 2024: RM {result['future_predictions'][0]:.2f}<br>"
                f"📊 R² Score: {result['r2']:.2f}, MSE: {result['mse']:.2f}"
            ),
        plot_url=plot_url
)


    except Exception as e:
        return render_template('frontend_design.html', prediction=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)