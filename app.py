from flask import Flask, render_template, request, jsonify
import joblib
import warnings
import pandas as pd
import zipfile
import os
from flask_cors import CORS

# Suppress sklearn version mismatch warning
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)

# Use environment variables or defaults for file paths (for Render compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ZIP_PATH = os.environ.get('MODEL_ZIP_PATH', os.path.join(BASE_DIR, 'modell.zip'))
MODEL_EXTRACT_DIR = os.environ.get('MODEL_EXTRACT_DIR', os.path.join(BASE_DIR, 'modell_extracted'))
MODEL_FILENAME = os.environ.get('MODEL_FILENAME', 'model.pkl')
DF_PATH = os.environ.get('DF_PATH', os.path.join(BASE_DIR, 'Datas.csv'))

# Unzip only if not already extracted
model = None
if os.path.exists(MODEL_ZIP_PATH):
    if not os.path.exists(os.path.join(MODEL_EXTRACT_DIR, MODEL_FILENAME)):
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_EXTRACT_DIR)
    MODEL_PATH = os.path.join(MODEL_EXTRACT_DIR, MODEL_FILENAME)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}: {type(model)}")
    else:
        print(f"ERROR: Model file {MODEL_PATH} not found after extraction.")
else:
    print(f"ERROR: Model zip file {MODEL_ZIP_PATH} not found.")

# Load your CSV for lookup (do this once, at the top)
try:
    df = pd.read_csv(DF_PATH)
except FileNotFoundError:
    print(f"ERROR: Could not find {DF_PATH}. Please check the file path and make sure the file exists.")
    df = None

@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ''
    if request.method == 'POST':
        input_data = request.form.get('input_field', '').strip()
        print("Received input:", input_data)
        if not input_data:
            prediction_text = "Please enter input values."
        else:
            try:
                input_list = [float(x.strip()) for x in input_data.split(',')]
                processed_input = [input_list]
                print("Processed input for model:", processed_input)
                if model:
                    prediction = model.predict(processed_input)
                    print("Model prediction:", prediction)
                    result = prediction[0]
                    prediction_text = f"Result: {result}"
                else:
                    prediction_text = "Model not loaded."
            except Exception as e:
                print("Prediction error:", str(e))
                prediction_text = f"Prediction failed: {str(e)}"
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'message': 'Use POST with JSON body: {"symptoms": "...", "age": ...}'})
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        age = data.get('age', '')

        print("Raw symptoms:", symptoms, "| Raw age:", age)

        if not isinstance(symptoms, str):
            return jsonify({'error': 'Symptoms must be text.'})
        try:
            age_value = float(age)
        except Exception:
            age_value = None

        if not model:
            return jsonify({'error': 'Model not loaded on server.'})

        model_input = [symptoms]
        print("Trying model_input:", model_input)
        try:
            prediction = model.predict(model_input)
        except Exception as e:
            print("Failed with [symptoms]:", e)
            model_input = [[symptoms]]
            print("Trying model_input:", model_input)
            prediction = model.predict(model_input)

        print("Model prediction:", prediction)
        predicted_condition = prediction[0]
        print("Predicted condition:", predicted_condition)

        # Lookup in DataFrame for remedy, medicine, advice
        if df is not None:
            match = df[df['condition'].str.strip().str.lower() == str(predicted_condition).strip().lower()]
            if not match.empty:
                row = match.iloc[0]
                remedy = row.get('natural_remedy', '')
                medicine = row.get('medicine', '')
                advice = row.get('advice', '')
                mg_suggestion = ""
                if age_value:
                    if age_value < 12:
                        mg_suggestion = " (child dose)"
                    elif age_value > 60:
                        mg_suggestion = " (mild dose)"
                    else:
                        mg_suggestion = " (standard dose)"
                medicine = f"{medicine}{mg_suggestion}"
            else:
                remedy = 'Stay hydrated and rest.'
                medicine = 'Paracetamol (if needed, consult doctor).'
                advice = 'If symptoms persist, seek medical attention.'
        else:
            remedy = 'Data file not found.'
            medicine = 'Data file not found.'
            advice = 'Data file not found.'

        return jsonify({
            'condition': str(predicted_condition),
            'remedy': str(remedy),
            'medicine': str(medicine),
            'advice': str(advice)
        })
    except Exception as e:
        import traceback
        print("Prediction failed:", traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
