# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
import os

# Initialize the Flask application
app = Flask(__name__)

# --- CORRECTED CODE ---
# Use relative paths to find the model and scaler files.
# This assumes the files are in the same directory as app.py.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
# --- END CORRECTED CODE ---

# Load the trained model and the scaler
# It is CRUCIAL to use the same scaler that was used to train the model
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Could not find model or scaler file. Check the path: {e}")
    # Exit if the model/scaler are not found, as the app cannot function
    exit()

# List of the 57 features (column names) used in the training data
# This is required to create a DataFrame for the model prediction
feature_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_semicolon', 'char_freq_parenthesis', 'char_freq_square_bracket',
    'char_freq_exclamation', 'char_freq_dollar', 'char_freq_pound',
    'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total'
]

# Function to extract features from a given email text
# This function is a simplified version of the preprocessing logic
def extract_features(text, feature_list):
    # Initialize a dictionary to hold the feature values
    features = {name: 0.0 for name in feature_list}
    text = text.lower() # Convert to lowercase for consistent counting

    # --- Word Frequency Features ---
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    if word_count > 0:
        for word in words:
            word_key = f'word_freq_{word}'
            if word_key in features:
                features[word_key] += 1
    
    # Calculate word frequencies as percentages
    if word_count > 0:
        for key in features:
            if key.startswith('word_freq_'):
                features[key] = (features[key] / word_count) * 100

    # --- Character Frequency Features ---
    features['char_freq_semicolon'] = (text.count(';') / len(text)) * 100
    features['char_freq_parenthesis'] = (text.count('(') / len(text)) * 100
    features['char_freq_square_bracket'] = (text.count('[') / len(text)) * 100
    features['char_freq_exclamation'] = (text.count('!') / len(text)) * 100
    features['char_freq_dollar'] = (text.count('$') / len(text)) * 100
    features['char_freq_pound'] = (text.count('#') / len(text)) * 100

    # --- Capital Letter Features ---
    caps_runs = re.findall(r'[A-Z]+', text)
    features['capital_run_length_total'] = sum(len(run) for run in caps_runs)
    if len(caps_runs) > 0:
        features['capital_run_length_longest'] = max(len(run) for run in caps_runs)
        features['capital_run_length_average'] = features['capital_run_length_total'] / len(caps_runs)
    else:
        features['capital_run_length_longest'] = 0
        features['capital_run_length_average'] = 0

    return pd.DataFrame([features])

# Define the main route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route for the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            email_text = request.form['email_text']
            
            if not email_text:
                return jsonify({'prediction': 'Please enter some text.'})

            # Extract features from the input text
            features_df = extract_features(email_text, feature_names)

            # Scale the features using the loaded scaler
            features_scaled = scaler.transform(features_df)
            
            # Make a prediction with the loaded model
            prediction = model.predict(features_scaled)
            
            # Get the prediction probability to show more info
            prediction_proba = model.predict_proba(features_scaled)[0]

            # Return the result as a JSON object
            if prediction[0] == 1:
                result = f'SPAM (Confidence: {prediction_proba[1]:.2%})'
            else:
                result = f'NOT SPAM (Confidence: {prediction_proba[0]:.2%})'

            return jsonify({'prediction': result})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)