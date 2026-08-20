from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Updated to match your exact file name
MODEL_PATH = 'decision_tree_model.pkl'

# Load the trained Decision Tree model safely
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        print(f"Error loading the model: {e}")
else:
    print(f"File {MODEL_PATH} not found. Make sure it is in the same directory.")

@app.route('/')
def home():
    # This will load the modern interface from the templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on the server.'}), 500
        
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('age', 0)),
            float(data.get('gender', 0)),
            float(data.get('daily_screen_time_hours', 0)),
            float(data.get('social_media_hours', 0)),
            float(data.get('gaming_hours', 0)),
            float(data.get('work_study_hours', 0)),
            float(data.get('sleep_hours', 0)),
            float(data.get('notifications_per_day', 0)),
            float(data.get('app_opens_per_day', 0)),
            float(data.get('weekend_screen_time', 0)),
            float(data.get('stress_level', 0)),
            float(data.get('academic_work_impact', 0)),
            float(data.get('addiction_level', 0))
        ]
        
        features_array = np.array([features])
        prediction = model.predict(features_array)
        
        return jsonify({'prediction': int(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
