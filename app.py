from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Decision Tree model
# Make sure decision_tree.pkl is in the same directory as this file
try:
    with open('decision_tree.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return jsonify({"message": "Decision Tree Model API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract the 13 features in the exact order the model expects
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
        
        # Convert to a 2D numpy array: shape (1, 13)
        features_array = np.array([features])
        
        # Make a prediction
        prediction = model.predict(features_array)
        
        # Return the result (converted to standard Python int)
        return jsonify({'prediction': int(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app in debug mode if executed directly
    app.run(host='0.0.0.0', port=5000, debug=True)
