from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Model file name
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

# Embedded Modern HTML Interface with Dropdowns, Placeholders, and Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Predictor Interface</title>
    <style>
        :root {
            --primary: #6366f1;
            --bg: #0f172a;
            --surface: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg), #1e1b4b);
            color: var(--text);
            min-height: 100vh;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        .container {
            background-color: var(--surface);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 800px;
            animation: slideUp 0.6s ease-out forwards;
            opacity: 0;
            transform: translateY(30px);
        }
        @keyframes slideUp {
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            text-align: center;
            margin-top: 0;
            margin-bottom: 2rem;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }
        .input-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            color: var(--text-muted);
            transition: color 0.3s;
        }
        .input-group:focus-within label {
            color: var(--primary);
        }
        input, select {
            padding: 0.75rem;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            color: white;
            outline: none;
            transition: all 0.3s ease;
            font-family: inherit;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        /* Style the select dropdown options */
        option {
            background: #0f172a;
            color: white;
        }
        button {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary), #a855f7);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(168, 85, 247, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        #result {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        /* Prediction Effects */
        @keyframes pulseSuccess {
            0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
            70% { box-shadow: 0 0 15px 10px rgba(74, 222, 128, 0); }
            100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
        }
        @keyframes shakeWarning {
            0%, 100% { transform: translateX(0); }
            20%, 60% { transform: translateX(-5px); }
            40%, 80% { transform: translateX(5px); }
        }
        
        .result-low {
            opacity: 1 !important;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
            animation: pulseSuccess 2s infinite;
        }
        .result-high {
            opacity: 1 !important;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            animation: shakeWarning 0.5s ease-in-out;
        }
        .error-result {
            opacity: 1 !important;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predictive Model Interface</h1>
        <form id="predictionForm">
            <div class="form-grid">
                <!-- Dropdowns & Placeholders added -->
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" name="age" required placeholder="e.g., 18" step="any">
                </div>
                <div class="input-group">
                    <label>Gender</label>
                    <select name="gender" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Daily Screen Time (hrs)</label>
                    <input type="number" name="daily_screen_time_hours" required placeholder="e.g., 5" step="any">
                </div>
                <div class="input-group">
                    <label>Social Media (hrs)</label>
                    <input type="number" name="social_media_hours" required placeholder="e.g., 2" step="any">
                </div>
                <div class="input-group">
                    <label>Gaming (hrs)</label>
                    <input type="number" name="gaming_hours" required placeholder="e.g., 1" step="any">
                </div>
                <div class="input-group">
                    <label>Work/Study (hrs)</label>
                    <input type="number" name="work_study_hours" required placeholder="e.g., 8" step="any">
                </div>
                <div class="input-group">
                    <label>Sleep (hrs)</label>
                    <input type="number" name="sleep_hours" required placeholder="e.g., 7" step="any">
                </div>
                <div class="input-group">
                    <label>Notifications per Day</label>
                    <input type="number" name="notifications_per_day" required placeholder="e.g., 50" step="any">
                </div>
                <div class="input-group">
                    <label>App Opens per Day</label>
                    <input type="number" name="app_opens_per_day" required placeholder="e.g., 30" step="any">
                </div>
                <div class="input-group">
                    <label>Weekend Screen Time (hrs)</label>
                    <input type="number" name="weekend_screen_time" required placeholder="e.g., 6" step="any">
                </div>
                <div class="input-group">
                    <label>Stress Level (0-10)</label>
                    <input type="number" name="stress_level" required placeholder="e.g., 4" min="0" max="10" step="any">
                </div>
                <div class="input-group">
                    <label>Academic/Work Impact (0-10)</label>
                    <input type="number" name="academic_work_impact" required placeholder="e.g., 3" min="0" max="10" step="any">
                </div>
                <div class="input-group">
                    <label>Addiction Level (0-10)</label>
                    <input type="number" name="addiction_level" required placeholder="e.g., 2" min="0" max="10" step="any">
                </div>
            </div>
            <button type="submit">Predict Now</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const resultDiv = document.getElementById('result');
            resultDiv.className = '';
            resultDiv.innerText = 'Calculating...';
            resultDiv.style.opacity = 1;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Output updated to "Not Addicted" and "Addicted"
                    if (result.prediction === 0) {
                        resultDiv.innerText = 'Prediction: Not Addicted';
                        resultDiv.className = 'result-low';
                    } else {
                        resultDiv.innerText = 'Prediction: Addicted';
                        resultDiv.className = 'result-high';
                    }
                } else {
                    resultDiv.innerText = `Error: ${result.error}`;
                    resultDiv.className = 'error-result';
                }
            } catch (error) {
                resultDiv.innerText = 'Failed to connect to the server.';
                resultDiv.className = 'error-result';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on the server.'}), 500
        
    try:
        data = request.get_json()
        
        # Extract features ensuring exact order
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
