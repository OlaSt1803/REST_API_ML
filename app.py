from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = load('wine_model.pkl')
scaler = load('scaler.joblib')

wine_classes = ['class_0', 'class_1', 'class_2']

@app.route('/')
def home():
    return "API is working. Use the /predict endpoint for predictions."

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the incoming request
        input_data = request.json.get('features', None)
        if not input_data:
            return jsonify({'error': 'No input features provided'}), 400

        print(f"Received input: {input_data}")

        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        predicted_class = wine_classes[prediction]

        print(f"Prediction: {prediction} ({predicted_class})")
        # Return the prediction as JSON
        return jsonify({
            'prediction': int(prediction),
            'predicted_class': predicted_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
