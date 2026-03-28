from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

labels = ["Car", "Truck", "Ambulance", "Bus", "Bike"]

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        # Preprocess image
        image = Image.open(file).convert("L")
        image = image.resize((64, 64))
        image = np.array(image) / 255.0
        image = image.reshape(1, -1)

        # Prediction
        prediction = model.predict(image)[0]

        # ✅ SAFE CONFIDENCE (no error)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(image)
            confidence = float(max(proba[0]))
        else:
            confidence = 1.0  # fallback if not supported

        return jsonify({
            "prediction": labels[prediction],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)