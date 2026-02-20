from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables for model and scaler
model = None
scaler = None
MODEL_PATH = 'heart_model.pkl'
SCALER_PATH = 'scaler.pkl'

# ============== LOAD OR TRAIN MODEL ==============
def load_or_train_model():
    global model, scaler
    
    # Try to load existing model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Loading pre-trained model...")
        model = pickle.load(open(MODEL_PATH, 'rb'))
        scaler = pickle.load(open(SCALER_PATH, 'rb'))
    else:
        print("🔧 Training new model...")
        train_model()

def train_model():
    """Train a Random Forest model on sample heart disease data"""
    global model, scaler
    
    # Sample training data (you can replace this with real data)
    # Features: age, sex, cp, trestbps, chol, thalach
    X_train = np.array([
        [63, 1, 3, 145, 233, 150],
        [37, 1, 2, 130, 250, 187],
        [41, 0, 1, 130, 204, 172],
        [56, 1, 1, 120, 236, 178],
        [57, 0, 0, 120, 354, 163],
        [63, 1, 0, 130, 254, 147],
        [53, 1, 0, 140, 203, 155],
        [57, 1, 0, 140, 192, 148],
        [56, 0, 1, 140, 294, 153],
        [44, 1, 1, 120, 220, 170],
        [52, 1, 0, 172, 199, 162],
        [57, 1, 0, 150, 168, 174],
        [48, 1, 2, 110, 229, 168],
        [54, 1, 0, 140, 239, 160],
        [48, 0, 2, 130, 275, 139],
        [49, 1, 1, 130, 266, 171],
        [64, 1, 0, 110, 211, 144],
        [58, 0, 1, 150, 283, 162],
        [50, 0, 2, 120, 219, 158],
        [55, 0, 0, 110, 264, 144],
        [65, 0, 2, 140, 417, 157],
        [48, 1, 2, 130, 245, 180],
        [63, 0, 0, 150, 407, 154],
        [55, 1, 0, 140, 217, 111],
        [65, 1, 3, 138, 282, 174],
        [48, 0, 1, 120, 284, 120],
        [63, 0, 0, 161, 252, 97],
        [51, 1, 2, 125, 245, 144],
        [55, 1, 0, 140, 271, 182],
        [54, 1, 0, 150, 158, 187],
    ])
    
    # Labels: 0 = Low Risk, 1 = High Risk
    y_train = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    
    # Initialize and train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))
    print("✅ Model trained and saved!")

# ============== ROUTES ==============
@app.route('/')
def home():
    """Serve the frontend HTML"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease risk
    
    Expected JSON input:
    {
        "age": int,
        "sex": int (0=Female, 1=Male),
        "cp": int (chest pain type 0-3),
        "trestbps": int (resting blood pressure),
        "chol": int (serum cholesterol),
        "thalach": int (max heart rate achieved)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing required fields",
                "required": required_fields
            }), 400
        
        # Extract and validate data
        age = float(data['age'])
        sex = float(data['sex'])
        cp = float(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        thalach = float(data['thalach'])
        
        # Validate ranges
        if not (0 <= age <= 150):
            return jsonify({"error": "Age must be between 0 and 150"}), 400
        if sex not in [0, 1]:
            return jsonify({"error": "Sex must be 0 (Female) or 1 (Male)"}), 400
        if not (0 <= cp <= 3):
            return jsonify({"error": "Chest pain type must be 0-3"}), 400
        if not (0 <= trestbps <= 200):
            return jsonify({"error": "Blood pressure must be between 0 and 200"}), 400
        if not (0 <= chol <= 600):
            return jsonify({"error": "Cholesterol must be between 0 and 600"}), 400
        if not (0 <= thalach <= 250):
            return jsonify({"error": "Max heart rate must be between 0 and 250"}), 400
        
        # Prepare features for prediction
        features = np.array([[age, sex, cp, trestbps, chol, thalach]])
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Convert prediction to human-readable format
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = float(probability[prediction] * 100)
        
        return jsonify({
            "prediction": risk_level,
            "confidence": round(confidence, 2),
            "low_risk_prob": round(float(probability[0]) * 100, 2),
            "high_risk_prob": round(float(probability[1]) * 100, 2)
        })
    
    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

# ============== ERROR HANDLERS ==============
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============== MAIN ==============
if __name__ == '__main__':
    print("🚀 Initializing SmartCare Backend...")
    load_or_train_model()
    print("✅ Backend ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)