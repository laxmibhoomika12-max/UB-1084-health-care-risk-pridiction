from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# ============== LOGGING SETUP ==============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables for model and scaler
model = None
scaler = None
MODEL_PATH = 'heart_model.pkl'
SCALER_PATH = 'scaler.pkl'
PREDICTIONS_LOG = 'predictions.log'

# ============== LOAD OR TRAIN MODEL ==============
def load_or_train_model():
    """Load existing model or train a new one"""
    global model, scaler
    
    try:
        # Try to load existing model
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            logger.info("✅ Loading pre-trained model...")
            model = pickle.load(open(MODEL_PATH, 'rb'))
            scaler = pickle.load(open(SCALER_PATH, 'rb'))
            logger.info("✅ Model loaded successfully!")
        else:
            logger.info("🔧 Training new model...")
            train_model()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        train_model()

def train_model():
    """Train a Random Forest model on sample heart disease data"""
    global model, scaler
    
    try:
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
        logger.info("✅ Model trained and saved successfully!")
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

# ============== UTILITY FUNCTIONS ==============
def log_prediction(data, prediction, confidence):
    """Log prediction for audit trail"""
    try:
        with open(PREDICTIONS_LOG, 'a') as f:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'input': data,
                'prediction': prediction,
                'confidence': confidence
            }
            f.write(f"{log_entry}\n")
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def validate_input(data):
    """Validate input data"""
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    
    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return False, f"Missing required fields: {required_fields}"
    
    try:
        age = float(data['age'])
        sex = float(data['sex'])
        cp = float(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        thalach = float(data['thalach'])
        
        # Validate ranges
        validations = [
            (0 <= age <= 150, "Age must be between 0 and 150"),
            (sex in [0, 1], "Sex must be 0 (Female) or 1 (Male)"),
            (0 <= cp <= 3, "Chest pain type must be 0-3"),
            (0 <= trestbps <= 200, "Blood pressure must be between 0 and 200"),
            (0 <= chol <= 600, "Cholesterol must be between 0 and 600"),
            (0 <= thalach <= 250, "Max heart rate must be between 0 and 250"),
        ]
        
        for condition, message in validations:
            if not condition:
                return False, message
        
        return True, (age, sex, cp, trestbps, chol, thalach)
    
    except ValueError as e:
        return False, f"Invalid input values: {str(e)}"

# ============== ROUTES ==============
@app.route('/')
def home():
    """Serve the frontend HTML"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return jsonify({"error": "Unable to load homepage"}), 500

@app.route('/api/predict', methods=['POST'])
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
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        is_valid, result = validate_input(data)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        age, sex, cp, trestbps, chol, thalach = result
        
        # Prepare features for prediction
        features = np.array([[age, sex, cp, trestbps, chol, thalach]])
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Convert prediction to human-readable format
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = float(probability[prediction] * 100)
        
        response = {
            "prediction": risk_level,
            "confidence": round(confidence, 2),
            "low_risk_prob": round(float(probability[0]) * 100, 2),
            "high_risk_prob": round(float(probability[1]) * 100, 2),
            "status": "success"
        }
        
        # Log prediction
        log_prediction(data, risk_level, confidence)
        logger.info(f"Prediction made: {risk_level} (Confidence: {confidence}%)")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}", "status": "error"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        "app_name": "SmartCare Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict (POST)",
            "health": "/api/health (GET)",
            "info": "/api/info (GET)"
        },
        "features": ["age", "sex", "cp", "trestbps", "chol", "thalach"]
    }), 200

# ============== ERROR HANDLERS ==============
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "status": "error"}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}")
    return jsonify({"error": "Internal server error", "status": "error"}), 500

# ============== MAIN ==============
if __name__ == '__main__':
    try:
        logger.info("🚀 Initializing SmartCare Backend...")
        load_or_train_model()
        logger.info("✅ Backend ready! Starting server on http://0.0.0.0:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise