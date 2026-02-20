from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "SmartCare Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    age = data.get("age")
    bp = data.get("bp")

    if age > 50 and bp > 140:
        result = "High Risk"
    else:
        result = "Low Risk"

    return jsonify({"risk": result})

if __name__ == "__main__":
    app.run(debug=True)