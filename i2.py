from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print(data)

    age = int(data.get("age") or 0)
    bp = int(data.get("trestbps") or 0)
    chol = int(data.get("chol", 0))
    thalach = int(data.get("thalach", 0))

    # Simple demo logic (you can change later)
    if age > 50 and bp > 140:
        result = "High Risk"
    else:
        result = "Low Risk"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)