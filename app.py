from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = float(request.form["gender"])
    tb = float(request.form["tb"])
    db = float(request.form["db"])
    alk = float(request.form["alk"])
    alt = float(request.form["alt"])
    ast = float(request.form["ast"])
    tp = float(request.form["tp"])
    alb = float(request.form["alb"])
    ratio = float(request.form["ratio"])

    features = np.array([[
        age,
        gender,
        tb,
        db,
        alk,
        alt,
        ast,
        tp,
        alb,
        ratio
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "⚠️ Liver Disease Detected"
    else:
        result = "✅ No Liver Disease"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)