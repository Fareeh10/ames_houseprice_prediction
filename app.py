from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("ridge_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f1 = float(request.form["feature1"])
    f2 = float(request.form["feature2"])
    f3 = float(request.form["feature3"])
    # Add more as needed

    features = np.array([[f1, f2, f3]])  # Adjust dimensions
    pred = model.predict(features)[0]
    return render_template("index.html", prediction_text=f"Predicted Price: ${pred:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)