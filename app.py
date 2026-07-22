from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("models", "heart_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    try:
        
        data = pd.DataFrame({
    "age": [int(request.form["age"])],
    "sex": [int(request.form["sex"])],
    "cp": [int(request.form["cp"])],
    "trestbps": [int(request.form["trestbps"])],
    "chol": [int(request.form["chol"])],
    "fbs": [int(request.form["fbs"])],
    "restecg": [int(request.form["restecg"])],
    "thalch": [int(request.form["thalach"])],
    "exang": [int(request.form["exang"])],
    "oldpeak": [float(request.form["oldpeak"])],
    "slope": [int(request.form["slope"])],
    "ca": [int(request.form["ca"])],
    "thal": [int(request.form["thal"])]
})

        

        pred = model.predict(data)[0]

        # Probability if model supports it
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(data)[0][1])
            risk = round(probability * 100)
        else:
            risk = 90 if pred == 1 else 15

        health_score = 100 - risk

        # High Risk
        if pred == 1:

            prediction = "High Risk of Heart Disease"
            color = "danger"

            if risk >= 90:
                risk_level = "Very High Risk"
            elif risk >= 75:
                risk_level = "High Risk"
            else:
                risk_level = "Moderate Risk"

            recommendations = [
                "Consult a cardiologist immediately.",
                "Schedule ECG/EKG and Echocardiogram.",
                "Monitor blood pressure daily.",
                "Reduce salt and saturated fat intake.",
                "Avoid smoking and alcohol.",
                "Exercise only after consulting your doctor.",
                "Maintain healthy cholesterol levels.",
                "Monitor blood sugar regularly.",
                "Take prescribed medicines regularly.",
                "Visit the hospital immediately if chest pain or breathing difficulty occurs."
            ]

        # Low Risk
        else:

            prediction = "Low Risk of Heart Disease"
            color = "success"
            risk_level = "Low Risk"

            recommendations = [
                "Continue regular physical exercise.",
                "Eat fruits and vegetables daily.",
                "Avoid smoking.",
                "Limit alcohol consumption.",
                "Maintain healthy body weight.",
                "Sleep 7–8 hours daily.",
                "Reduce stress through yoga or meditation.",
                "Monitor blood pressure every few months.",
                "Get annual health checkups.",
                "Maintain healthy cholesterol levels."
            ]

        return render_template(
            "result.html",
            prediction=prediction,
            color=color,
            risk=risk,
            health_score=health_score,
            risk_level=risk_level,
            recommendations=recommendations,
            chart_data={
                "Healthy": health_score,
                "Risk": risk
            }
        )

    except Exception as e:
        return f"<h2>Error</h2><pre>{e}</pre>"


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)