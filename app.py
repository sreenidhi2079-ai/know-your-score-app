from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Load environment variables (from .env file)
load_dotenv()

# Load model and label encoder
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")

app = Flask(__name__)


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION PAGE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    screen_time = float(request.form["screen_time"])
    phone_checks = int(request.form["phone_checks"])
    hours_without = float(request.form["hours_without"])

    # Convert to model input
    features = np.array([[screen_time, phone_checks, hours_without]])

    # Predict
    prediction = model.predict(features)
    result = le.inverse_transform(prediction)[0]

    # =========================
    # 🎯 AI DYNAMIC SUGGESTIONS LOGIC (GEMINI)
    # =========================
    suggestion = ""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        # Fallback to rule-based if no API key is provided
        if result == "Low":
            suggestion = "👍 Good habits! Keep maintaining balanced phone usage."
        elif result == "Medium":
            suggestion = "⚠ Try reducing screen time and avoid frequent phone checks."
        else:
            suggestion = "🚨 High usage detected! Take a digital detox and limit phone dependency."
    else:
        # Call Gemini REST API
        prompt = (f"The user has a {result} risk of smartphone addiction. "
                  f"They just logged {screen_time} hours of screen time today, {phone_checks} phone checks, "
                  f"and were without their phone for {hours_without} hours. "
                  f"Give them 2 very short, friendly, and practical bullet points of advice to manage their usage. "
                  f"Start each point with a dash (-). Do not write paragraphs.")
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            
            # Request LLM generation
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                response_json = response.json()
                suggestion = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                suggestion = suggestion.replace("*", "") # Clean up common markdown asterisks
            else:
                suggestion = f"⚠️ AI Error ({response.status_code}): {response.text}"
                
        except Exception as e:
            suggestion = f"Oops! The AI could not generate an insight right now."

    return render_template("predictor.html", prediction=result, suggestion=suggestion)


# =========================
# ABOUT PAGE
# =========================
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)