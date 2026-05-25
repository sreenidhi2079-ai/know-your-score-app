from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Load environment variables (from .env file)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and label encoder using absolute paths (avoids cwd issues on deploy)
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
le_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
try:
    model = joblib.load(model_path)
    le = joblib.load(le_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {e}")

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
    # 🎯 INSIGHTS LOGIC (AI First, Fallback to Rules)
    # =========================
    
    # 1. Define Rule-based Suggestion (Baseline / Fallback)
    if result == "Low":
        suggestion = "👍 Good habits! Keep maintaining balanced phone usage."
    elif result == "Medium":
        suggestion = "⚠ Try reducing screen time and avoid frequent phone checks."
    else:
        suggestion = "🚨 High usage detected! Take a digital detox and limit phone dependency."

    # 2. Try AI Dynamic Suggestions (Highest Priority)
    # Check both common naming versions just in case
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
    
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        print(f"DEBUG: Attempting AI call. Key found (starts with: {api_key[:5]}...)")
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
                if "candidates" in response_json and response_json["candidates"]:
                    ai_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if ai_text:
                        suggestion = ai_text.replace("*", "")
                else:
                    suggestion = f"⚠️ AI Error: No candidates. Response: {response.text}"
            else:
                suggestion = f"⚠️ AI API Error ({response.status_code}): {response.text}"
                
        except Exception as e:
            suggestion = f"⚠️ AI Exception: {str(e)}"
    else:
        suggestion = "⚠️ DEBUG: No API Key found in Environment Variables!"

    return render_template("predictor.html", prediction=result, suggestion=suggestion)


# =========================
# ABOUT PAGE
# =========================
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # Use PORT from environment (Render sets $PORT) and bind to 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)