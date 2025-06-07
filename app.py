from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ✅ Load Model
try:
    with open("student_gpa_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("❌ Model file 'student_gpa_model.pkl' not found. Train the model first!")

# ✅ Load Scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    raise FileNotFoundError("❌ Scaler file 'scaler.pkl' not found. Run preprocessing.py first!")

# ✅ Feature Names
FEATURE_NAMES = [
    "HSC_Score", "SSC_Score", "Attendance", "English_Proficiency", 
    "Daily_Study_Hours", "Study_Efficiency", "Exam_Proficiency", "English_Impact"
]

# ✅ Manual GPA Logic
def manual_gpa_prediction(features):
    hsc, ssc, attendance, english, study_hours, study_eff, exam_prof, eng_impact = features
    gpa = 2.0
    score = 0
    if hsc >= 90: score += 2
    elif hsc >= 75: score += 1.5
    elif hsc >= 60: score += 1
    elif hsc >= 40: score += 0.5
    if ssc >= 90: score += 2
    elif ssc >= 75: score += 1.5
    elif ssc >= 60: score += 1
    elif ssc >= 40: score += 0.5
    if attendance >= 95: score += 1.5
    elif attendance >= 85: score += 1
    elif attendance >= 75: score += 0.5
    if english >= 8: score += 1.5
    elif english >= 5: score += 1
    elif english >= 3: score += 0.5
    if study_hours >= 4: score += 1.5
    elif study_hours >= 2: score += 1
    elif study_hours >= 1: score += 0.5
    if study_eff >= 300: score += 2
    elif study_eff >= 200: score += 1.5
    elif study_eff >= 100: score += 1
    elif study_eff >= 50: score += 0.5
    if exam_prof >= 90: score += 2
    elif exam_prof >= 75: score += 1.5
    elif exam_prof >= 60: score += 1
    elif exam_prof >= 50: score += 0.5
    if eng_impact >= 40: score += 1.5
    elif eng_impact >= 25: score += 1
    elif eng_impact >= 10: score += 0.5
    if score >= 11:
        gpa = 4.0
    elif score >= 9:
        gpa = 3.5
    elif score >= 7:
        gpa = 3.0
    elif score >= 5:
        gpa = 2.5
    elif score >= 3:
        gpa = 2.0
    else:
        gpa = 1.5
    return round(min(max(gpa, 1.0), 4.0), 2)

# 📊 Return pre-generated graphs only (no red overlay)
def generate_user_comparison_graphs(full_features):
    filenames = []
    base_filenames = [
        "hsc_score_box.png",
        "ssc_histogram.png",
        "study_scatter.png",
        "english_violin.png",
        "attendance_strip.png",
        "english_swarm.png",
        "exam_kde.png",
        "gpa_distplot.png",
        "ssc_ecdf.png",
        "study_bar.png"
    ]
    for filename in base_filenames:
        path = f"static/graphs/{filename}"
        if os.path.exists(path):
            filenames.append(filename)
    return filenames

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("features")
        if len(data) != 5:
            return jsonify({"error": f"❌ Incorrect number of features. Expected 5, got {len(data)}"}), 400

        # Extract input values
        hsc, ssc, attendance, english, study_hours = map(float, data)

        # ✅ Derived features
        study_eff = study_hours * attendance
        exam_prof = (hsc + ssc) / 2
        temp_manual_gpa = manual_gpa_prediction([hsc, ssc, attendance, english, study_hours, study_eff, exam_prof, 0])
        eng_impact = english * temp_manual_gpa

        # ✅ Final input vector
        full_features = [hsc, ssc, attendance, english, study_hours, study_eff, exam_prof, eng_impact]

        # ✅ Scaled input
        scaled_input = scaler.transform([full_features])
        predicted_gpa = round(float(model.predict(scaled_input)[0]), 2)
        final_gpa = temp_manual_gpa if temp_manual_gpa is not None else predicted_gpa

        # ✅ Get static comparison graphs
        new_graphs = generate_user_comparison_graphs(full_features)

        response = {
            "prediction": final_gpa,
            "category": categorize_gpa(final_gpa),
            "graphs": new_graphs
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"❌ Prediction error: {str(e)}"}), 500

def categorize_gpa(gpa):
    if gpa >= 3.5:
        return "Excellent"
    elif gpa >= 3.0:
        return "Good"
    elif gpa >= 2.0:
        return "Average"
    else:
        return "Needs Improvement"

if __name__ == "__main__":
    app.run(debug=True)
