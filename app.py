"""
app.py
──────
Flask REST API — MedCore AI backend.

Endpoints:
  POST /api/predict/heart       → heart disease prediction
  POST /api/predict/cancer      → cancer risk prediction
  POST /api/ocr                 → OCR report analysis
  POST /api/patient             → create/update patient
  GET  /api/patient/<id>        → get patient + history
  GET  /api/stats               → dashboard statistics
  GET  /api/health              → health check

Run:
  python app.py
  # API runs on http://localhost:5000
"""

import os, sys, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Path setup so sub-modules can import each other ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from ml.predict import predict_heart, predict_cancer, HEART_FEATURES, CANCER_FEATURES
from database.db import (
    init_db, upsert_patient, get_patient, get_all_patients,
    save_prediction, get_patient_predictions, get_all_predictions,
    save_ocr_report, get_ocr_reports, save_recommendations, get_stats,
)
from ocr.extractor import process_pasted_text, process_report

# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)    # Allow Streamlit (different port) to call this API

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialise DB on startup
init_db()


# ─── Utility ─────────────────────────────────────────────────────────────────

def success(data: dict, code: int = 200):
    return jsonify({"status": "ok", **data}), code

def error(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code


# ─── Health check ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return success({"message": "MedCore AI API is running", "version": "3.0"})


# ─── Statistics ──────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    return success({"stats": get_stats()})


# ─── Patient endpoints ───────────────────────────────────────────────────────

@app.route("/api/patient", methods=["POST"])
def create_patient():
    data = request.get_json()
    try:
        upsert_patient(
            patient_id=data["patient_id"],
            name=data["name"],
            age=int(data.get("age", 0)),
            sex=data.get("sex", "M"),
            ward=data.get("ward", "General"),
        )
        return success({"message": "Patient saved", "patient_id": data["patient_id"]})
    except KeyError as e:
        return error(f"Missing field: {e}")


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_api(patient_id):
    pt = get_patient(patient_id)
    if not pt:
        return error("Patient not found", 404)
    preds  = get_patient_predictions(patient_id)
    reports = get_ocr_reports(patient_id)
    return success({"patient": pt, "predictions": preds, "ocr_reports": reports})


@app.route("/api/patients", methods=["GET"])
def list_patients():
    return success({"patients": get_all_patients()})


# ─── Prediction endpoints ─────────────────────────────────────────────────────

@app.route("/api/predict/heart", methods=["POST"])
def predict_heart_api():
    """
    Body (JSON):
    {
      "patient_id": "MED-0001",     (optional)
      "model_type": "ensemble",      (optional: rf|ann|ensemble)
      "threshold":  50,              (optional: int 30-80)
      "inputs": {
        "age": 55, "sex": 1, "cp": 1, "trestbps": 130,
        "chol": 245, "fbs": 0, "restecg": 0, "thalach": 148,
        "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
      }
    }
    """
    body = request.get_json()
    inputs     = body.get("inputs", {})
    model_type = body.get("model_type", "ensemble")
    threshold  = int(body.get("threshold", 50))
    patient_id = body.get("patient_id")

    # Validate all required features are present
    missing = [f for f in HEART_FEATURES if f not in inputs]
    if missing:
        return error(f"Missing input features: {missing}")

    try:
        result = predict_heart(inputs, model_type=model_type, threshold=threshold)
    except Exception as e:
        return error(f"Prediction failed: {str(e)}", 500)

    # Persist to DB
    if patient_id:
        pred_id = save_prediction(
            patient_id=patient_id,
            disease="heart",
            model_type=model_type,
            risk_pct=result["risk_pct"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            input_features=inputs,
            shap_values={s["feature"]: s["shap_value"] for s in result["shap_values"]},
        )
        recs = result["recommendations"]
        rec_rows = []
        for cat, items in [("diet", recs.get("diet",[])), ("lifestyle", recs.get("lifestyle",[])), ("tests", recs.get("tests",[]))]:
            for item in items:
                rec_rows.append({"category": cat, "content": item})
        save_recommendations(pred_id, rec_rows)
        result["prediction_id"] = pred_id

    return success(result)


@app.route("/api/predict/cancer", methods=["POST"])
def predict_cancer_api():
    """
    Body (JSON):
    {
      "patient_id": "MED-0001",    (optional)
      "model_type": "ensemble",
      "threshold":  50,
      "inputs": {
        "mean radius": 14.1, "mean texture": 19.2, ...
      }
    }
    """
    body = request.get_json()
    inputs     = body.get("inputs", {})
    model_type = body.get("model_type", "ensemble")
    threshold  = int(body.get("threshold", 50))
    patient_id = body.get("patient_id")

    missing = [f for f in CANCER_FEATURES if f not in inputs]
    if missing:
        return error(f"Missing input features: {missing}")

    try:
        result = predict_cancer(inputs, model_type=model_type, threshold=threshold)
    except Exception as e:
        return error(f"Prediction failed: {str(e)}", 500)

    if patient_id:
        pred_id = save_prediction(
            patient_id=patient_id,
            disease="cancer",
            model_type=model_type,
            risk_pct=result["risk_pct"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            input_features=inputs,
            shap_values={s["feature"]: s["shap_value"] for s in result["shap_values"]},
        )
        result["prediction_id"] = pred_id

    return success(result)


# ─── OCR endpoint ─────────────────────────────────────────────────────────────

@app.route("/api/ocr/text", methods=["POST"])
def ocr_from_text():
    """Parse pasted report text via NLP."""
    body = request.get_json()
    text       = body.get("text", "").strip()
    patient_id = body.get("patient_id")

    if not text:
        return error("No text provided")

    result = process_pasted_text(text)

    if patient_id:
        save_ocr_report(patient_id, "pasted_text", text, result["parsed_values"])

    return success({
        "parsed_values":  result["parsed_values"],
        "heart_features": result["heart_features"],
        "entities":       result["entities"],
        "raw_text":       text[:2000],  # truncate for response size
    })


@app.route("/api/ocr/upload", methods=["POST"])
def ocr_from_file():
    """Upload a file (PDF/image) and run OCR + NLP."""
    if "file" not in request.files:
        return error("No file uploaded")

    f   = request.files["file"]
    eng = request.form.get("engine", "easyocr")
    patient_id = request.form.get("patient_id")

    if f.filename == "":
        return error("Empty filename")

    # Save temp file
    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    try:
        result = process_report(save_path, engine=eng)
    except Exception as e:
        return error(f"OCR failed: {str(e)}", 500)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    if patient_id:
        save_ocr_report(patient_id, f.filename, result["raw_text"], result["parsed_values"])

    return success({
        "parsed_values":  result["parsed_values"],
        "heart_features": result["heart_features"],
        "entities":       result["entities"],
        "raw_text":       result["raw_text"][:2000],
    })


# ─── Predictions history ──────────────────────────────────────────────────────

@app.route("/api/predictions", methods=["GET"])
def all_predictions():
    return success({"predictions": get_all_predictions()})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏥 MedCore AI Flask API — starting on http://localhost:5000")
    print("   Endpoints: /api/health | /api/predict/heart | /api/predict/cancer")
    print("              /api/ocr/text | /api/ocr/upload  | /api/stats")
    app.run(debug=True, host="0.0.0.0", port=5000)