"""
ml/predict.py
─────────────
Unified prediction engine:
  • Random Forest (existing .pkl models)
  • ANN / Keras (new .h5 models)
  • Ensemble (weighted average of RF + ANN)
  • SHAP explainability
  • Rule-based recommendations
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

# ── Optional deep learning imports ──────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not installed — ANN predictions unavailable")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed — explainability unavailable")

BASE = Path(__file__).parent.parent / "models"

# ─── Feature metadata ────────────────────────────────────────────────────────

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CANCER_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave pts", "mean symmetry", "mean fractal dim"
]

HEART_FEATURE_DESCRIPTIONS = {
    "age":      "Patient age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol":     "Serum cholesterol (mg/dL)",
    "fbs":      "Fasting blood sugar > 120 (1=Yes)",
    "restecg":  "Resting ECG results (0–2)",
    "thalach":  "Maximum heart rate achieved",
    "exang":    "Exercise-induced angina (1=Yes)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment",
    "ca":       "Major vessels coloured by fluoroscopy",
    "thal":     "Thalassemia type (1=normal, 2=fixed, 3=reversable)",
}


# ─── Model loader ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Lazy-loads and caches all models."""

    def __init__(self):
        self._heart_rf  = None
        self._cancer_rf = None
        self._heart_ann  = None
        self._cancer_ann = None
        self._scaler     = None

    def _load(self, attr, path, loader):
        if getattr(self, attr) is None:
            p = BASE / path
            if p.exists():
                setattr(self, attr, loader(str(p)))
                print(f"[OK] Loaded {path}")
            else:
                print(f"[WARN] Model not found: {p}")
        return getattr(self, attr)

    @property
    def heart_rf(self):
        return self._load("_heart_rf", "heart_model.pkl", joblib.load)

    @property
    def cancer_rf(self):
        return self._load("_cancer_rf", "cancer_model.pkl", joblib.load)

    @property
    def heart_ann(self):
        if not TF_AVAILABLE:
            return None
        return self._load("_heart_ann", "heart_ann.h5",
                          lambda p: tf.keras.models.load_model(p))

    @property
    def cancer_ann(self):
        if not TF_AVAILABLE:
            return None
        return self._load("_cancer_ann", "cancer_ann.h5",
                          lambda p: tf.keras.models.load_model(p))

    @property
    def scaler(self):
        return self._load("_scaler", "scaler.pkl", joblib.load)


registry = ModelRegistry()


# ─── Prediction functions ────────────────────────────────────────────────────

def predict_rf(model, X: np.ndarray) -> dict:
    """Random Forest prediction → risk_pct, confidence, level."""
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    # proba[1] = probability of disease class (class 1)
    risk_pct   = round(float(proba[1]) * 100, 1)
    confidence = round(float(max(proba)) * 100, 1)
    return {"risk_pct": risk_pct, "confidence": confidence, "raw_pred": int(pred)}


def predict_ann(model, X_scaled: np.ndarray) -> dict:
    """ANN/Keras prediction — expects scaled inputs."""
    if model is None:
        return None
    proba = model.predict(X_scaled, verbose=0)[0]
    # Binary: sigmoid output = P(disease)
    risk_pct   = round(float(proba[0]) * 100, 1)
    confidence = round(float(max(proba[0], 1 - proba[0])) * 100, 1)
    return {"risk_pct": risk_pct, "confidence": confidence}


def ensemble(rf_result: dict, ann_result: dict, rf_weight=0.6) -> dict:
    """Weighted average of RF + ANN outputs."""
    if ann_result is None:
        return rf_result
    ann_weight = 1 - rf_weight
    risk = rf_result["risk_pct"] * rf_weight + ann_result["risk_pct"] * ann_weight
    conf = rf_result["confidence"] * rf_weight + ann_result["confidence"] * ann_weight
    return {"risk_pct": round(risk, 1), "confidence": round(conf, 1)}


def risk_level(pct: float, threshold: int = 50) -> str:
    if pct >= threshold:
        return "HIGH RISK"
    if pct >= 35:
        return "MODERATE"
    return "LOW RISK"


# ─── SHAP explainability ─────────────────────────────────────────────────────

def get_shap_values_rf(model, X: np.ndarray, feature_names: list) -> list[dict]:
    """
    Compute SHAP values for a Random Forest prediction.
    Returns list of {feature, importance, shap_value} sorted by |shap|.
    """
    if not SHAP_AVAILABLE:
        # Fallback: return model feature importances
        return [
            {
                "feature":    feature_names[i],
                "importance": round(float(v), 4),
                "shap_value": round(float(v), 4),
                "direction":  "positive"
            }
            for i, v in enumerate(model.feature_importances_)
        ]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # For binary classification, shap_values returns list [class0, class1] or 3D array (samples, features, classes)
    # Use class 1 (disease present) values
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
        vals = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
    else:
        vals = shap_vals[0]

    results = []
    for i, (fname, sv) in enumerate(zip(feature_names, vals)):
        try:
            if hasattr(sv, "__len__") and len(sv) > 1:
                sv_val = float(sv[1])
            elif hasattr(sv, "item"):
                sv_val = float(sv.item())
            else:
                sv_val = float(sv)
        except Exception:
            sv_val = float(np.ravel(sv)[0])

        results.append({
            "feature":    fname,
            "importance": round(abs(sv_val), 4),
            "shap_value": round(sv_val, 4),
            "direction":  "positive" if sv_val > 0 else "negative",
        })

    return sorted(results, key=lambda x: -x["importance"])


# ─── Recommendation engine ───────────────────────────────────────────────────

HEART_RECS = {
    "HIGH RISK": {
        "medicines": [
            ("Atorvastatin",         "20–80 mg daily",  "LDL reduction · plaque stabilisation · start with 20 mg, titrate"),
            ("Aspirin (low dose)",   "75–100 mg daily", "Antiplatelet · prevents clot formation · take with food"),
            ("Ramipril (ACE-I)",     "2.5–10 mg daily", "BP + cardiac protection · check potassium + creatinine"),
            ("Metoprolol succinate", "25–200 mg daily", "Heart rate control · do NOT stop abruptly · taper required"),
            ("Clopidogrel",          "75 mg daily",     "ACS / post-stent antiplatelet · avoid NSAIDs"),
            ("Eplerenone",           "25–50 mg daily",  "Aldosterone antagonist · post-MI with EF <35%"),
        ],
        "diet": [
            "DASH / Mediterranean diet — olive oil, nuts, legumes, fatty fish",
            "Strictly limit saturated fats to <7% of total daily calories",
            "Eliminate trans-fats (fried food, packaged biscuits, margarine)",
            "Increase soluble fibre: oats, psyllium, lentils (target 30–35 g/day)",
            "Reduce sodium to <1,500 mg/day — avoid table salt and processed meats",
            "Omega-3 rich foods: salmon, mackerel, walnuts, flaxseeds (2x/week)",
        ],
        "lifestyle": [
            "150 min/week moderate aerobic exercise (walking, cycling, swimming)",
            "Quit smoking immediately — cardiac risk halves within 1 year",
            "Limit alcohol to ≤1–2 units/day; avoid binge drinking",
            "Achieve and maintain BMI 18.5–24.9 kg/m²",
            "Stress management: yoga, mindfulness, or guided breathing",
            "7–9 hours quality sleep; treat sleep apnoea if present",
        ],
        "tests": [
            "12-lead ECG (resting + stress)",
            "Echocardiogram (LVEF assessment)",
            "Coronary angiography / CT angiogram",
            "Lipid panel (every 3 months on statin)",
            "HbA1c (diabetes screening)",
            "Renal function (urea, creatinine, eGFR)",
            "Thyroid function (TSH)",
            "BNP / NT-proBNP (heart failure marker)",
        ],
        "precautions": [
            "Attend cardiology consultation within 1 week",
            "Do not ignore chest pain, breathlessness, or syncope",
            "Carry emergency Nitroglycerine spray if prescribed",
            "Report any leg swelling or sudden weight gain to doctor",
        ],
    },
    "MODERATE": {
        "medicines": [
            ("Rosuvastatin",          "10–20 mg daily",   "Moderate LDL reduction — review at 3 months"),
            ("Aspirin (preventive)",  "75 mg daily",      "Primary prevention in higher-risk individuals"),
            ("Amlodipine",            "5 mg daily",       "Calcium channel blocker · BP management"),
            ("Vitamin D3 + K2",       "2000 IU D3 daily", "Cardiovascular + bone health · check 25-OH Vit D"),
            ("Omega-3 (fish oil)",    "2–4 g daily",      "TG reduction · anti-inflammatory"),
        ],
        "diet": [
            "Adopt Mediterranean dietary pattern",
            "Reduce dietary cholesterol (<200 mg/day)",
            "Replace refined carbs with whole grains and legumes",
            "Limit red meat to 1–2 servings per week; prefer poultry or fish",
            "Increase potassium-rich foods: bananas, spinach, sweet potato",
        ],
        "lifestyle": [
            "150 min/week structured moderate exercise",
            "Monitor blood pressure at home (target <130/80 mmHg)",
            "Achieve 5–10% body weight reduction if overweight",
            "Reduce occupational and psychosocial stress",
            "Stop smoking and limit alcohol",
        ],
        "tests": [
            "Lipid panel (fasting)",
            "Fasting blood glucose / HbA1c",
            "Resting ECG",
            "Blood pressure monitoring (serial)",
            "Renal and thyroid profile",
        ],
        "precautions": [
            "Schedule cardiology review within 4–6 weeks",
            "Monitor for new symptoms: chest tightness, palpitations",
            "Follow up on all abnormal lab results",
        ],
    },
    "LOW RISK": {
        "medicines": [
            ("No medication required",  "—",             "Continue healthy lifestyle"),
            ("Vitamin D3",              "1000 IU daily", "General cardiac and bone health"),
            ("Omega-3 Fatty Acids",     "1 g daily",     "Prophylactic anti-inflammatory"),
        ],
        "diet": [
            "Maintain a balanced whole-food diet",
            "Limit processed food and added sugars",
            "Ensure adequate fruit and vegetable intake (5 portions/day)",
        ],
        "lifestyle": [
            "Regular physical activity ≥150 min/week",
            "Annual health screening including BP and lipid check",
            "Avoid new smoking or excessive alcohol habits",
        ],
        "tests": ["Annual physical exam","Lipid panel every 5 years","BP check yearly"],
        "precautions": ["Maintain current healthy habits","Annual check-up with GP"],
    },
}

CANCER_RECS = {
    "HIGH RISK": {
        "medicines": [
            ("Tamoxifen",                "20 mg daily",          "ER+ve breast cancer · antiestrogen · 5-year regimen"),
            ("Trastuzumab (Herceptin)", "IV per oncology protocol", "HER2+ve · monoclonal antibody · every 3 weeks"),
            ("Anastrozole",              "1 mg daily",           "Post-menopausal ER+ve · aromatase inhibitor"),
            ("Docetaxel / Paclitaxel",  "Per oncology schedule", "Chemotherapy · taxane class · hair loss side-effect"),
            ("Ondansetron",              "8 mg BD",              "Antiemetic · control chemo-induced nausea/vomiting"),
            ("Zoledronic acid",          "4 mg IV every 4 weeks", "Bone protection during anti-hormonal therapy"),
            ("Filgrastim (G-CSF)",       "Per protocol",         "Prevent febrile neutropenia during chemotherapy"),
        ],
        "diet": [
            "High-protein anti-inflammatory diet (1.2–1.5 g/kg/day protein)",
            "Include cruciferous vegetables (broccoli, cauliflower, kale) daily",
            "Turmeric (curcumin) and green tea — anti-carcinogenic polyphenols",
            "Avoid sugar, refined carbs — cancer cells depend on glucose (Warburg effect)",
            "Maintain caloric intake during chemotherapy to prevent muscle wasting",
            "Berries, pomegranate, flaxseed — antioxidant and anti-oestrogen properties",
        ],
        "lifestyle": [
            "Engage in light physical activity as tolerated during treatment",
            "Psychological counselling / cancer support groups",
            "Avoid ALL tobacco products and carcinogenic chemicals",
            "Minimise alcohol — strongly associated with breast cancer risk",
            "Adequate rest; manage fatigue carefully during treatment cycles",
        ],
        "tests": [
            "Diagnostic mammogram + bilateral breast ultrasound",
            "Core needle biopsy (CNB) for tissue diagnosis",
            "ER / PR / HER2 receptor immunohistochemistry panel",
            "MRI breast (bilateral) for extent of disease",
            "Sentinel lymph node biopsy",
            "CT staging: chest / abdomen / pelvis",
            "Bone scan (if metastasis suspected)",
            "CA 15-3, CEA tumour markers",
            "BRCA1 / BRCA2 genetic testing",
        ],
        "precautions": [
            "Do not delay oncology referral — early treatment is critical",
            "Avoid self-medication or unverified herbal remedies",
            "Genetic counselling for first-degree relatives",
            "Maintain strict hand hygiene during immune-compromised periods",
        ],
    },
    "MODERATE": {
        "medicines": [
            ("Tamoxifen (preventive)", "20 mg daily",  "High-risk pre-menopausal — chemoprevention"),
            ("Raloxifene",             "60 mg daily",  "Post-menopausal risk reduction — less DVT risk than Tamoxifen"),
            ("Exemestane",             "25 mg daily",  "Aromatase inhibitor for elevated-risk post-menopausal"),
            ("Vitamin D3",             "2000 IU daily", "Low Vit D associated with higher breast cancer risk"),
            ("Calcium carbonate",      "1000 mg daily", "Bone health during aromatase inhibitor therapy"),
        ],
        "diet": [
            "Anti-inflammatory Mediterranean diet",
            "Increase fibre intake — reduces circulating oestrogen",
            "Limit processed and red meat",
            "Avoid alcohol or limit to <1 unit/day",
            "Include soy (moderate) — phytoestrogens may be protective in non-ER+ cases",
        ],
        "lifestyle": [
            "150 min/week moderate aerobic exercise — reduces breast cancer risk by 20–40%",
            "Achieve and maintain healthy BMI (adipose tissue = oestrogen source)",
            "Perform monthly breast self-examination",
            "Clinical breast exam every 6 months",
            "Avoid prolonged exogenous hormone use (OCP / HRT) without specialist review",
        ],
        "tests": [
            "Diagnostic mammogram + ultrasound",
            "Fine Needle Aspiration (FNA) if mass palpable",
            "CA 15-3 / CA 27.29 tumour markers",
            "BRCA gene testing if strong family history",
            "Breast MRI if mammogram inconclusive",
        ],
        "precautions": [
            "Specialist breast clinic review within 2 weeks",
            "Report any new lump, skin dimpling, or nipple discharge immediately",
            "Avoid hormone replacement therapy without oncologist clearance",
        ],
    },
    "LOW RISK": {
        "medicines": [
            ("No medication required", "—",             "Maintain healthy lifestyle"),
            ("Vitamin D3",             "1000 IU daily", "General health + cancer risk reduction"),
            ("Omega-3 Fatty Acids",    "1 g daily",     "Anti-inflammatory support"),
        ],
        "diet": [
            "Balanced whole-food diet rich in fruits, vegetables, and whole grains",
            "Limit alcohol and avoid smoking",
            "Maintain healthy body weight",
        ],
        "lifestyle": [
            "Annual mammogram (from age 40)",
            "Monthly breast self-examination",
            "Regular aerobic exercise",
        ],
        "tests": ["Annual mammogram (age ≥40)", "Clinical breast exam yearly"],
        "precautions": [
            "Maintain routine screening schedule",
            "Report any breast changes to GP promptly",
        ],
    },
}


def get_recommendations(disease: str, level: str) -> dict:
    """Return recommendations dict for disease + risk level."""
    recs = HEART_RECS if disease == "heart" else CANCER_RECS
    return recs.get(level, recs["LOW RISK"])


# ─── Main prediction entry point ─────────────────────────────────────────────

def predict_heart(inputs: dict, model_type: str = "ensemble", threshold: int = 50) -> dict:
    """
    inputs: dict with keys matching HEART_FEATURES
    model_type: 'rf' | 'ann' | 'ensemble'
    Returns full clinical result dict.
    """
    X = np.array([[inputs.get(f, 0) for f in HEART_FEATURES]])

    rf_result  = predict_rf(registry.heart_rf, X) if registry.heart_rf else None
    ann_result = None
    if model_type in ("ann", "ensemble") and registry.heart_ann and registry.scaler:
        X_s = registry.scaler.transform(X)
        ann_result = predict_ann(registry.heart_ann, X_s)

    if model_type == "rf" or ann_result is None:
        final = rf_result
    elif model_type == "ann":
        final = ann_result
    else:
        final = ensemble(rf_result, ann_result)

    level = risk_level(final["risk_pct"], threshold)

    # SHAP
    shap_vals = []
    if registry.heart_rf:
        shap_vals = get_shap_values_rf(registry.heart_rf, X, HEART_FEATURES)

    recs = get_recommendations("heart", level)

    return {
        "disease":     "heart",
        "model_type":  model_type,
        "risk_pct":    final["risk_pct"],
        "confidence":  final["confidence"],
        "risk_level":  level,
        "shap_values": shap_vals,
        "recommendations": recs,
        "rf_result":   rf_result,
        "ann_result":  ann_result,
    }


def predict_cancer(inputs: dict, model_type: str = "ensemble", threshold: int = 50) -> dict:
    """
    inputs: dict with keys matching CANCER_FEATURES
    """
    X = np.array([[inputs.get(f, 0) for f in CANCER_FEATURES]])

    rf_result  = predict_rf(registry.cancer_rf, X) if registry.cancer_rf else None
    ann_result = None
    if model_type in ("ann", "ensemble") and registry.cancer_ann and registry.scaler:
        X_s = registry.scaler.transform(X)
        ann_result = predict_ann(registry.cancer_ann, X_s)

    if model_type == "rf" or ann_result is None:
        final = rf_result
    elif model_type == "ann":
        final = ann_result
    else:
        final = ensemble(rf_result, ann_result)

    level = risk_level(final["risk_pct"], threshold)
    shap_vals = []
    if registry.cancer_rf:
        shap_vals = get_shap_values_rf(registry.cancer_rf, X, CANCER_FEATURES)

    recs = get_recommendations("cancer", level)

    return {
        "disease":     "cancer",
        "model_type":  model_type,
        "risk_pct":    final["risk_pct"],
        "confidence":  final["confidence"],
        "risk_level":  level,
        "shap_values": shap_vals,
        "recommendations": recs,
        "rf_result":   rf_result,
        "ann_result":  ann_result,
    }