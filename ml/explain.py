"""
ml/explain.py
─────────────
SHAP explainability for MedCore AI.

Supports:
  • Random Forest  → shap.TreeExplainer   (fast, exact)
  • Keras ANN      → shap.DeepExplainer   (deep learning)
  • Fallback       → feature_importances_ (if shap not installed)

Usage:
    from ml.explain import explain_heart, explain_cancer
    shap_result = explain_heart(model, X)
    shap_result = explain_cancer(model, X)
"""

import numpy as np

# ── Feature name lists ────────────────────────────────────────────────────────

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

HEART_FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol":     "Cholesterol",
    "fbs":      "Fasting blood sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Max heart rate",
    "exang":    "Exercise angina",
    "oldpeak":  "ST depression (oldpeak)",
    "slope":    "Slope of ST segment",
    "ca":       "Vessels (fluoroscopy)",
    "thal":     "Thalassemia type",
}

CANCER_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave pts", "mean symmetry", "mean fractal dim"
]

CANCER_FEATURE_LABELS = {
    "mean radius":      "Mean radius",
    "mean texture":     "Mean texture",
    "mean perimeter":   "Mean perimeter",
    "mean area":        "Mean area",
    "mean smoothness":  "Mean smoothness",
    "mean compactness": "Mean compactness",
    "mean concavity":   "Mean concavity",
    "mean concave pts": "Mean concave points",
    "mean symmetry":    "Mean symmetry",
    "mean fractal dim": "Mean fractal dimension",
}

# ── Try importing SHAP ────────────────────────────────────────────────────────

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed — using feature_importances_ fallback")
    print("  Install: pip install shap")


# ── Core explainer functions ──────────────────────────────────────────────────

def _format_shap_output(feature_names: list, shap_values: np.ndarray,
                         label_map: dict) -> list[dict]:
    """
    Convert raw SHAP array into a sorted list of dicts.

    Returns (sorted by absolute importance, descending):
    [
      {
        "feature":    "thal",
        "label":      "Thalassemia type",
        "shap_value": 0.42,
        "importance": 0.42,
        "direction":  "positive",   # pushes risk UP
        "pct":        88.0          # percentage of max, for bar width
      },
      ...
    ]
    """
    results = []
    for fname, sv in zip(feature_names, shap_values):
        results.append({
            "feature":    fname,
            "label":      label_map.get(fname, fname),
            "shap_value": round(float(sv), 4),
            "importance": round(abs(float(sv)), 4),
            "direction":  "positive" if sv >= 0 else "negative",
        })

    results.sort(key=lambda x: -x["importance"])

    # Add percentage of max for rendering bar widths
    max_imp = results[0]["importance"] if results else 1.0
    for r in results:
        r["pct"] = round((r["importance"] / max_imp) * 100, 1)

    return results


def _fallback_importance(model, feature_names: list,
                          label_map: dict) -> list[dict]:
    """
    When SHAP is unavailable, use the model's built-in
    feature_importances_ (Random Forest).
    Values are always positive (no direction info).
    """
    imp = model.feature_importances_
    results = []
    for fname, v in zip(feature_names, imp):
        results.append({
            "feature":    fname,
            "label":      label_map.get(fname, fname),
            "shap_value": round(float(v), 4),
            "importance": round(float(v), 4),
            "direction":  "positive",
        })

    results.sort(key=lambda x: -x["importance"])

    max_imp = results[0]["importance"] if results else 1.0
    for r in results:
        r["pct"] = round((r["importance"] / max_imp) * 100, 1)

    return results


# ── Random Forest explainer ───────────────────────────────────────────────────

def explain_rf(model, X: np.ndarray,
               feature_names: list, label_map: dict) -> list[dict]:
    """
    TreeExplainer for scikit-learn Random Forest.
    X shape: (1, n_features)
    Returns SHAP values for class 1 (disease present).
    """
    if not SHAP_AVAILABLE:
        return _fallback_importance(model, feature_names, label_map)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # Binary RF: shap_values returns [class_0_array, class_1_array]
    # We want class 1 (disease = positive)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        vals = shap_vals[1][0]      # class 1, first (only) sample
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        vals = shap_vals[0]         # single sample
    else:
        vals = shap_vals[0]

    return _format_shap_output(feature_names, vals, label_map)


# ── ANN (Keras) explainer ─────────────────────────────────────────────────────

def explain_ann(model, X_scaled: np.ndarray, X_background: np.ndarray,
                feature_names: list, label_map: dict) -> list[dict]:
    """
    DeepExplainer for Keras ANN.

    X_scaled     : scaled input for this patient (1, n_features)
    X_background : scaled background dataset (subset of training data, ~100 rows)
                   Used to compute baseline expected values.
    """
    if not SHAP_AVAILABLE:
        return []

    try:
        explainer = shap.DeepExplainer(model, X_background)
        shap_vals = explainer.shap_values(X_scaled)

        # DeepExplainer returns list with one array per output neuron
        if isinstance(shap_vals, list):
            vals = shap_vals[0][0]
        else:
            vals = shap_vals[0]

        return _format_shap_output(feature_names, vals, label_map)

    except Exception as e:
        print(f"⚠ DeepExplainer failed: {e} — skipping ANN SHAP")
        return []


# ── Gradient explainer (alternative for ANN) ─────────────────────────────────

def explain_ann_gradient(model, X_scaled: np.ndarray, X_background: np.ndarray,
                          feature_names: list, label_map: dict) -> list[dict]:
    """
    GradientExplainer — faster alternative to DeepExplainer for Keras.
    Use when DeepExplainer is slow or raises errors.
    """
    if not SHAP_AVAILABLE:
        return []

    try:
        explainer = shap.GradientExplainer(model, X_background)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            vals = shap_vals[0][0]
        else:
            vals = shap_vals[0]
        return _format_shap_output(feature_names, vals, label_map)
    except Exception as e:
        print(f"⚠ GradientExplainer failed: {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def explain_heart(rf_model, X: np.ndarray,
                  ann_model=None, X_scaled: np.ndarray = None,
                  X_background: np.ndarray = None) -> dict:
    """
    Generate SHAP explanations for a heart disease prediction.

    Args:
        rf_model    : loaded RandomForestClassifier
        X           : raw (unscaled) input, shape (1, 13)
        ann_model   : optional Keras model
        X_scaled    : scaled version of X for ANN
        X_background: scaled background data for DeepExplainer

    Returns:
        {
          "rf_shap":     [...],   # SHAP from Random Forest
          "ann_shap":    [...],   # SHAP from ANN (empty if unavailable)
          "summary":     [...],   # rf_shap (primary display list)
          "top_driver":  "thal",  # single most influential feature
          "explanation": "str"    # plain-English explanation
        }
    """
    rf_shap  = explain_rf(rf_model, X, HEART_FEATURES, HEART_FEATURE_LABELS)
    ann_shap = []

    if ann_model is not None and X_scaled is not None and X_background is not None:
        ann_shap = explain_ann(ann_model, X_scaled, X_background,
                               HEART_FEATURES, HEART_FEATURE_LABELS)

    top = rf_shap[0] if rf_shap else {}
    explanation = _build_explanation(rf_shap, disease="heart disease")

    return {
        "rf_shap":     rf_shap,
        "ann_shap":    ann_shap,
        "summary":     rf_shap,          # use RF SHAP as primary display
        "top_driver":  top.get("label", "—"),
        "explanation": explanation,
    }


def explain_cancer(rf_model, X: np.ndarray,
                   ann_model=None, X_scaled: np.ndarray = None,
                   X_background: np.ndarray = None) -> dict:
    """
    Generate SHAP explanations for a cancer risk prediction.
    Same signature as explain_heart.
    """
    rf_shap  = explain_rf(rf_model, X, CANCER_FEATURES, CANCER_FEATURE_LABELS)
    ann_shap = []

    if ann_model is not None and X_scaled is not None and X_background is not None:
        ann_shap = explain_ann(ann_model, X_scaled, X_background,
                               CANCER_FEATURES, CANCER_FEATURE_LABELS)

    top = rf_shap[0] if rf_shap else {}
    explanation = _build_explanation(rf_shap, disease="cancer risk")

    return {
        "rf_shap":     rf_shap,
        "ann_shap":    ann_shap,
        "summary":     rf_shap,
        "top_driver":  top.get("label", "—"),
        "explanation": explanation,
    }


# ── Plain-English explanation builder ────────────────────────────────────────

def _build_explanation(shap_list: list[dict], disease: str) -> str:
    """
    Build a 2-sentence plain-English explanation from SHAP values.
    Example:
      "The strongest driver of this heart disease prediction is
       Thalassemia type (+0.42), pushing risk higher. Cholesterol
       (+0.15) and ST depression (+0.28) also contributed significantly."
    """
    if not shap_list:
        return "SHAP explanation unavailable."

    positive = [s for s in shap_list if s["direction"] == "positive"]
    negative = [s for s in shap_list if s["direction"] == "negative"]

    top_pos = positive[:2] if positive else []
    top_neg = negative[:1] if negative else []

    if not top_pos:
        return f"No dominant risk drivers found for {disease}."

    top1 = top_pos[0]
    sentence1 = (
        f"The strongest driver of this {disease} prediction is "
        f"{top1['label']} ({top1['shap_value']:+.2f}), "
        f"pushing risk {'higher' if top1['direction']=='positive' else 'lower'}."
    )

    parts = []
    if len(top_pos) > 1:
        t = top_pos[1]
        parts.append(f"{t['label']} ({t['shap_value']:+.2f})")
    if top_neg:
        t = top_neg[0]
        parts.append(f"{t['label']} ({t['shap_value']:+.2f}, protective)")

    sentence2 = ""
    if parts:
        sentence2 = " Also notable: " + ", ".join(parts) + "."

    return sentence1 + sentence2


# ── Streamlit rendering helper ────────────────────────────────────────────────

def render_shap_bars_streamlit(shap_list: list[dict], top_n: int = 8):
    """
    Render SHAP bars directly in a Streamlit app.
    Call inside a `with st.container():` block.

    Usage:
        import streamlit as st
        from ml.explain import explain_heart, render_shap_bars_streamlit
        result = explain_heart(heart_rf, X)
        render_shap_bars_streamlit(result["summary"])
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available")
        return

    for item in shap_list[:top_n]:
        color = "#dc2626" if item["direction"] == "positive" else "#059669"
        arrow = "↑ risk" if item["direction"] == "positive" else "↓ risk"

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">
            <span style="font-size:12px;color:#64748b;width:180px;
                  flex-shrink:0;font-weight:500">{item['label']}</span>
            <div style="flex:1;height:8px;background:#f1f5f9;
                  border-radius:4px;overflow:hidden">
                <div style="width:{item['pct']}%;height:100%;
                      background:{color};border-radius:4px;
                      transition:width 0.5s ease"></div>
            </div>
            <span style="font-size:11px;color:{color};font-weight:700;
                  width:90px;text-align:right;flex-shrink:0">
                {item['shap_value']:+.3f} {arrow}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib, os

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "heart_model.pkl")

    if not os.path.exists(model_path):
        print("❌ heart_model.pkl not found — run train_model.py first")
    else:
        model = joblib.load(model_path)

        # Sample patient input (age=55, sex=1, cp=1, ...)
        X = np.array([[55, 1, 1, 130, 245, 0, 0, 148, 0, 1.2, 1, 0, 2]])

        result = explain_heart(model, X)

        print("\n🧠 SHAP Explainability — Heart Disease")
        print("=" * 50)
        for item in result["summary"]:
            bar = "█" * int(item["pct"] / 5)
            sign = "+" if item["direction"] == "positive" else "-"
            print(f"  {item['label']:<30} {bar:<20} {sign}{item['importance']:.4f}")

        print(f"\n📌 Top driver : {result['top_driver']}")
        print(f"💬 Explanation: {result['explanation']}")