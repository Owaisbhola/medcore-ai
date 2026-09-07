"""
ml/recommend.py
───────────────
Rule-based personalized recommendation engine for MedCore AI.

Given a disease type + risk level + (optional) patient biomarkers,
returns structured recommendations across 5 categories:
  • medicines   — drug name, dose, purpose
  • diet        — food and nutrition advice
  • lifestyle   — exercise, habits, sleep
  • tests       — diagnostic investigations to order
  • precautions — warnings and follow-up actions

Usage:
    from ml.recommend import get_recommendations, get_biomarker_flags

    recs  = get_recommendations("heart", "HIGH RISK")
    flags = get_biomarker_flags({"chol": 280, "trestbps": 145})
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

HEART_RECOMMENDATIONS = {

    "HIGH RISK": {
        "alert_short": "Cardiology referral urgently",
        "alert_full":  (
            "This patient has a high predicted probability of heart disease. "
            "Immediate clinical evaluation and cardiology referral is advised. "
            "Do not delay investigation or treatment."
        ),
        "medicines": [
            ("Atorvastatin",          "20–80 mg at night",    "High-intensity statin · LDL reduction · plaque stabilisation"),
            ("Aspirin (low dose)",    "75–100 mg daily",      "Antiplatelet · prevents arterial clot formation · take with food"),
            ("Ramipril (ACE-I)",      "2.5–10 mg daily",      "BP + cardiac protection · check K⁺ & creatinine at 2 weeks"),
            ("Metoprolol succinate",  "25–200 mg daily",      "Beta-blocker · heart rate control · do NOT stop abruptly"),
            ("Clopidogrel",           "75 mg daily",          "Dual antiplatelet (post-ACS/stent) · avoid NSAIDs"),
            ("Eplerenone",            "25–50 mg daily",       "Aldosterone antagonist · post-MI with EF < 35%"),
            ("Isosorbide mononitrate","30–60 mg daily",       "Long-acting nitrate · angina relief · avoid in hypotension"),
        ],
        "diet": [
            "Follow DASH or Mediterranean diet — olive oil, nuts, legumes, fatty fish",
            "Strictly limit saturated fats to < 7% of total daily calories",
            "Eliminate trans-fats — avoid fried food, packaged biscuits, margarine",
            "Increase soluble fibre: oats, psyllium husk, lentils (target 30–35 g/day)",
            "Reduce sodium to < 1,500 mg/day — avoid table salt & processed meats",
            "Omega-3 sources: salmon, mackerel, walnuts, flaxseeds (2× per week)",
            "Include plant sterols (2 g/day) — fortified spreads or supplements",
            "Avoid sugar-sweetened beverages and refined carbohydrates",
        ],
        "lifestyle": [
            "150 min/week moderate aerobic exercise (walking, cycling, swimming)",
            "Quit smoking immediately — cardiac risk halves within 1 year of cessation",
            "Limit alcohol to ≤ 1–2 units/day; avoid binge drinking",
            "Achieve and maintain BMI 18.5–24.9 kg/m²",
            "Stress management: yoga, mindfulness, guided breathing (10 min/day)",
            "7–9 hours quality sleep per night; screen and treat sleep apnoea",
            "Monitor BP at home twice daily (target < 130/80 mmHg)",
        ],
        "tests": [
            "12-lead resting ECG",
            "Exercise stress test (treadmill / Bruce protocol)",
            "Echocardiogram (LVEF, wall motion, valves)",
            "Coronary CT angiogram or invasive angiography",
            "Fasting lipid panel (LDL, HDL, TG, total cholesterol)",
            "HbA1c + fasting glucose (diabetes screening)",
            "Renal function panel (urea, creatinine, eGFR)",
            "Thyroid function test (TSH)",
            "BNP / NT-proBNP (heart failure marker)",
            "Troponin I/T (if acute coronary syndrome suspected)",
        ],
        "precautions": [
            "Attend cardiology outpatient within 1 week",
            "Do NOT ignore chest pain, breathlessness, or syncope — go to ED",
            "Carry sublingual Nitroglycerine spray if prescribed",
            "Report any leg swelling, sudden weight gain (> 2 kg/day), or palpitations",
            "Avoid strenuous unmonitored exercise until cleared by cardiologist",
            "Review all current medications for interactions (NSAIDs, decongestants)",
        ],
    },

    "MODERATE": {
        "alert_short": "Cardiology review in 4–6 weeks",
        "alert_full":  (
            "Moderate cardiovascular risk detected. Lifestyle modification "
            "and pharmacological management should be considered. "
            "Schedule specialist review."
        ),
        "medicines": [
            ("Rosuvastatin",          "10–20 mg daily",       "Moderate-intensity statin · review LDL at 3 months"),
            ("Aspirin (preventive)",  "75 mg daily",          "Primary prevention in high-moderate risk patients"),
            ("Amlodipine",            "5–10 mg daily",        "Calcium channel blocker · BP management · ankle oedema SE"),
            ("Omega-3 (fish oil)",    "2–4 g daily",          "TG reduction · anti-inflammatory · EPA + DHA"),
            ("Vitamin D3 + K2",       "2000 IU D3 daily",     "Cardiovascular + bone health · check 25-OH Vit D level"),
        ],
        "diet": [
            "Adopt Mediterranean dietary pattern as daily habit",
            "Reduce dietary cholesterol to < 200 mg/day",
            "Replace refined carbohydrates with whole grains and legumes",
            "Limit red meat to 1–2 servings/week; prefer poultry or fish",
            "Increase potassium-rich foods: bananas, spinach, sweet potato",
            "Avoid processed food with hidden sodium (soups, sauces, bread)",
        ],
        "lifestyle": [
            "Build up to 150 min/week of structured moderate exercise",
            "Monitor BP at home (target < 130/80 mmHg)",
            "Achieve 5–10% body weight reduction if overweight",
            "Reduce occupational and psychosocial stress",
            "Stop smoking and limit alcohol",
            "Consider a cardiac rehabilitation programme",
        ],
        "tests": [
            "Fasting lipid panel",
            "Fasting blood glucose / HbA1c",
            "Resting 12-lead ECG",
            "Serial blood pressure measurement (home + clinic)",
            "Renal and thyroid function profile",
            "Urine albumin-to-creatinine ratio (early renal damage marker)",
        ],
        "precautions": [
            "Schedule cardiology review within 4–6 weeks",
            "Monitor for new symptoms: chest tightness, palpitations, dyspnoea",
            "Follow up all abnormal lab results promptly",
            "Reassess risk in 3 months after lifestyle changes",
        ],
    },

    "LOW RISK": {
        "alert_short": "Routine annual surveillance",
        "alert_full":  (
            "Low cardiovascular risk at this time. "
            "Maintain a healthy lifestyle and attend routine annual screening."
        ),
        "medicines": [
            ("No medication required", "—",              "Maintain healthy lifestyle habits"),
            ("Vitamin D3",             "1000 IU daily",  "General cardiac and bone health"),
            ("Omega-3 Fatty Acids",    "1 g daily",      "Prophylactic anti-inflammatory support"),
        ],
        "diet": [
            "Maintain a balanced whole-food diet",
            "Limit processed food and added sugars",
            "Ensure 5 portions of fruit and vegetables daily",
            "Stay adequately hydrated (1.5–2 L water/day)",
        ],
        "lifestyle": [
            "Regular physical activity ≥ 150 min/week",
            "Annual health screening including BP and lipid check",
            "Avoid starting smoking or excessive alcohol",
            "Maintain healthy weight (BMI < 25)",
        ],
        "tests": [
            "Annual physical examination with GP",
            "Lipid panel every 5 years (more often if family history)",
            "Blood pressure check yearly",
        ],
        "precautions": [
            "Continue current healthy habits",
            "Annual GP check-up",
            "Seek advice early if new cardiac symptoms arise",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CANCER RISK RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

CANCER_RECOMMENDATIONS = {

    "HIGH RISK": {
        "alert_short": "Oncology referral urgently",
        "alert_full":  (
            "High malignancy risk predicted. Immediate oncology referral and "
            "comprehensive diagnostic workup is required. Early intervention "
            "significantly improves outcomes."
        ),
        "medicines": [
            ("Tamoxifen",                "20 mg daily",              "ER+ve · antiestrogen · 5-year regimen standard"),
            ("Trastuzumab (Herceptin)", "IV per oncology protocol",  "HER2+ve · monoclonal antibody · every 3 weeks"),
            ("Anastrozole",              "1 mg daily",               "Post-menopausal ER+ve · aromatase inhibitor"),
            ("Docetaxel / Paclitaxel",  "Per oncology schedule",    "Taxane chemotherapy · hair loss & neuropathy SE"),
            ("Ondansetron",              "8 mg BD",                  "Antiemetic · chemo-induced nausea & vomiting"),
            ("Zoledronic acid",          "4 mg IV every 4 weeks",    "Bone protection during anti-hormonal therapy"),
            ("Filgrastim (G-CSF)",       "Per protocol",             "Prevent febrile neutropenia during chemotherapy"),
            ("Dexamethasone",            "4–8 mg BD",                "Pre-chemo steroid · reduces hypersensitivity"),
        ],
        "diet": [
            "High-protein anti-inflammatory diet (1.2–1.5 g protein/kg/day)",
            "Cruciferous vegetables daily: broccoli, cauliflower, kale, Brussels sprouts",
            "Turmeric (curcumin) and green tea — polyphenols with anti-carcinogenic properties",
            "Avoid sugar and refined carbohydrates — cancer cells rely on glucose (Warburg effect)",
            "Maintain adequate caloric intake during chemotherapy to prevent muscle wasting",
            "Berries, pomegranate, flaxseed — antioxidant and anti-oestrogenic properties",
            "Avoid alcohol completely — strongly associated with breast cancer risk",
            "Probiotic-rich foods (yoghurt, kefir) to support gut microbiome during treatment",
        ],
        "lifestyle": [
            "Light physical activity as tolerated during treatment (walking, gentle yoga)",
            "Psychological counselling and cancer support group participation",
            "Avoid ALL tobacco products and carcinogenic chemical exposure",
            "Minimise alcohol — strongly linked to breast and other cancer risk",
            "Adequate rest; carefully manage fatigue during chemotherapy cycles",
            "Use sun protection (SPF 50+) — treatment increases photosensitivity",
        ],
        "tests": [
            "Diagnostic mammogram + bilateral breast ultrasound",
            "Core needle biopsy (CNB) — tissue histology and grading",
            "ER / PR / HER2 receptor immunohistochemistry panel",
            "MRI breast bilateral — extent of disease and multifocality",
            "Sentinel lymph node biopsy",
            "CT staging: chest / abdomen / pelvis",
            "Bone scan (if bone metastasis suspected)",
            "CA 15-3, CEA, CA 125 tumour markers",
            "BRCA1 / BRCA2 genetic testing",
            "Full blood count, liver function, renal function before chemo",
        ],
        "precautions": [
            "Do NOT delay oncology referral — early treatment is critical",
            "Avoid self-medication or unverified herbal remedies during treatment",
            "Genetic counselling recommended for first-degree relatives",
            "Maintain strict hand hygiene during immune-compromised periods (low neutrophils)",
            "Report fever > 38°C immediately during chemotherapy (possible neutropenic sepsis)",
        ],
    },

    "MODERATE": {
        "alert_short": "Specialist review in 2 weeks",
        "alert_full":  (
            "Moderate cancer risk detected. Specialist breast clinic review and "
            "enhanced imaging are recommended. Chemoprevention may be considered."
        ),
        "medicines": [
            ("Tamoxifen (preventive)", "20 mg daily",   "High-risk pre-menopausal chemoprevention"),
            ("Raloxifene",             "60 mg daily",   "Post-menopausal risk reduction · less DVT risk than Tamoxifen"),
            ("Exemestane",             "25 mg daily",   "Aromatase inhibitor · elevated-risk post-menopausal women"),
            ("Vitamin D3",             "2000 IU daily", "Low Vit D associated with higher breast cancer risk"),
            ("Calcium carbonate",      "1000 mg daily", "Bone health during aromatase inhibitor therapy"),
        ],
        "diet": [
            "Anti-inflammatory Mediterranean diet as daily standard",
            "Increase dietary fibre — reduces circulating oestrogen levels",
            "Limit processed and red meat consumption",
            "Avoid alcohol or limit strictly to < 1 unit/day",
            "Moderate soy consumption — phytoestrogens may be protective in non-ER+ context",
            "Include selenium-rich foods: Brazil nuts, sunflower seeds, fish",
        ],
        "lifestyle": [
            "150 min/week moderate aerobic exercise — reduces breast cancer risk by 20–40%",
            "Achieve and maintain healthy BMI — adipose tissue is an oestrogen source",
            "Perform monthly breast self-examination",
            "Clinical breast examination every 6 months",
            "Avoid prolonged exogenous hormone use (OCP, HRT) without specialist review",
            "Minimise radiation exposure (medical imaging) unless clinically necessary",
        ],
        "tests": [
            "Diagnostic mammogram + ultrasound",
            "Fine Needle Aspiration (FNA) if a mass is palpable",
            "CA 15-3 / CA 27.29 tumour markers",
            "BRCA gene testing if strong family history (1st degree)",
            "Breast MRI if mammogram is inconclusive or dense tissue",
            "Oestrogen receptor status (if FNA performed)",
        ],
        "precautions": [
            "Specialist breast clinic review within 2 weeks",
            "Report any new lump, skin dimpling, or nipple discharge immediately",
            "Avoid hormone replacement therapy without oncologist clearance",
            "Reassess risk every 12 months",
        ],
    },

    "LOW RISK": {
        "alert_short": "Routine screening schedule",
        "alert_full":  (
            "Low cancer risk at this time. "
            "Maintain a healthy lifestyle and follow routine cancer screening guidelines."
        ),
        "medicines": [
            ("No medication required", "—",             "Maintain healthy lifestyle"),
            ("Vitamin D3",             "1000 IU daily", "General health + cancer risk reduction"),
            ("Omega-3 Fatty Acids",    "1 g daily",     "Anti-inflammatory maintenance"),
        ],
        "diet": [
            "Balanced whole-food diet rich in fruits, vegetables, whole grains",
            "Limit alcohol and avoid smoking",
            "Maintain healthy body weight",
            "Include antioxidant-rich foods (berries, leafy greens, tomatoes)",
        ],
        "lifestyle": [
            "Annual mammogram from age 40 (or 25 if strong family history)",
            "Monthly breast self-examination",
            "Regular aerobic exercise (150 min/week)",
            "Maintain a non-smoking, low-alcohol lifestyle",
        ],
        "tests": [
            "Annual mammogram (from age 40)",
            "Clinical breast examination yearly",
            "Monthly self-breast examination",
        ],
        "precautions": [
            "Maintain routine screening schedule",
            "Report any breast changes to GP promptly",
            "Consider genetic testing if 2+ first-degree relatives affected",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_recommendations(disease: str, risk_level: str) -> dict:
    """
    Main entry point.

    Args:
        disease    : "heart" or "cancer"
        risk_level : "HIGH RISK" | "MODERATE" | "LOW RISK"

    Returns:
        Full recommendations dict with keys:
          alert_short, alert_full, medicines, diet,
          lifestyle, tests, precautions
    """
    disease = disease.lower().strip()
    risk_level = risk_level.upper().strip()

    if disease == "heart":
        recs = HEART_RECOMMENDATIONS
    elif disease in ("cancer", "breast", "breast cancer"):
        recs = CANCER_RECOMMENDATIONS
    else:
        raise ValueError(f"Unknown disease: '{disease}'. Use 'heart' or 'cancer'.")

    if risk_level not in recs:
        # Fallback to LOW RISK if unknown level
        print(f"⚠ Unknown risk level '{risk_level}' — defaulting to LOW RISK")
        risk_level = "LOW RISK"

    return recs[risk_level]


def get_biomarker_flags(biomarkers: dict) -> list[dict]:
    """
    Flag individual biomarker values against clinical normal ranges.

    Args:
        biomarkers: dict of {parameter: value}
        e.g. {"chol": 280, "trestbps": 145, "thalach": 58}

    Returns:
        List of flagged parameters:
        [
          {"parameter": "Cholesterol", "value": 280,
           "unit": "mg/dL", "flag": "High",
           "normal": "< 200 mg/dL", "action": "..."},
          ...
        ]
    """
    RULES = {
        "chol": {
            "label":  "Cholesterol",
            "unit":   "mg/dL",
            "normal": "< 200 mg/dL",
            "flag": lambda v: (
                "Critical" if v > 300 else
                "High"     if v > 239 else
                "Borderline" if v > 199 else "Normal"
            ),
            "action": {
                "Critical":   "Urgent lipid clinic referral. Start high-intensity statin.",
                "High":       "Start statin therapy. Dietary modification. 3-month review.",
                "Borderline": "Dietary changes. Recheck in 6 months.",
                "Normal":     "Continue healthy diet.",
            },
        },
        "trestbps": {
            "label":  "Resting Blood Pressure",
            "unit":   "mmHg",
            "normal": "< 120 mmHg",
            "flag": lambda v: (
                "Critical" if v > 180 else
                "High"     if v > 139 else
                "Elevated" if v > 119 else "Normal"
            ),
            "action": {
                "Critical": "Hypertensive emergency — immediate intervention.",
                "High":     "Antihypertensive therapy required. Monitor daily.",
                "Elevated": "Lifestyle modifications. Recheck in 3 months.",
                "Normal":   "Continue monitoring annually.",
            },
        },
        "thalach": {
            "label":  "Maximum Heart Rate",
            "unit":   "bpm",
            "normal": "60–100 bpm (resting)",
            "flag": lambda v: (
                "High (Tachycardia)"  if v > 100 else
                "Low (Bradycardia)"   if v < 60  else "Normal"
            ),
            "action": {
                "High (Tachycardia)": "ECG and Holter monitoring recommended.",
                "Low (Bradycardia)":  "Cardiology review. Check medications for rate-slowing agents.",
                "Normal":             "Within normal limits.",
            },
        },
        "oldpeak": {
            "label":  "ST Depression (Oldpeak)",
            "unit":   "mm",
            "normal": "< 1.0 mm",
            "flag": lambda v: (
                "High" if v > 2.0 else
                "Borderline" if v >= 1.0 else "Normal"
            ),
            "action": {
                "High":       "Significant ST depression — stress ECG and cardiology referral.",
                "Borderline": "Borderline ST changes — monitor and retest.",
                "Normal":     "No significant ST changes.",
            },
        },
    }

    results = []
    for key, value in biomarkers.items():
        if key in RULES:
            rule = RULES[key]
            try:
                flag = rule["flag"](float(value))
                results.append({
                    "parameter": rule["label"],
                    "value":     value,
                    "unit":      rule["unit"],
                    "normal":    rule["normal"],
                    "flag":      flag,
                    "action":    rule["action"].get(flag, "Review with clinician."),
                })
            except (TypeError, ValueError):
                continue

    return results


def format_medicines_html(medicines: list[tuple]) -> str:
    """
    Convert medicines list → HTML table string for Streamlit.

    Args:
        medicines: list of (name, dose, purpose) tuples

    Returns:
        HTML string with <table class="med-table">...</table>
    """
    rows = "".join([
        f"""<tr>
            <td><div class="med-name">{m[0]}</div>
                <div class="med-dose">{m[1]}</div></td>
            <td><div class="med-purpose">{m[2]}</div></td>
        </tr>"""
        for m in medicines
    ])
    return f"""
    <table class="med-table">
        <thead>
            <tr><th>Drug &amp; Dose</th><th>Purpose / Notes</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def format_checklist_html(items: list[str], dot_class: str = "check-dot-blue") -> str:
    """
    Convert a list of strings → HTML checklist for Streamlit.

    Args:
        items      : list of text items
        dot_class  : "check-dot-red" | "check-dot-amber" |
                     "check-dot-blue" | "check-dot-green"

    Returns:
        HTML string with <div class="check-item">... repeated
    """
    return "".join([
        f'<div class="check-item">'
        f'<div class="{dot_class}"></div>'
        f'<div class="check-text">{item}</div>'
        f'</div>'
        for item in items
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MedCore AI — Recommendation Engine Test")
    print("=" * 60)

    for disease in ("heart", "cancer"):
        for level in ("HIGH RISK", "MODERATE", "LOW RISK"):
            recs = get_recommendations(disease, level)
            print(f"\n🔹 {disease.upper()} · {level}")
            print(f"   Alert : {recs['alert_short']}")
            print(f"   Meds  : {len(recs['medicines'])} medications")
            print(f"   Diet  : {len(recs['diet'])} tips")
            print(f"   Tests : {len(recs['tests'])} investigations")

    print("\n\n🔹 Biomarker Flags Test")
    flags = get_biomarker_flags({
        "chol":     280,
        "trestbps": 145,
        "thalach":  58,
        "oldpeak":  2.4,
    })
    for f in flags:
        print(f"  {f['parameter']:<30} {f['value']} {f['unit']:<8} → {f['flag']}")
        print(f"    ↳ {f['action']}")