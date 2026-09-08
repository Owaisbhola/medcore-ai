"""
rag_chat.py
───────────
Retrieval-Augmented Generation (RAG) engine for the MedCore Health Chat Assistant.

Pipeline
--------
1. A curated medical knowledge base (heart / cancer / lab-value / medication / lifestyle
   documents) is split into overlapping chunks.
2. A user's question is scored against every chunk using TF-IDF + cosine similarity
   (pure Python — no extra dependencies, no external embedding service required).
3. The top-k chunks, plus (optionally) the patient's own most recently parsed lab
   report, are assembled into a grounding context.
4. That context + the question are sent to a local Ollama model (free) or, if
   configured, the Claude API — the same two backends already used for report
   summaries in report_parser.py.
5. If neither LLM backend is available, retrieval still runs and the best-matching
   knowledge base passage is returned directly, so the chat never goes fully blank.

This is intentionally dependency-free (no scikit-learn / faiss / chromadb) so it
runs anywhere Python does. Swapping in real sentence embeddings (e.g. Ollama's
`nomic-embed-text` or `sentence-transformers`) later is a drop-in upgrade — see
`retrieve()` for where to plug that in.
"""

import re
import math
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is one topic document. Longer documents get chunked automatically.

KNOWLEDGE_BASE = [
    {
        "id": "chol", "title": "Cholesterol & Lipid Profile", "category": "heart",
        "text": (
            "Cholesterol is a fatty substance carried in the blood, essential for building cells "
            "but harmful in excess. Total cholesterol above 200 mg/dL is considered borderline-high, "
            "and above 240 mg/dL is high. LDL ('bad' cholesterol) deposits in artery walls and forms "
            "plaques; LDL should stay under 100 mg/dL for most people, and under 70 mg/dL for those "
            "with existing heart disease. HDL ('good' cholesterol) removes cholesterol from the "
            "bloodstream; higher HDL (above 60 mg/dL) is protective, while below 40 mg/dL raises risk. "
            "Triglycerides, another blood fat, should stay under 150 mg/dL. Lipid panels are best "
            "measured after a 9-12 hour fast. Lifestyle changes — reducing saturated fat, increasing "
            "fibre, regular aerobic exercise, and weight loss — can lower LDL by 10-20%. When lifestyle "
            "changes are not enough, statins (e.g. atorvastatin, rosuvastatin) are the first-line "
            "medication, typically lowering LDL by 30-50%."
        ),
    },
    {
        "id": "bp", "title": "Blood Pressure", "category": "heart",
        "text": (
            "Blood pressure measures the force of blood against artery walls, recorded as systolic "
            "(pressure during a heartbeat) over diastolic (pressure between beats). Normal is below "
            "120/80 mmHg. Elevated is 120-129 systolic with diastolic below 80. Stage 1 hypertension is "
            "130-139/80-89. Stage 2 hypertension is 140/90 or higher. A hypertensive crisis is above "
            "180/120 and needs immediate medical attention. Uncontrolled high blood pressure damages "
            "artery walls over years, increasing risk of heart attack, stroke, and kidney disease. "
            "Management includes reducing sodium intake (under 2g/day), regular exercise, limiting "
            "alcohol, and medications such as ACE inhibitors, ARBs, calcium channel blockers, or "
            "diuretics depending on the patient's profile."
        ),
    },
    {
        "id": "ecg_angina", "title": "ECG, Angina & Exercise Stress Findings", "category": "heart",
        "text": (
            "An ECG (electrocardiogram) records the heart's electrical activity and can reveal "
            "rhythm problems, prior heart attacks, or signs of reduced blood flow. Angina is chest "
            "pain or discomfort caused by reduced blood flow to the heart muscle; exercise-induced "
            "angina is a significant marker of coronary artery disease. ST-segment depression during "
            "exercise (sometimes reported as 'oldpeak') reflects how much the heart's electrical "
            "recovery is delayed under stress — values greater than 2mm are considered significant and "
            "suggestive of ischaemia (inadequate blood supply to heart muscle). Maximum heart rate "
            "achieved during exercise testing is compared to the age-predicted maximum (roughly "
            "220 minus age); a lower-than-expected peak can indicate reduced cardiac reserve or "
            "medication effects such as beta-blockers."
        ),
    },
    {
        "id": "thal", "title": "Thalassemia & Fixed/Reversible Defects", "category": "heart",
        "text": (
            "Thalassemia is an inherited blood disorder affecting haemoglobin production; in cardiac "
            "risk models the 'thal' feature usually refers to thallium stress test results rather than "
            "the blood disorder itself: normal blood flow, a fixed defect (an area of the heart with "
            "permanently reduced blood flow, often from a prior heart attack), or a reversible defect "
            "(an area with reduced flow only during stress, suggesting a narrowed but not blocked "
            "artery). Reversible defects generally carry the highest short-term cardiac risk because "
            "they indicate actively reduced blood supply that could worsen."
        ),
    },
    {
        "id": "cardiac_meds", "title": "Common Cardiac Medications", "category": "medication",
        "text": (
            "Statins (atorvastatin, rosuvastatin, simvastatin) lower LDL cholesterol by blocking a "
            "liver enzyme (HMG-CoA reductase); they reduce heart attack risk by roughly 25-35% in "
            "high-risk patients. Low-dose aspirin (75-100mg) prevents platelets from clumping and "
            "forming clots; it is used for secondary prevention in patients with confirmed heart "
            "disease, not usually for primary prevention in low-risk individuals. Beta-blockers "
            "(metoprolol, atenolol) slow heart rate and reduce blood pressure, easing the heart's "
            "workload. ACE inhibitors (ramipril, lisinopril) relax blood vessels and are especially "
            "useful after a heart attack or in heart failure. None of these should be started, "
            "stopped, or changed without a doctor's guidance."
        ),
    },
    {
        "id": "cancer_markers_tumour", "title": "Tumour Cell Morphology Markers", "category": "cancer",
        "text": (
            "In breast tumour imaging analysis, several cell-shape features help distinguish benign "
            "from malignant growths. Mean radius is the average distance from the tumour cell "
            "nucleus centre to its perimeter — larger radii (generally above 15) are more often seen "
            "in malignant tumours. Mean texture measures variation in grey-scale intensity across the "
            "cell image; higher texture values indicate more irregular cell surfaces, which correlates "
            "with malignancy. Concavity measures how severe the indentations (concave portions) of the "
            "cell contour are — higher concavity (above roughly 0.15) strongly correlates with "
            "malignant cells, since cancer cells tend to have more irregular, jagged borders than "
            "smooth benign growths."
        ),
    },
    {
        "id": "cancer_diagnosis", "title": "Malignant vs Benign & Diagnosis", "category": "cancer",
        "text": (
            "Malignant means cancerous: the tumour cells can invade nearby tissue and spread "
            "(metastasise) to other parts of the body, and immediate oncology referral is required. "
            "Benign means non-cancerous: the growth does not invade surrounding tissue or spread, "
            "though it may still need monitoring or removal if it is causing symptoms. A biopsy is "
            "the definitive diagnostic test — a tissue sample is removed and examined under a "
            "microscope by a pathologist. Core needle biopsy is the standard approach for suspicious "
            "breast lumps. Imaging tests like mammograms, ultrasounds, and MRIs can flag suspicious "
            "areas but cannot confirm cancer on their own — a biopsy is needed for that."
        ),
    },
    {
        "id": "cancer_markers_blood", "title": "Tumour Marker Blood Tests (PSA, CA-125, CEA, AFP, etc.)", "category": "cancer",
        "text": (
            "Tumour markers are substances, often proteins, that can be elevated in the blood when "
            "certain cancers are present — though they can also be elevated for non-cancerous reasons, "
            "so they are used alongside imaging and biopsy, not as a stand-alone diagnosis. PSA "
            "(prostate-specific antigen) above 4.0 ng/mL warrants further prostate evaluation, though "
            "levels also rise with benign prostate enlargement and infection. CA-125 above 35 U/mL is "
            "associated with ovarian cancer but also rises with endometriosis, pregnancy, and "
            "menstruation. CA 19-9 is used mainly for pancreatic and gastrointestinal cancers, normal "
            "under 37 U/mL. CA 15-3 is used for breast cancer monitoring, normal under 30 U/mL. CEA "
            "(carcinoembryonic antigen) is used for colorectal cancer, normal under 3.0-5.0 ng/mL "
            "(smokers run slightly higher). AFP (alpha-fetoprotein) is used for liver and testicular "
            "cancers, normal under 10 ng/mL. Any elevated tumour marker should be discussed with a "
            "doctor and interpreted alongside imaging and clinical history, not in isolation."
        ),
    },
    {
        "id": "cancer_meds", "title": "Common Cancer Treatment Concepts", "category": "medication",
        "text": (
            "Treatment for cancer depends heavily on type, stage, and individual factors, and is "
            "always decided by an oncology team. Tamoxifen is an anti-oestrogen drug used for "
            "oestrogen-receptor-positive breast cancer; it blocks oestrogen receptors on cancer cells "
            "to slow their growth, typically taken for about 5 years. Chemotherapy uses drugs that "
            "target rapidly dividing cells throughout the body. Radiation therapy targets a specific "
            "area with high-energy beams. Surgery removes the tumour directly when feasible. "
            "Screening — such as annual mammograms from around age 40, or earlier with family "
            "history — remains one of the most effective tools for catching cancer early, when "
            "treatment is most successful."
        ),
    },
    {
        "id": "diabetes_labs", "title": "HbA1c & Blood Glucose", "category": "lab",
        "text": (
            "HbA1c (glycosylated haemoglobin) reflects average blood glucose over roughly the past "
            "3 months, because glucose attaches to haemoglobin in red blood cells over their "
            "120-day lifespan. Normal is below 5.7%. Pre-diabetes is 5.7-6.4%. Diabetes is 6.5% or "
            "higher. For people managing diabetes, a common treatment target is below 7%. Fasting "
            "blood glucose, measured after at least 8 hours without food, is normal at 70-99 mg/dL, "
            "pre-diabetic at 100-125 mg/dL, and diabetic at 126 mg/dL or above. Estimated average "
            "glucose (eAG), often reported alongside HbA1c, translates the percentage into an "
            "equivalent average daily glucose reading in mg/dL for easier interpretation."
        ),
    },
    {
        "id": "cbc_labs", "title": "Complete Blood Count (CBC) Basics", "category": "lab",
        "text": (
            "Haemoglobin carries oxygen in red blood cells; normal range is roughly 12-17 g/dL "
            "(varies by sex and lab). Low haemoglobin (anaemia) causes fatigue, weakness, and "
            "breathlessness, and has many causes including iron deficiency, chronic disease, or blood "
            "loss. High haemoglobin can indicate dehydration or a blood disorder like polycythaemia. "
            "White blood cell count reflects the immune system's activity; elevated counts often "
            "indicate infection or inflammation, while low counts can indicate bone marrow problems or "
            "certain infections. Platelet count reflects the blood's clotting ability; both very high "
            "and very low platelet counts carry bleeding or clotting risks and need medical evaluation."
        ),
    },
    {
        "id": "kidney_thyroid_labs", "title": "Kidney & Thyroid Function Tests", "category": "lab",
        "text": (
            "Creatinine is a waste product filtered out by the kidneys; normal is roughly "
            "0.6-1.2 mg/dL. Elevated creatinine suggests the kidneys are not filtering waste "
            "efficiently, which can indicate chronic kidney disease, dehydration, or certain "
            "medication effects. TSH (thyroid-stimulating hormone) is produced by the pituitary "
            "gland to regulate the thyroid; normal is roughly 0.4-4.0 mIU/L. High TSH usually means "
            "an underactive thyroid (hypothyroidism, since the pituitary is working harder to "
            "stimulate a sluggish thyroid). Low TSH usually means an overactive thyroid "
            "(hyperthyroidism). Uric acid, a waste product of purine metabolism, is normal at roughly "
            "3.5-7.2 mg/dL; elevated levels can cause gout (painful joint inflammation) and are "
            "linked to kidney stones."
        ),
    },
    {
        "id": "vitamin_labs", "title": "Vitamin D & Other Nutrient Markers", "category": "lab",
        "text": (
            "Vitamin D supports bone health, calcium absorption, and immune function. A level above "
            "30 ng/mL is considered sufficient; 20-29 ng/mL is insufficient; below 20 ng/mL is "
            "deficient. Vitamin D deficiency is very common, especially with limited sun exposure, "
            "and has been linked in research to higher cardiovascular and certain cancer risks, "
            "though supplementation to fix a deficiency is far better supported by evidence than "
            "high-dose supplementation in people who are not deficient. Sunlight exposure, fatty fish, "
            "fortified foods, and supplements (typically 1000-2000 IU/day for maintenance, higher for "
            "correcting deficiency under medical guidance) are the main ways to maintain healthy "
            "levels."
        ),
    },
    {
        "id": "lifestyle", "title": "Diet, Exercise, Smoking & Weight", "category": "lifestyle",
        "text": (
            "A Mediterranean-style diet — olive oil, nuts, fish, legumes, vegetables, and whole "
            "grains, with limited red meat and processed food — is one of the best-studied diets for "
            "reducing heart disease and supporting overall health. Aiming for at least 25-30g of "
            "fibre daily and keeping sodium under 2g/day supports both heart and metabolic health. "
            "150 minutes per week of moderate aerobic activity (brisk walking, cycling, swimming) is "
            "the standard recommendation, and is associated with roughly a 35% reduction in heart "
            "disease risk and meaningful reductions in several cancer risks. Smoking roughly doubles "
            "heart disease risk and significantly raises risk for several cancers; cardiac risk drops "
            "substantially within about a year of quitting. BMI (weight in kg divided by height in "
            "metres squared) categorises 18.5-24.9 as normal weight, 25-29.9 as overweight, and 30+ as "
            "obese — higher BMI raises risk for heart disease, type 2 diabetes, and several cancers, "
            "though BMI alone doesn't capture body composition and shouldn't be the only measure used."
        ),
    },
    {
        "id": "stress_sleep", "title": "Stress, Sleep & Mental Wellbeing", "category": "lifestyle",
        "text": (
            "Chronic stress raises cortisol levels, which can elevate blood pressure and promote "
            "inflammation, both of which raise cardiovascular risk over time. Poor sleep (regularly "
            "under 6 hours) is independently linked to higher blood pressure, weight gain, and "
            "impaired glucose control. Aiming for 7-9 hours of sleep per night, regular physical "
            "activity, mindfulness or meditation practices, and maintaining social connections are "
            "all evidence-supported ways to manage stress and support long-term health."
        ),
    },
    {
        "id": "model_info", "title": "How MedCore's Risk Models Work", "category": "model",
        "text": (
            "MedCore's heart disease and cancer risk models are Random Forest classifiers — an "
            "ensemble of 200 decision trees that each 'vote' on a prediction, and the votes are "
            "combined into a final risk probability. The heart model uses 13 clinical features "
            "(age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, resting "
            "ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, number of "
            "major vessels, and thalassemia/thallium test result) and achieves roughly 93% accuracy "
            "with an AUC-ROC around 0.96 on its test set. The cancer model uses 10 core tumour "
            "morphology features and achieves roughly 89% accuracy with an AUC-ROC around 0.93. A "
            "risk score is the model's estimated probability of disease, not a certainty — scores "
            "above 50% are generally treated as high risk, 35-50% as moderate, and below 35% as low "
            "risk, but any elevated score should prompt a conversation with a doctor rather than be "
            "treated as a diagnosis on its own."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_words(text: str, max_words: int = 90, overlap: int = 20):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _build_chunks():
    chunks = []
    for doc in KNOWLEDGE_BASE:
        for i, piece in enumerate(_chunk_words(doc["text"])):
            chunks.append({
                "doc_id": doc["id"], "title": doc["title"], "category": doc["category"],
                "chunk_idx": i, "text": piece,
            })
    return chunks


CHUNKS = _build_chunks()

# ─────────────────────────────────────────────────────────────────────────────
# 3. TF-IDF RETRIEVAL (pure Python — no sklearn/faiss dependency)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","he","in",
    "is","it","its","of","on","or","that","the","to","was","were","will","with",
    "what","which","who","whom","this","these","those","i","you","your","my",
    "me","do","does","did","can","could","should","would","if","about","how",
}


def _tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _build_tfidf_index(chunks):
    """Return (doc_freq, chunk_term_freqs) used to score queries against chunks."""
    doc_freq = Counter()
    chunk_tfs = []
    for c in chunks:
        tokens = _tokenize(c["text"] + " " + c["title"])
        tf = Counter(tokens)
        chunk_tfs.append(tf)
        for term in tf:
            doc_freq[term] += 1
    return doc_freq, chunk_tfs


_DOC_FREQ, _CHUNK_TFS = _build_tfidf_index(CHUNKS)
_N_CHUNKS = len(CHUNKS)


def _tfidf_vector(tf: Counter, doc_freq: Counter, n_docs: int):
    vec = {}
    for term, count in tf.items():
        idf = math.log((n_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1
        vec[term] = (1 + math.log(count)) * idf
    return vec


def _cosine(v1: dict, v2: dict):
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


_CHUNK_VECTORS = [_tfidf_vector(tf, _DOC_FREQ, _N_CHUNKS) for tf in _CHUNK_TFS]


def retrieve(query: str, k: int = 4, min_score: float = 0.05):
    """
    Return the top-k knowledge-base chunks most relevant to `query`, as a list of
    dicts with a similarity `score` added, sorted highest-first.

    To upgrade this to real embedding-based retrieval later: replace
    `_tfidf_vector` + `_cosine` with calls to an embedding model (e.g. Ollama's
    `nomic-embed-text` via /api/embeddings, or sentence-transformers) and swap in
    a vector index (faiss/chromadb) — `retrieve()`'s signature can stay the same.
    """
    q_tf = Counter(_tokenize(query))
    q_vec = _tfidf_vector(q_tf, _DOC_FREQ, _N_CHUNKS)
    scored = []
    for chunk, vec in zip(CHUNKS, _CHUNK_VECTORS):
        score = _cosine(q_vec, vec)
        if score >= min_score:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATIENT CONTEXT (grounds the answer in the user's own latest report, if any)
# ─────────────────────────────────────────────────────────────────────────────

def build_patient_context(parsed_result: dict | None) -> str:
    """Turn the most recently parsed OCR report (if any) into a short context block."""
    if not parsed_result or not parsed_result.get("all_values"):
        return "No lab report has been uploaded/parsed yet in this session."

    all_values = parsed_result.get("all_values", {})
    abnormal   = parsed_result.get("abnormal_flags", [])
    lines = [f"Report type detected: {parsed_result.get('report_type', 'general')}"]
    lines += [f"- {k.replace('_',' ').title()}: {v}" for k, v in all_values.items()]
    if abnormal:
        lines.append("Abnormal values:")
        lines += [f"- {f['key'].replace('_',' ').title()}: {f['value']} {f['unit']} "
                   f"({f['flag']}, normal {f['normal']})" for f in abnormal]
    else:
        lines.append("All values are within normal range.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROMPT ASSEMBLY + ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_RAG_SYSTEM = """You are the MedCore Health Assistant, embedded in a clinical-intelligence \
dashboard. Answer the patient's question using ONLY the information given in the "Reference \
material" and "Patient's latest report" sections below — do not invent facts, drug dosages, or \
numbers that are not there. If the reference material doesn't cover the question, say so plainly \
and suggest they ask their doctor, rather than guessing.

Rules:
- Keep answers concise (3-6 sentences unless the question needs a list).
- Explain medical terms in plain language.
- If the patient's own report contains a value relevant to the question, refer to it specifically.
- Never diagnose, prescribe, or recommend a specific medication dose.
- End with a short reminder to consult a doctor ONLY when the topic is something that genuinely \
needs clinical judgement (not for simple definitions).
"""

_RAG_USER_TEMPLATE = """Reference material:
{context}

Patient's latest report:
{patient_context}

Patient's question: {question}"""


def build_rag_prompt(question: str, parsed_result: dict | None = None, k: int = 4):
    chunks = retrieve(question, k=k)
    context = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks) or "No closely matching reference material found."
    patient_context = build_patient_context(parsed_result)
    user = _RAG_USER_TEMPLATE.format(context=context, patient_context=patient_context, question=question)
    return _RAG_SYSTEM, user, chunks


def answer_question(
    question: str,
    parsed_result: dict | None = None,
    backend: str = "ollama",
    model: str = "llama3",
    k: int = 4,
):
    """
    Full RAG pipeline: retrieve → assemble prompt → generate. Falls back to
    returning the best-matching raw passage if no LLM backend is available or
    the call fails, so the chat always returns something useful.
    """
    system, user, chunks = build_rag_prompt(question, parsed_result, k=k)

    if backend == "ollama":
        try:
            import report_parser as rpx
        except ImportError:
            from ocr import report_parser as rpx
        if rpx.ollama_is_running():
            try:
                import requests
                resp = requests.post(
                    f"{rpx.OLLAMA_HOST}/api/chat",
                    json={"model": model, "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ], "stream": False},
                    timeout=90,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"], chunks
            except Exception as e:
                pass  # fall through to raw-passage fallback below

    elif backend == "claude":
        try:
            import report_parser as rpx
        except ImportError:
            from ocr import report_parser as rpx
        if rpx.CLAUDE_AVAILABLE:
            try:
                response = rpx._get_client().messages.create(
                    model="claude-sonnet-4-6", max_tokens=800,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text, chunks
            except Exception:
                pass

    # ── No LLM available / call failed → return the best-matching passage directly ──
    if chunks:
        top = chunks[0]
        return (
            f"(No AI backend available — showing the closest reference passage.)\n\n"
            f"**{top['title']}**: {top['text']}"
        ), chunks
    return (
        "I couldn't find anything relevant in the knowledge base for that, and no AI backend "
        "is currently running. Try asking about a specific term (e.g. cholesterol, HbA1c, "
        "CA-125, blood pressure), or set up Ollama for full free-form answers."
    ), chunks