import re, os, json
from pathlib import Path

# ── Optional OCR / image deps ─────────────────────────────────────────────────

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠  pytesseract/Pillow not installed  →  pip install pytesseract Pillow")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader   = None
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠  easyocr not installed  →  pip install easyocr")

try:
    import cv2, numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠  pdf2image not installed  →  pip install pdf2image")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False

# ── Anthropic (for AI summary) ────────────────────────────────────────────────
try:
    import anthropic
    _anthropic_client: anthropic.Anthropic | None = None
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("⚠  anthropic not installed  →  pip install anthropic")


# ═══════════════════════════════════════════════════════════════════════════════
# 0. LANGUAGE SUPPORT
# Supported values for the `language` parameter (case-insensitive).
# Users can also pass any language name — Claude handles it natively.
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", "Telugu",
    "Marathi", "Kannada", "Malayalam", "Punjabi", "Urdu",
    "Spanish", "French", "German", "Arabic", "Portuguese",
    "Indonesian", "Swahili", "Japanese", "Chinese (Simplified)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMAGE PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_pil(image_path: str):
    """Grayscale → upscale → contrast enhance → sharpen."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    if w < 1000:
        img = img.resize((int(w * 1000 / w), int(h * 1000 / w)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def preprocess_cv2(image_path: str) -> str:
    """Adaptive threshold + light dilation. Returns path to temp file."""
    if not CV2_AVAILABLE:
        return image_path
    img    = cv2.imread(image_path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    tmp = image_path.replace(".", "_proc.")
    cv2.imwrite(tmp, dilated)
    return tmp


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_tesseract(image_path: str) -> str:
    """Tesseract OCR — good for clean printed lab reports."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed. Run: pip install pytesseract Pillow")
    img = preprocess_pil(image_path)
    cfg = "--psm 6 --oem 3 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(img, config=cfg, lang="eng").strip()


def extract_text_easyocr(image_path: str) -> str:
    """EasyOCR — better on handwritten, skewed, or poor-quality scans."""
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not installed. Run: pip install easyocr")
    global _easyocr_reader
    if _easyocr_reader is None:
        print("⏳ Loading EasyOCR model (first time only)…")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    results = _easyocr_reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results)


def extract_text_from_pdf(
    pdf_path: str,
    engine: str = "easyocr",
    progress_callback=None,   # callable(page_num, total_pages)
) -> str:
    """
    Convert PDF pages to images then run OCR on each page.
    progress_callback(i, total) is called after each page — useful for Streamlit progress bars.
    """
    if not PDF_AVAILABLE:
        raise RuntimeError(
            "pdf2image not installed. Run: pip install pdf2image\n"
            "Also install Poppler:\n"
            "  macOS  : brew install poppler\n"
            "  Ubuntu : sudo apt install poppler-utils"
        )
    pages = convert_from_path(pdf_path, dpi=300)
    texts = []
    total = len(pages)
    for i, page in enumerate(pages):
        tmp = f"/tmp/medcore_page_{i}.png"
        page.save(tmp, "PNG")
        try:
            text = (
                extract_text_tesseract(tmp)
                if engine == "tesseract"
                else extract_text_easyocr(tmp)
            )
            texts.append(f"--- Page {i+1} ---\n{text}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if progress_callback:
            progress_callback(i + 1, total)
    return "\n\n".join(texts)


def extract_text(file_path: str, engine: str = "easyocr", progress_callback=None) -> str:
    """Unified entry point — auto-detects PDF vs image."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, engine, progress_callback)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
        return (
            extract_text_tesseract(file_path)
            if engine == "tesseract"
            else extract_text_easyocr(file_path)
        )
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGEX PARAMETER RULES  (expanded — 20 parameters across all report types)
# ═══════════════════════════════════════════════════════════════════════════════

PARAMETER_RULES = [
    # ── Lipid panel ───────────────────────────────────────────────────────────
    {
        "key": "Cholesterol",
        "patterns": [
            r"(?:total\s+)?cholesterol(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"chol(?:esterol)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "< 200 mg/dL",
        "flag": lambda v: ("Critical" if v > 300 else "High" if v > 239 else "Borderline" if v > 199 else "Normal"),
        "heart_feature": "chol",
    },
    {
        "key": "LDL Cholesterol",
        "patterns": [
            r"ldl(?:\s+cholesterol|-c)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"low\s+density\s+lipoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "< 100 mg/dL",
        "flag": lambda v: ("Very High" if v > 189 else "High" if v > 159 else "Borderline" if v > 129 else "Near Optimal" if v > 99 else "Optimal"),
        "heart_feature": None,
    },
    {
        "key": "HDL Cholesterol",
        "patterns": [
            r"hdl(?:\s+cholesterol|-c)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"high\s+density\s+lipoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "> 40 mg/dL",
        "flag": lambda v: ("Low (Risk)" if v < 40 else "Borderline" if v < 60 else "Protective"),
        "heart_feature": None,
    },
    {
        "key": "Triglycerides",
        "patterns": [
            r"triglycerides?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\btg\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "< 150 mg/dL",
        "flag": lambda v: ("Very High" if v > 499 else "High" if v > 199 else "Borderline" if v > 149 else "Normal"),
        "heart_feature": None,
    },
    # ── Cardiac ───────────────────────────────────────────────────────────────
    {
        "key": "Systolic BP",
        "patterns": [
            r"(?:blood\s*pressure|b\.?p\.?|systolic)(?:\s*[:=\-]\s*|\s+)(\d{2,3})\s*/\s*\d+",
            r"sbp(?:\s*[:=\-]\s*|\s+)(\d{2,3})",
        ],
        "unit": "mmHg", "normal": "< 120 mmHg",
        "flag": lambda v: ("Critical" if v > 179 else "High" if v > 139 else "Elevated" if v > 119 else "Normal"),
        "heart_feature": "trestbps",
    },
    {
        "key": "Diastolic BP",
        "patterns": [
            r"(?:blood\s*pressure|b\.?p\.?)(?:\s*[:=\-]\s*|\s+)\d+\s*/\s*(\d{2,3})",
            r"dbp(?:\s*[:=\-]\s*|\s+)(\d{2,3})",
        ],
        "unit": "mmHg", "normal": "< 80 mmHg",
        "flag": lambda v: ("High" if v > 89 else "Elevated" if v > 79 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Heart Rate",
        "patterns": [
            r"(?:heart\s+rate|h\.?r\.?|pulse)(?:\s*[:=\-]\s*|\s+)(\d+)",
            r"(?:max(?:imum)?\s+heart\s+rate|thalach)(?:\s*[:=\-]\s*|\s+)(\d+)",
        ],
        "unit": "bpm", "normal": "60 – 100 bpm",
        "flag": lambda v: ("High (Tachycardia)" if v > 100 else "Low (Bradycardia)" if v < 60 else "Normal"),
        "heart_feature": "thalach",
    },
    # ── CBC ───────────────────────────────────────────────────────────────────
    {
        "key": "Haemoglobin",
        "patterns": [
            r"ha?emoglobin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\bhb\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"hgb(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "g/dL", "normal": "12.0 – 17.5 g/dL",
        "flag": lambda v: ("Critical Low" if v < 8.0 else "Low" if v < 12.0 else "High" if v > 18.0 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "WBC",
        "patterns": [r"(?:wbc|white\s+blood\s+cell|tlc)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "× 10³/µL", "normal": "4.0 – 11.0 × 10³/µL",
        "flag": lambda v: ("High (Leukocytosis)" if v > 11.0 else "Low (Leukopenia)" if v < 4.0 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Platelets",
        "patterns": [
            r"platelet(?:s)?\s*(?:count)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"plt(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "× 10³/µL", "normal": "150 – 400 × 10³/µL",
        "flag": lambda v: ("High (Thrombocytosis)" if v > 400 else "Low (Thrombocytopenia)" if v < 150 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "RBC",
        "patterns": [r"(?:rbc|red\s+blood\s+cell)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "× 10⁶/µL", "normal": "4.2 – 5.9 × 10⁶/µL",
        "flag": lambda v: ("Low (Anaemia)" if v < 4.2 else "High (Polycythaemia)" if v > 5.9 else "Normal"),
        "heart_feature": None,
    },
    # ── Metabolic ─────────────────────────────────────────────────────────────
    {
        "key": "Fasting Blood Glucose",
        "patterns": [
            r"(?:fasting\s+(?:blood\s+)?glucose|f\.?b\.?g\.?|fbs)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"glucose\s*(?:\(fasting\))?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "70 – 99 mg/dL",
        "flag": lambda v: ("Critical" if v > 300 else "Diabetic" if v > 125 else "Pre-diabetic" if v > 99 else "Low" if v < 70 else "Normal"),
        "heart_feature": "fbs",
    },
    {
        "key": "HbA1c",
        "patterns": [
            r"hba\s*1\s*c\)?(?:\s*\([^\)]*\))?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"(?:glycosylated|glycated)\s+h(?:a)?emoglobin(?:\s*\([^\)]*\))?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\ba1c\)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "%", "normal": "< 5.7%",
        "flag": lambda v: ("Critical" if v > 10.0 else "Diabetic" if v >= 6.5 else "Pre-diabetic" if v >= 5.7 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Estimated Average Glucose",
        "patterns": [
            r"(?:estimated\s+(?:average\s+)?glucose|\beag\b)(?:\s*\([^\)]*\))?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "< 126 mg/dL",
        "flag": lambda v: ("Critical" if v > 200 else "High" if v > 126 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Creatinine",
        "patterns": [
            r"(?:serum\s+)?creatinine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"s\.?\s*creatinine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "0.7 – 1.2 mg/dL",
        "flag": lambda v: ("High (Renal concern)" if v > 1.5 else "Low" if v < 0.5 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Uric Acid",
        "patterns": [r"uric\s+acid(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "mg/dL", "normal": "3.5 – 7.2 mg/dL",
        "flag": lambda v: ("High (Gout risk)" if v > 7.2 else "Low" if v < 3.5 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Urea / BUN",
        "patterns": [
            r"(?:blood\s+urea|serum\s+urea|urea)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\bbun\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "7 – 20 mg/dL",
        "flag": lambda v: ("High" if v > 20 else "Low" if v < 7 else "Normal"),
        "heart_feature": None,
    },
    # ── Liver ─────────────────────────────────────────────────────────────────
    {
        "key": "SGPT / ALT",
        "patterns": [r"(?:sgpt|alt|alanine\s+aminotransferase)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "U/L", "normal": "< 40 U/L",
        "flag": lambda v: ("High (Liver concern)" if v > 40 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "SGOT / AST",
        "patterns": [r"(?:sgot|ast|aspartate\s+aminotransferase)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "U/L", "normal": "< 40 U/L",
        "flag": lambda v: ("High (Liver concern)" if v > 40 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Bilirubin",
        "patterns": [
            r"(?:total\s+)?bilirubin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"t\.?\s*bili(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mg/dL", "normal": "< 1.2 mg/dL",
        "flag": lambda v: ("High (Jaundice risk)" if v > 1.2 else "Normal"),
        "heart_feature": None,
    },
    # ── Thyroid ───────────────────────────────────────────────────────────────
    {
        "key": "TSH",
        "patterns": [
            r"\btsh\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"thyroid\s+stimulating\s+hormone(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mIU/L", "normal": "0.4 – 4.0 mIU/L",
        "flag": lambda v: ("High (Hypothyroid)" if v > 4.0 else "Low (Hyperthyroid)" if v < 0.4 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "T3",
        "patterns": [
            r"\bt3\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"triiodothyronine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "ng/dL", "normal": "80 – 200 ng/dL",
        "flag": lambda v: ("Low" if v < 80 else "High" if v > 200 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "T4",
        "patterns": [
            r"\bt4\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"thyroxine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "µg/dL", "normal": "5.1 – 14.1 µg/dL",
        "flag": lambda v: ("Low" if v < 5.1 else "High" if v > 14.1 else "Normal"),
        "heart_feature": None,
    },
    # ── Kidney ────────────────────────────────────────────────────────────────
    {
        "key": "eGFR",
        "patterns": [
            r"(?:egfr|estimated\s+gfr|glomerular\s+filtration)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mL/min/1.73m²", "normal": "> 60",
        "flag": lambda v: ("Severe (Stage 4–5)" if v < 30 else "Moderate (Stage 3)" if v < 60 else "Mild reduction" if v < 90 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Sodium",
        "patterns": [
            r"(?:serum\s+)?sodium(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\bna\+?\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mEq/L", "normal": "136 – 145 mEq/L",
        "flag": lambda v: ("Low (Hyponatraemia)" if v < 136 else "High (Hypernatraemia)" if v > 145 else "Normal"),
        "heart_feature": None,
    },
    {
        "key": "Potassium",
        "patterns": [
            r"(?:serum\s+)?potassium(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\bk\+?\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mEq/L", "normal": "3.5 – 5.0 mEq/L",
        "flag": lambda v: ("Low (Hypokalaemia)" if v < 3.5 else "High (Hyperkalaemia)" if v > 5.0 else "Normal"),
        "heart_feature": None,
    },
    # ── Cancer / Tumour markers ──────────────────────────────────────────────
    {
        "key": "PSA (Prostate)",
        "patterns": [
            r"(?:total\s+)?psa(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"prostate\s+specific\s+antigen(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "ng/mL", "normal": "< 4.0 ng/mL",
        "flag": lambda v: ("High (Needs follow-up)" if v > 10 else "Borderline" if v > 4.0 else "Normal"),
        "heart_feature": None, "cancer_feature": "psa",
    },
    {
        "key": "CA-125 (Ovarian)",
        "patterns": [r"ca[\s\-]?125(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "U/mL", "normal": "< 35 U/mL",
        "flag": lambda v: ("High (Needs follow-up)" if v > 35 else "Normal"),
        "heart_feature": None, "cancer_feature": "ca125",
    },
    {
        "key": "CA 19-9 (Pancreatic/GI)",
        "patterns": [r"ca[\s\-]?19[\s\-]?9(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "U/mL", "normal": "< 37 U/mL",
        "flag": lambda v: ("High (Needs follow-up)" if v > 37 else "Normal"),
        "heart_feature": None, "cancer_feature": "ca19_9",
    },
    {
        "key": "CA 15-3 (Breast)",
        "patterns": [r"ca[\s\-]?15[\s\-]?3(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"],
        "unit": "U/mL", "normal": "< 30 U/mL",
        "flag": lambda v: ("High (Needs follow-up)" if v > 30 else "Normal"),
        "heart_feature": None, "cancer_feature": "ca15_3",
    },
    {
        "key": "CEA (Colorectal/General)",
        "patterns": [
            r"\bcea\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"carcinoembryonic\s+antigen(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "ng/mL", "normal": "< 3.0 ng/mL (non-smoker)",
        "flag": lambda v: ("High (Needs follow-up)" if v > 5.0 else "Borderline" if v > 3.0 else "Normal"),
        "heart_feature": None, "cancer_feature": "cea",
    },
    {
        "key": "AFP (Liver/Testicular)",
        "patterns": [
            r"\bafp\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"alpha[\s\-]?fetoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "ng/mL", "normal": "< 10 ng/mL",
        "flag": lambda v: ("High (Needs follow-up)" if v > 10 else "Normal"),
        "heart_feature": None, "cancer_feature": "afp",
    },
    {
        "key": "Beta-hCG",
        "patterns": [
            r"beta[\s\-]?hcg(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
            r"\bhcg\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)",
        ],
        "unit": "mIU/mL", "normal": "< 5 mIU/mL (non-pregnant)",
        "flag": lambda v: ("High (Needs follow-up)" if v > 5 else "Normal"),
        "heart_feature": None, "cancer_feature": "beta_hcg",
    },
]

# ── Keyword signals used by detect_report_type() ──────────────────────────────
_REPORT_TYPE_KEYWORDS = {
    "cancer": [
        "psa", "ca-125", "ca 125", "ca125", "ca 19-9", "ca19-9", "ca 15-3",
        "ca15-3", "cea", "afp", "tumour", "tumor", "oncology", "biopsy",
        "carcinoma", "malignan", "alpha-fetoprotein", "carcinoembryonic",
    ],
    "heart": [
        "cardiac", "cardiology", "ecg", "ekg", "electrocardiogram", "troponin",
        "bnp", "echocardiogram", "angiogram", "lipid profile", "coronary",
    ],
    "diabetes": [
        "glucose", "hba1c", "a1c", "diabetic", "diabetes", "insulin",
    ],
}


def detect_report_type(text: str) -> str:
    """
    Best-effort classification of a report as 'cancer', 'heart', 'diabetes',
    or 'general', based on keyword signals in the raw OCR text. A report can
    contain markers for more than one condition — this returns the type with
    the strongest signal, used to route the summary/model, not to exclude data.
    """
    text_lower = text.lower()
    scores = {kind: 0 for kind in _REPORT_TYPE_KEYWORDS}
    for kind, keywords in _REPORT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[kind] += 1
    best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_kind if best_score > 0 else "general"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PARSING + MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_report_text(text: str) -> dict:
    """Run all regex rules against text. Returns dict of found parameters."""
    text_lower = text.lower()
    results    = {}
    for rule in PARAMETER_RULES:
        for pattern in rule["patterns"]:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1))
                    results[rule["key"]] = {
                        "value":          val,
                        "unit":           rule["unit"],
                        "normal":         rule["normal"],
                        "flag":           rule["flag"](val),
                        "heart_feature":  rule.get("heart_feature"),
                        "cancer_feature": rule.get("cancer_feature"),
                    }
                    break
                except (ValueError, IndexError):
                    continue
    return results


def map_to_heart_features(parsed: dict) -> dict:
    """Map parsed values → heart model feature names."""
    mapped = {}
    for param, info in parsed.items():
        feat = info.get("heart_feature")
        if not feat:
            continue
        if feat == "fbs":
            mapped["fbs"] = 1 if info["value"] > 120 else 0
        else:
            mapped[feat] = info["value"]
    return mapped


def map_to_cancer_features(parsed: dict) -> dict:
    """Map parsed tumour-marker values → cancer model feature names."""
    mapped = {}
    for param, info in parsed.items():
        feat = info.get("cancer_feature")
        if feat:
            mapped[feat] = info["value"]
    return mapped


def extract_patient_info(text: str) -> dict:
    """Try to extract name, age, sex, date, doctor, lab name from report header."""
    info = {}
    name = re.search(
        r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{2,40}?)(?:\n|age|dob|\d)",
        text, re.IGNORECASE,
    )
    if name:
        info["name"] = name.group(1).strip().title()

    age = re.search(r"(?:age|yrs?|years?)\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?)?", text, re.IGNORECASE)
    if age:
        val = int(age.group(1))
        if 1 <= val <= 120:
            info["age"] = val

    sex = re.search(r"(?:sex|gender)\s*[:\-]\s*(male|female|m|f)\b", text, re.IGNORECASE)
    if sex:
        s = sex.group(1).lower()
        info["sex"] = "M" if s.startswith("m") else "F"

    date = re.search(
        r"(?:date|collected|reported|sample\s+date)\s*[:\-]\s*(\d{1,2}[\s\/\-\.]\w{2,9}[\s\/\-\.]\d{2,4})",
        text, re.IGNORECASE,
    )
    if date:
        info["report_date"] = date.group(1).strip()

    doctor = re.search(
        r"(?:ref(?:erring)?\s+(?:doctor|physician|dr\.?)|dr\.?)\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,30}?)(?:\n|\d|lab)",
        text, re.IGNORECASE,
    )
    if doctor:
        info["referring_doctor"] = doctor.group(1).strip().title()

    lab = re.search(
        r"^([A-Za-z][A-Za-z\s&\.]{5,50}(?:hospital|lab(?:oratory)?|diagnostic|clinic|centre))",
        text, re.IGNORECASE | re.MULTILINE,
    )
    if lab:
        info["lab_name"] = lab.group(1).strip().title()

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 5. spaCy NLP
# ═══════════════════════════════════════════════════════════════════════════════

def nlp_extract_entities(text: str) -> list:
    """Extract named entities using spaCy en_core_web_sm."""
    if not SPACY_AVAILABLE:
        return []
    doc = _nlp(text[:50_000])
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in ("QUANTITY", "CARDINAL", "PERCENT", "ORG", "DATE", "PERSON", "GPE")
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AI SUMMARY  (Claude-powered, any language)
# ═══════════════════════════════════════════════════════════════════════════════

_AI_SYSTEM = """You are a compassionate medical interpreter AI for MedCore.
Your job is to explain a patient's lab report in simple, non-technical language —
as if explaining to a family member with no medical background.

Your response MUST:
1. Be in {language} — use that language exclusively, including medical term translations.
2. Start with a one-sentence overall verdict (e.g., "Your report looks mostly normal with a few things to watch.")
3. List each ABNORMAL value in plain language: what it measures, what your value means, and why it matters.
4. List any NORMAL values briefly in one short paragraph.
5. Give 2–3 simple lifestyle suggestions based on the findings.
6. End with: "Please show this report to your doctor for proper advice."
7. Use simple words, short sentences, and friendly tone. No complex medical jargon.
8. Do NOT diagnose. Do NOT recommend specific medications.
"""

_AI_USER_TEMPLATE = """Here is the patient's lab report data:

Patient: {patient_info}

Detected Values:
{values_block}

Abnormal Findings:
{abnormal_block}

Please summarise this report in {language} in plain, easy-to-understand language."""


def _get_anthropic_client() -> "anthropic.Anthropic":
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    return _anthropic_client


def ai_summarise(
    parsed_result: dict,
    language: str = "English",
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Generate a patient-friendly AI summary of a parsed report in any language.

    Parameters
    ----------
    parsed_result : dict returned by process_report() or process_pasted_text()
    language      : target language (e.g. "Hindi", "Gujarati", "Spanish", "English")
    model         : Claude model string

    Returns
    -------
    str — plain-language summary in the requested language.
         On error, returns an English fallback message.
    """
    if not CLAUDE_AVAILABLE:
        return "AI summary unavailable — install anthropic: pip install anthropic"

    parsed_values = parsed_result.get("parsed_values", {})
    patient_info  = parsed_result.get("patient_info", {})

    # Build values block
    values_lines = []
    abnormal_lines = []
    for key, info in parsed_values.items():
        line = f"  • {key}: {info['value']} {info['unit']}  (Normal: {info['normal']})  → {info['flag']}"
        values_lines.append(line)
        if info["flag"].lower() not in ("normal", "optimal", "protective", "near optimal"):
            abnormal_lines.append(line)

    if not values_lines:
        return f"No lab values were detected in the report. Please verify the report content."

    patient_str = ", ".join(f"{k}: {v}" for k, v in patient_info.items()) or "Not provided"

    system = _AI_SYSTEM.replace("{language}", language)
    user   = _AI_USER_TEMPLATE.format(
        patient_info   = patient_str,
        values_block   = "\n".join(values_lines) or "No values detected",
        abnormal_block = "\n".join(abnormal_lines) or "None — all values within normal range",
        language       = language,
    )

    try:
        client   = _get_anthropic_client()
        response = client.messages.create(
            model      = model,
            max_tokens = 1500,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return response.content[0].text

    except Exception as e:
        return f"AI summary could not be generated: {e}"


def ai_summarise_stream(
    parsed_result: dict,
    language: str = "English",
    model: str = "claude-sonnet-4-6",
):
    """
    Streaming version — yields text chunks for st.write_stream() in Streamlit.

    Usage in dashboard.py:
        with st.spinner(""):
            summary = st.write_stream(ai_summarise_stream(result, language=selected_lang))
    """
    if not CLAUDE_AVAILABLE:
        yield "AI summary unavailable — install anthropic: pip install anthropic"
        return

    parsed_values = parsed_result.get("parsed_values", {})
    patient_info  = parsed_result.get("patient_info", {})

    values_lines   = []
    abnormal_lines = []
    for key, info in parsed_values.items():
        line = f"  • {key}: {info['value']} {info['unit']}  (Normal: {info['normal']})  → {info['flag']}"
        values_lines.append(line)
        if info["flag"].lower() not in ("normal", "optimal", "protective", "near optimal"):
            abnormal_lines.append(line)

    if not values_lines:
        yield "No lab values were detected in the report."
        return

    patient_str = ", ".join(f"{k}: {v}" for k, v in patient_info.items()) or "Not provided"

    system = _AI_SYSTEM.replace("{language}", language)
    user   = _AI_USER_TEMPLATE.format(
        patient_info   = patient_str,
        values_block   = "\n".join(values_lines),
        abnormal_block = "\n".join(abnormal_lines) or "None",
        language       = language,
    )

    try:
        client = _get_anthropic_client()
        with client.messages.stream(
            model      = model,
            max_tokens = 1500,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\nAI summary error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_result(raw_text: str, engine: str, language: str = "English", with_ai: bool = True) -> dict:
    parsed        = parse_report_text(raw_text)
    heart_feats   = map_to_heart_features(parsed)
    cancer_feats  = map_to_cancer_features(parsed)
    report_type   = detect_report_type(raw_text)
    patient_info  = extract_patient_info(raw_text)
    entities      = nlp_extract_entities(raw_text)

    result = {
        "raw_text":        raw_text,
        "parsed_values":   parsed,
        "heart_features":  heart_feats,
        "cancer_features": cancer_feats,
        "report_type":     report_type,
        "patient_info":    patient_info,
        "entities":        entities,
        "engine_used":     engine,
        "language":        language,
        "ai_summary":      None,   # filled below if with_ai=True
    }

    if with_ai and parsed:
        result["ai_summary"] = ai_summarise(result, language=language)

    return result


def process_report(
    file_path: str,
    engine: str = "easyocr",
    language: str = "English",
    with_ai: bool = True,
    progress_callback=None,
) -> dict:
    """
    End-to-end: file path → structured output dict with AI summary.

    Parameters
    ----------
    file_path        : path to PDF, JPG, PNG, TIFF, etc.
    engine           : "easyocr" or "tesseract"
    language         : language for AI summary (e.g. "Hindi", "Gujarati", "English")
    with_ai          : whether to call Claude for the summary (default True)
    progress_callback: callable(page, total) for PDF progress bars
    """
    raw_text = extract_text(file_path, engine=engine, progress_callback=progress_callback)
    return _build_result(raw_text, engine, language=language, with_ai=with_ai)


def process_pasted_text(text: str, language: str = "English", with_ai: bool = True) -> dict:
    """
    End-to-end: pasted text string → structured output dict with AI summary.

    Parameters
    ----------
    text     : raw report text pasted by user
    language : language for AI summary
    with_ai  : whether to call Claude for the summary
    """
    return _build_result(text, "text_input", language=language, with_ai=with_ai)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    SAMPLE = """
    Apollo Hospitals — Diagnostic Laboratory
    Patient Name : Arjun Sharma
    Age : 52 Years    Sex: Male
    Date: 14 March 2025

    Haemoglobin          : 13.4 g/dL
    WBC (TLC)            : 8.2
    Platelets            : 210
    Total Cholesterol    : 268 mg/dL
    LDL Cholesterol      : 172 mg/dL
    HDL Cholesterol      : 38  mg/dL
    Triglycerides        : 210 mg/dL
    Blood Pressure       : 142/88 mmHg
    Heart Rate           : 94 bpm
    Fasting Blood Glucose: 118 mg/dL
    HbA1c                : 6.1 %
    Creatinine           : 1.1 mg/dL
    Uric Acid            : 7.8 mg/dL
    SGPT (ALT)           : 52  U/L
    TSH                  : 3.2 mIU/L
    """

    lang = sys.argv[1] if len(sys.argv) > 1 else "English"
    print(f"\n🔬 Processing sample report — AI summary in: {lang}\n")

    result = process_pasted_text(SAMPLE, language=lang)

    print(f"👤 Patient  : {result['patient_info']}")
    print(f"\n📋 Parsed ({len(result['parsed_values'])} parameters):")
    for k, v in result["parsed_values"].items():
        flag_icon = "⚠️" if v["flag"] not in ("Normal", "Optimal", "Protective") else "✅"
        print(f"  {flag_icon} {k:<28} {str(v['value'])+' '+v['unit']:<18} → {v['flag']}")

    print(f"\n🤖 AI Summary ({lang}):\n")
    print(result["ai_summary"] or "No AI summary generated.")