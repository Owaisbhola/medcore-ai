import re, os, json
from datetime import datetime

# ── Anthropic ─────────────────────────────────────────────────────────────────
try:
    import anthropic
    _client: anthropic.Anthropic | None = None
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("⚠  anthropic not installed  →  pip install anthropic")


def _get_client() -> "anthropic.Anthropic":
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


# ── Groq (Ultra-fast cloud Llama-3, free tier) ─────────────────────────────────
GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            api_k = os.environ.get("GROQ_API_KEY")
            if api_k:
                _groq_client = Groq(api_key=api_k)
        except ImportError:
            _groq_client = None
    return _groq_client


def groq_is_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


# ── Ollama (free, local, no API key) ───────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

def get_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

OLLAMA_HOST = get_ollama_host()
OLLAMA_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "MedCoreAI/1.0",
}


def ollama_is_running(host: str | None = None, timeout: float = 4.0) -> bool:
    """Quick check whether an Ollama server is up (local or ngrok remote)."""
    if not REQUESTS_AVAILABLE:
        return False
    target = (host or get_ollama_host()).rstrip("/")
    try:
        r = requests.get(f"{target}/api/tags", headers=OLLAMA_HEADERS, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return isinstance(data, dict) and "models" in data
        return False
    except Exception:
        return False


def ollama_list_models(host: str | None = None) -> list:
    """Return names of models currently pulled in the Ollama install."""
    if not REQUESTS_AVAILABLE:
        return []
    target = (host or get_ollama_host()).rstrip("/")
    try:
        r = requests.get(f"{target}/api/tags", headers=OLLAMA_HEADERS, timeout=6.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_PATTERNS = {
    "header": [
        r"patient\s*(name|info|details)",
        r"(lab|laboratory|diagnostic)\s*report",
        r"hospital|clinic|diagnostic\s*centre",
    ],
    "lipid_panel": [
        r"lipid\s*(profile|panel)",
        r"cholesterol",
        r"triglyceride",
        r"hdl|ldl|vldl",
    ],
    "cbc": [
        r"complete\s*blood\s*(count|picture)",
        r"haemoglobin|hemoglobin",
        r"wbc|rbc|platelets",
        r"blood\s*count",
    ],
    "cardiac": [
        r"cardiac|cardiology|heart",
        r"troponin|bnp|nt.?probnp",
        r"ecg|ekg|electrocardiogram",
        r"blood\s*pressure|heart\s*rate",
    ],
    "metabolic": [
        r"metabolic|biochemistry|chemistry",
        r"glucose|hba1c|a1c",
        r"creatinine|urea|bun",
        r"uric\s*acid|electrolyte",
    ],
    "liver": [
        r"liver\s*(function|panel|profile)",
        r"sgot|sgpt|alt|ast",
        r"bilirubin|albumin",
        r"alkaline\s*phosphatase",
    ],
    "thyroid": [
        r"thyroid\s*(function|profile|panel)",
        r"\btsh\b|\bt3\b|\bt4\b",
        r"thyroxine|triiodothyronine",
    ],
    "kidney": [
        r"kidney\s*(function|profile)",
        r"renal\s*(function|panel)",
        r"egfr|glomerular",
        r"microalbumin|proteinuria",
    ],
    "vitamins": [
        r"vitamin\s*(b12|d|d3|b9|folate)",
        r"25.?oh.?vitd|25.?hydroxy",
        r"folate|folic\s*acid",
        r"ferritin|iron\s*studies",
    ],
    "urine": [
        r"urine\s*(analysis|routine|r/e|re\b)",
        r"urinalysis|uroscopy",
        r"specific\s*gravity",
        r"urine\s*glucose|urine\s*protein",
    ],
    "radiology": [
        r"x.?ray|radiograph",
        r"ultrasound|sonograph|usg\b",
        r"ct\s*scan|mri\b|pet\s*scan",
        r"impression|findings|conclusion",
    ],
    "oncology": [
        r"tumou?r\s*marker",
        r"oncolog(?:y|ist)",
        r"\bpsa\b|prostate\s*specific\s*antigen",
        r"ca[\s\-]?125|ca[\s\-]?19[\s\-]?9|ca[\s\-]?15[\s\-]?3",
        r"\bcea\b|carcinoembryonic",
        r"\bafp\b|alpha.?fetoprotein",
        r"biopsy|carcinoma|malignan(?:t|cy)",
    ],
}


def detect_sections(text: str) -> dict:
    """Split raw text into labelled sections."""
    lines   = text.split("\n")
    sections = {k: [] for k in SECTION_PATTERNS}
    sections["other"] = []
    current = "other"

    for line in lines:
        line_lower = line.lower()
        matched = False
        for section, patterns in SECTION_PATTERNS.items():
            if any(re.search(p, line_lower) for p in patterns):
                current = section
                matched = True
                break
        sections[current].append(line)

    return {
        k: "\n".join(v).strip()
        for k, v in sections.items()
        if "\n".join(v).strip()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find(patterns: list, text: str) -> float | None:
    """Run a list of regex patterns, return first numeric match."""
    text_lower = text.lower()
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def parse_lipid_panel(text: str) -> dict:
    return {
        "total_cholesterol": _find([r"(?:total\s+)?cholesterol(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"t\.?\s*chol(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ldl":               _find([r"ldl(?:\s*cholesterol|-c)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"low\s+density\s+lipoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "hdl":               _find([r"hdl(?:\s*cholesterol|-c)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"high\s+density\s+lipoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "triglycerides":     _find([r"triglycerides?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\btg\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "vldl":              _find([r"vldl(?:\s*cholesterol)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_cbc(text: str) -> dict:
    return {
        "haemoglobin":       _find([r"ha?emoglobin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bhb\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"hgb(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "wbc":               _find([r"(?:wbc|white\s+blood\s+cell|tlc)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "rbc":               _find([r"(?:rbc|red\s+blood\s+cell)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "platelets":         _find([r"platelet(?:s)?\s*(?:count)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bplt\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "packed_cell_volume":_find([r"(?:pcv|hematocrit|haematocrit)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "mcv":               _find([r"\bmcv\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "mch":               _find([r"\bmch\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_cardiac(text: str) -> dict:
    return {
        "systolic_bp":    _find([r"(?:blood\s*pressure|b\.?p\.?|systolic)(?:\s*[:=\-]\s*|\s+)(\d{2,3})\s*/\s*\d+", r"sbp(?:\s*[:=\-]\s*|\s+)(\d{2,3})"], text),
        "diastolic_bp":   _find([r"(?:blood\s*pressure|b\.?p\.?)(?:\s*[:=\-]\s*|\s+)\d+\s*/\s*(\d{2,3})", r"dbp(?:\s*[:=\-]\s*|\s+)(\d{2,3})"], text),
        "heart_rate":     _find([r"(?:heart\s+rate|h\.?r\.?|pulse)(?:\s*[:=\-]\s*|\s+)(\d+)"], text),
        "troponin_i":     _find([r"troponin\s*[-\s]?i(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"ctni(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "bnp":            _find([r"(?:bnp|b-type\s+natriuretic)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "creatine_kinase":_find([r"(?:creatine\s+kinase|ck|cpk)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_metabolic(text: str) -> dict:
    return {
        "fasting_glucose": _find([r"(?:fasting\s+(?:blood\s+)?glucose|fbg|fbs)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"glucose\s*(?:\(fasting\))?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "hba1c":           _find([r"hba1c(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)\s*%?", r"a1c(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "creatinine":      _find([r"(?:serum\s+)?creatinine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"s\.?\s*creat(?:inine)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "urea":            _find([r"(?:blood\s+urea|serum\s+urea|urea)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bbun\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "uric_acid":       _find([r"uric\s+acid(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"s\.?\s*urate(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "sodium":          _find([r"(?:serum\s+)?sodium(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bna\+?\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "potassium":       _find([r"(?:serum\s+)?potassium(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bk\+?\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_liver(text: str) -> dict:
    return {
        "sgpt_alt":             _find([r"(?:sgpt|alt|alanine\s+aminotransferase)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "sgot_ast":             _find([r"(?:sgot|ast|aspartate\s+aminotransferase)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "total_bilirubin":      _find([r"(?:total\s+)?bilirubin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"t\.?\s*bili(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "albumin":              _find([r"albumin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "alkaline_phosphatase": _find([r"(?:alkaline\s+phosphatase|alp|alkphos)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ggt":                  _find([r"(?:ggt|gamma\s*gt|gamma.glutamyl)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_thyroid(text: str) -> dict:
    return {
        "tsh": _find([r"\btsh\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"thyroid\s+stimulating\s+hormone(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "t3":  _find([r"\bt3\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"triiodothyronine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "t4":  _find([r"\bt4\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"thyroxine(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_kidney(text: str) -> dict:
    return {
        "egfr":          _find([r"(?:egfr|estimated\s+gfr)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "microalbumin":  _find([r"microalbumin(?:uria)?(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "urine_protein": _find([r"urine\s+protein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_vitamins(text: str) -> dict:
    return {
        "vitamin_b12": _find([r"(?:vitamin\s+b.?12|cobalamin)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "vitamin_d":   _find([r"(?:vitamin\s+d(?:3)?|25.?oh.?vitd|25.?hydroxy)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "folate":      _find([r"(?:folate|folic\s+acid)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ferritin":    _find([r"ferritin(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "iron":        _find([r"(?:serum\s+iron|s\.?\s*iron)(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


def parse_oncology(text: str) -> dict:
    return {
        "psa":      _find([r"(?:total\s+)?psa(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"prostate\s+specific\s+antigen(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ca125":    _find([r"ca[\s\-]?125(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ca19_9":   _find([r"ca[\s\-]?19[\s\-]?9(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "ca15_3":   _find([r"ca[\s\-]?15[\s\-]?3(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "cea":      _find([r"\bcea\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"carcinoembryonic\s+antigen(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "afp":      _find([r"\bafp\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"alpha[\s\-]?fetoprotein(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
        "beta_hcg": _find([r"beta[\s\-]?hcg(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)", r"\bhcg\b(?:\s*[:=\-]\s*|\s+)(\d+\.?\d*)"], text),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NORMAL RANGE CHECKER  (expanded)
# ═══════════════════════════════════════════════════════════════════════════════

NORMAL_RANGES = {
    # Lipid
    "total_cholesterol":    (0,    200,   "mg/dL",           "< 200"),
    "ldl":                  (0,    100,   "mg/dL",           "< 100"),
    "hdl":                  (40,   9999,  "mg/dL",           "> 40"),
    "triglycerides":        (0,    150,   "mg/dL",           "< 150"),
    # CBC
    "haemoglobin":          (12.0, 17.5,  "g/dL",            "12.0–17.5"),
    "wbc":                  (4.0,  11.0,  "×10³/µL",         "4.0–11.0"),
    "rbc":                  (4.2,  5.9,   "×10⁶/µL",         "4.2–5.9"),
    "platelets":            (150,  400,   "×10³/µL",         "150–400"),
    "packed_cell_volume":   (36,   52,    "%",               "36–52"),
    "mcv":                  (80,   100,   "fL",              "80–100"),
    "mch":                  (27,   33,    "pg",              "27–33"),
    # Cardiac
    "systolic_bp":          (0,    120,   "mmHg",            "< 120"),
    "diastolic_bp":         (0,    80,    "mmHg",            "< 80"),
    "heart_rate":           (60,   100,   "bpm",             "60–100"),
    "troponin_i":           (0,    0.04,  "ng/mL",           "< 0.04"),
    "bnp":                  (0,    100,   "pg/mL",           "< 100"),
    # Metabolic
    "fasting_glucose":      (70,   99,    "mg/dL",           "70–99"),
    "hba1c":                (0,    5.6,   "%",               "< 5.7"),
    "creatinine":           (0.7,  1.2,   "mg/dL",           "0.7–1.2"),
    "urea":                 (7,    20,    "mg/dL",           "7–20"),
    "uric_acid":            (3.5,  7.2,   "mg/dL",           "3.5–7.2"),
    "sodium":               (136,  145,   "mEq/L",           "136–145"),
    "potassium":            (3.5,  5.0,   "mEq/L",           "3.5–5.0"),
    # Liver
    "sgpt_alt":             (0,    40,    "U/L",             "< 40"),
    "sgot_ast":             (0,    40,    "U/L",             "< 40"),
    "total_bilirubin":      (0,    1.2,   "mg/dL",           "< 1.2"),
    "albumin":              (3.5,  5.0,   "g/dL",            "3.5–5.0"),
    "alkaline_phosphatase": (44,   147,   "U/L",             "44–147"),
    "ggt":                  (0,    55,    "U/L",             "< 55"),
    # Thyroid
    "tsh":                  (0.4,  4.0,   "mIU/L",           "0.4–4.0"),
    "t3":                   (80,   200,   "ng/dL",           "80–200"),
    "t4":                   (5.1,  14.1,  "µg/dL",           "5.1–14.1"),
    # Kidney
    "egfr":                 (60,   999,   "mL/min/1.73m²",   "> 60"),
    # Vitamins
    "vitamin_b12":          (200,  900,   "pg/mL",           "200–900"),
    "vitamin_d":            (30,   100,   "ng/mL",           "30–100"),
    "folate":               (3.0,  20.0,  "ng/mL",           "3.0–20.0"),
    "ferritin":             (12,   300,   "ng/mL",           "12–300"),
    "iron":                 (60,   170,   "µg/dL",           "60–170"),
    # Oncology / tumour markers
    "psa":                  (0,    4.0,   "ng/mL",           "< 4.0"),
    "ca125":                (0,    35,    "U/mL",            "< 35"),
    "ca19_9":               (0,    37,    "U/mL",            "< 37"),
    "ca15_3":               (0,    30,    "U/mL",            "< 30"),
    "cea":                  (0,    3.0,   "ng/mL",           "< 3.0"),
    "afp":                  (0,    10,    "ng/mL",           "< 10"),
    "beta_hcg":             (0,    5,     "mIU/mL",          "< 5"),
}

# Severity weights for sorting (lower = more urgent)
_SEVERITY = {
    "Critical":     0,
    "Critical Low": 1,
    "High":         2,
    "Low":          3,
    "Borderline":   4,
    "Elevated":     5,
    "Normal":       99,
    "Unknown":      100,
}


def check_normal_range(key: str, value: float) -> dict:
    """Check a single value against normal ranges."""
    if key not in NORMAL_RANGES:
        return {"key": key, "value": value, "flag": "Unknown", "status": "unknown",
                "unit": "", "normal": "N/A"}

    low, high, unit, normal_str = NORMAL_RANGES[key]

    if value < low:
        deviation = (low - value) / low * 100
        flag   = "Critical Low" if deviation > 30 else "Low"
        status = "abnormal"
    elif value > high:
        deviation = (value - high) / high * 100
        flag   = "Critical" if deviation > 50 else "High"
        status = "abnormal"
    else:
        flag   = "Normal"
        status = "normal"

    return {
        "key":    key,
        "value":  value,
        "unit":   unit,
        "normal": normal_str,
        "flag":   flag,
        "status": status,
    }


def collect_abnormal_flags(all_values: dict) -> list:
    """Return only abnormal values, sorted by severity."""
    flags = []
    for key, value in all_values.items():
        if value is None:
            continue
        result = check_normal_range(key, value)
        if result["status"] == "abnormal":
            flags.append(result)
    flags.sort(key=lambda x: _SEVERITY.get(x["flag"], 9))
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT INFO PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_patient_info(text: str) -> dict:
    """Extract name, age, sex, date, doctor, lab from report header."""
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
        info["sex"] = "M" if sex.group(1).lower().startswith("m") else "F"

    date = re.search(
        r"(?:date|collected\s*on|reported\s*on|sample\s*date)\s*[:\-]\s*(\d{1,2}[\s\/\-\.]\w{2,9}[\s\/\-\.]\d{2,4})",
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
# HEART MODEL PRE-FILL MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def build_heart_prefill(parsed: dict) -> dict:
    """Map structured parsed values → heart model input feature names."""
    prefill  = {}
    lipid    = parsed.get("lipid_panel", {})
    cardiac  = parsed.get("cardiac", {})
    metabolic = parsed.get("metabolic", {})

    if lipid.get("total_cholesterol") is not None:
        prefill["chol"] = lipid["total_cholesterol"]
    if cardiac.get("systolic_bp") is not None:
        prefill["trestbps"] = cardiac["systolic_bp"]
    if cardiac.get("heart_rate") is not None:
        prefill["thalach"] = cardiac["heart_rate"]
    if metabolic.get("fasting_glucose") is not None:
        prefill["fbs"] = 1 if metabolic["fasting_glucose"] > 120 else 0

    return prefill


def build_cancer_prefill(parsed: dict) -> dict:
    """Map structured parsed values → cancer model input feature names."""
    onco = parsed.get("oncology", {})
    return {k: v for k, v in onco.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT TYPE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_TYPE_KEYWORDS = {
    "cancer": [
        "psa", "ca-125", "ca 125", "ca125", "ca 19-9", "ca19-9", "ca 15-3",
        "ca15-3", "cea", "afp", "tumour", "tumor", "oncology", "biopsy",
        "carcinoma", "malignan",
    ],
    "heart": [
        "cardiac", "cardiology", "ecg", "ekg", "electrocardiogram", "troponin",
        "bnp", "echocardiogram", "angiogram", "lipid profile", "coronary",
    ],
    "diabetes": [
        "glucose", "hba1c", "a1c", "diabetic", "diabetes", "insulin",
    ],
}


def detect_report_type(text: str, sections: dict | None = None) -> str:
    """
    Best-effort classification of a report as 'cancer', 'heart', 'diabetes',
    or 'general'. Uses keyword hits plus any detected sections. A report can
    legitimately contain markers for more than one condition — this label is
    used to pick which model/summary framing to lead with, not to drop data.
    """
    text_lower = text.lower()
    scores = {kind: 0 for kind in _REPORT_TYPE_KEYWORDS}
    for kind, keywords in _REPORT_TYPE_KEYWORDS.items():
        scores[kind] += sum(1 for kw in keywords if kw in text_lower)
    if sections:
        if "oncology" in sections:
            scores["cancer"] += 2
        if "cardiac" in sections:
            scores["heart"] += 2
    best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_kind if best_score > 0 else "general"


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX PLAIN-ENGLISH SUMMARY  (kept for offline/fallback use)
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(patient_info: dict, all_values: dict, abnormal_flags: list) -> str:
    """Concise plain-English paragraph summarising the report (regex-based, no AI)."""
    parts = []

    name = patient_info.get("name", "Patient")
    age  = patient_info.get("age")
    sex  = patient_info.get("sex")
    date = patient_info.get("report_date")

    patient_str = f"Report for {name}"
    if age:
        patient_str += f" ({age} years"
        if sex:
            patient_str += f", {'Male' if sex == 'M' else 'Female'}"
        patient_str += ")"
    if date:
        patient_str += f" dated {date}"
    parts.append(patient_str + ".")

    if not abnormal_flags:
        parts.append("All tested values are within normal limits.")
    else:
        critical = [f for f in abnormal_flags if "critical" in f["flag"].lower()]
        parts.append(f"{len(abnormal_flags)} abnormal value(s) detected.")
        for flag in abnormal_flags[:3]:
            key_label = flag["key"].replace("_", " ").title()
            parts.append(
                f"{key_label} is {flag['flag']} at {flag['value']} {flag['unit']} "
                f"(normal: {flag['normal']})."
            )
        if critical:
            parts.append(f"⚠ {len(critical)} critical value(s) require immediate attention.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# AI FULL SUMMARY  (Claude — any language)
# ═══════════════════════════════════════════════════════════════════════════════

_AI_SYSTEM_PARSER = """You are a compassionate AI medical interpreter for MedCore Health.
You receive a fully parsed medical report and must explain it in {language}.

Your summary must:
1. Be entirely in {language} — use common words a layperson understands.
2. Open with an "Overall verdict" sentence (e.g. "Your report mostly looks good, but there are a few things that need attention.")
3. Group findings into sections using simple headings:
   - 🔴 Things that need attention  (critical / high / low flags)
   - 🟡 Values to watch             (borderline / elevated)
   - ✅ Good news                   (normal values, briefly)
4. For each abnormal finding:
   a. Say what the test measures in one plain sentence.
   b. State what the patient's result means (too high / too low / why it matters).
   c. Give ONE practical tip the patient can act on.
5. End with a "Next Steps" list of 2–4 actions.
6. Final line: "Please share this report with your doctor for proper treatment."
7. Tone: warm, clear, reassuring — never alarming.
8. Do NOT diagnose or recommend specific medications.
"""

_AI_USER_PARSER = """Patient Details: {patient_info}

Detected Lab Values:
{all_values_block}

Abnormal Flags (sorted by severity):
{abnormal_block}

Report sections found: {sections_found}

Write a complete plain-language summary in {language}."""


def _build_summary_prompt(parsed_result: dict, language: str = "English"):
    """Shared prompt builder used by both the Claude and Ollama summary backends.
    Returns (system, user, all_values) — all_values is empty if nothing was parsed."""
    all_values     = parsed_result.get("all_values", {})
    abnormal_flags = parsed_result.get("abnormal_flags", [])
    patient_info   = parsed_result.get("patient_info", {})
    sections       = list(parsed_result.get("sections", {}).keys())

    if not all_values:
        return None, None, all_values

    av_lines = [
        f"  • {k.replace('_',' ').title()}: {v}"
        for k, v in all_values.items() if v is not None
    ]
    ab_lines = [
        f"  ⚠ {f['key'].replace('_',' ').title()}: {f['value']} {f['unit']} "
        f"(normal {f['normal']}) → {f['flag']}"
        for f in abnormal_flags
    ] or ["  None — all values normal"]

    patient_str = ", ".join(f"{k}: {v}" for k, v in patient_info.items()) or "Not provided"

    system = _AI_SYSTEM_PARSER.replace("{language}", language)
    user   = _AI_USER_PARSER.format(
        patient_info      = patient_str,
        all_values_block  = "\n".join(av_lines),
        abnormal_block    = "\n".join(ab_lines),
        sections_found    = ", ".join(sections) or "general",
        language          = language,
    )
    return system, user, all_values


def ai_full_summary(
    parsed_result: dict,
    language: str = "English",
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Generate a full AI-powered plain-language summary of a parsed report using
    the Anthropic API (paid, requires ANTHROPIC_API_KEY).

    Parameters
    ----------
    parsed_result : dict returned by parse_full_report()
    language      : e.g. "Hindi", "Gujarati", "Spanish", "Tamil", "English"
    model         : Claude model string

    Returns
    -------
    Formatted string — patient-friendly summary in the chosen language.
    """
    if not CLAUDE_AVAILABLE:
        return "AI summary unavailable — pip install anthropic"

    system, user, all_values = _build_summary_prompt(parsed_result, language)
    if not all_values:
        return "No lab values detected. Please verify the report and try again."

    try:
        response = _get_client().messages.create(
            model      = model,
            max_tokens = 2000,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return response.content[0].text
    except Exception as e:
        return f"AI summary failed: {e}"


def ollama_full_summary(
    parsed_result: dict,
    language: str = "English",
    model: str = "llama3.1",
    host: str = OLLAMA_HOST,
    timeout: int = 120,
) -> str:
    """
    Generate a full plain-language summary using a locally-running Ollama model —
    free, no API key, runs entirely on your own machine.

    Prerequisites
    -------------
    1. Install Ollama: https://ollama.com/download
    2. Pull a model once:  ollama pull llama3.1   (or mistral, qwen2.5, phi3, etc.)
    3. Make sure the Ollama server is running (it starts automatically on most installs;
       otherwise run `ollama serve`).

    Parameters
    ----------
    parsed_result : dict returned by parse_full_report()
    language      : e.g. "Hindi", "Gujarati", "Spanish", "Tamil", "English"
    model         : any model name you've pulled with `ollama pull <name>`
    host          : Ollama server URL, defaults to http://localhost:11434
                    (override with the OLLAMA_HOST env var for a remote/Docker instance)

    Returns
    -------
    Formatted string — patient-friendly summary in the chosen language.
    """
    if not REQUESTS_AVAILABLE:
        return "Ollama summary unavailable — pip install requests"

    system, user, all_values = _build_summary_prompt(parsed_result, language)
    if not all_values:
        return "No lab values detected. Please verify the report and try again."

    try:
        resp = requests.post(
            f"{host}/api/chat",
            headers=OLLAMA_HEADERS,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            f"Could not reach Ollama at {host}. Make sure it's installed and running "
            f"(`ollama serve`), and that you've pulled the model with `ollama pull {model}`."
        )
    except Exception as e:
        return f"Ollama summary failed: {e}"


def groq_full_summary(
    parsed_result: dict,
    language: str = "English",
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate an ultra-fast plain-language summary using Groq Cloud API (Free Meta Llama 3).
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return "Groq API key not set — add GROQ_API_KEY to your environment variables."

    system, user, all_values = _build_summary_prompt(parsed_result, language)
    if not all_values:
        return "No lab values detected. Please verify the report and try again."

    client = _get_groq_client()
    if client:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=900,
            )
            return res.choices[0].message.content
        except Exception:
            pass

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 900,
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq summary failed: {e}"





def ai_full_summary_stream(
    parsed_result: dict,
    language: str = "English",
    model: str = "claude-sonnet-4-6",
):
    """
    Streaming version for Streamlit.

    Usage:
        summary = st.write_stream(ai_full_summary_stream(result, language="Hindi"))
    """
    if not CLAUDE_AVAILABLE:
        yield "AI summary unavailable — pip install anthropic"
        return

    all_values     = parsed_result.get("all_values", {})
    abnormal_flags = parsed_result.get("abnormal_flags", [])
    patient_info   = parsed_result.get("patient_info", {})
    sections       = list(parsed_result.get("sections", {}).keys())

    if not all_values:
        yield "No lab values detected in the report."
        return

    av_lines = [f"  • {k.replace('_',' ').title()}: {v}" for k, v in all_values.items() if v is not None]
    ab_lines = [
        f"  ⚠ {f['key'].replace('_',' ').title()}: {f['value']} {f['unit']} (normal {f['normal']}) → {f['flag']}"
        for f in abnormal_flags
    ] or ["  None — all values normal"]

    patient_str = ", ".join(f"{k}: {v}" for k, v in patient_info.items()) or "Not provided"

    system = _AI_SYSTEM_PARSER.replace("{language}", language)
    user   = _AI_USER_PARSER.format(
        patient_info     = patient_str,
        all_values_block = "\n".join(av_lines),
        abnormal_block   = "\n".join(ab_lines),
        sections_found   = ", ".join(sections) or "general",
        language         = language,
    )

    try:
        with _get_client().messages.stream(
            model      = model,
            max_tokens = 2000,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\nError generating summary: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT CARD  (ready-to-render dict for Streamlit)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report_card(parsed_result: dict, language: str = "English") -> dict:
    """
    Returns a display-ready summary card dict for Streamlit.

    {
        "patient_info":    {...},
        "total_found":     int,
        "total_abnormal":  int,
        "critical_count":  int,
        "top_abnormals":   [...],   # top 5 sorted by severity
        "sections_found":  [...],
        "ai_summary":      str,     # Claude summary in chosen language
        "generated_at":    str,
    }
    """
    abnormal = parsed_result.get("abnormal_flags", [])
    critical = [f for f in abnormal if "critical" in f["flag"].lower()]

    ai_sum = ai_full_summary(parsed_result, language=language)

    return {
        "patient_info":   parsed_result.get("patient_info", {}),
        "total_found":    len(parsed_result.get("all_values", {})),
        "total_abnormal": len(abnormal),
        "critical_count": len(critical),
        "top_abnormals":  abnormal[:5],
        "sections_found": list(parsed_result.get("sections", {}).keys()),
        "ai_summary":     ai_sum,
        "generated_at":   datetime.now().strftime("%d %b %Y %H:%M"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def parse_full_report(raw_text: str) -> dict:
    """
    Full structured parser — raw OCR text → everything.

    Returns
    -------
    {
        patient_info, sections, parsed, all_values,
        abnormal_flags, prefill_heart, summary, parsed_at
    }
    """
    sections  = detect_sections(raw_text)
    full_text = raw_text

    lipid     = parse_lipid_panel(full_text)
    cbc       = parse_cbc(full_text)
    cardiac   = parse_cardiac(full_text)
    metabolic = parse_metabolic(full_text)
    liver     = parse_liver(full_text)
    thyroid   = parse_thyroid(full_text)
    kidney    = parse_kidney(full_text)
    vitamins  = parse_vitamins(full_text)
    oncology  = parse_oncology(full_text)

    parsed = {
        "lipid_panel": lipid,
        "cbc":         cbc,
        "cardiac":     cardiac,
        "metabolic":   metabolic,
        "liver":       liver,
        "thyroid":     thyroid,
        "kidney":      kidney,
        "vitamins":    vitamins,
        "oncology":    oncology,
    }

    # Flatten all values
    all_values: dict = {}
    for section_dict in parsed.values():
        all_values.update({k: v for k, v in section_dict.items() if v is not None})

    abnormal_flags = collect_abnormal_flags(all_values)
    prefill_heart  = build_heart_prefill(parsed)
    prefill_cancer = build_cancer_prefill(parsed)
    patient_info   = parse_patient_info(raw_text)
    report_type    = detect_report_type(raw_text, sections)
    summary        = build_summary(patient_info, all_values, abnormal_flags)

    return {
        "patient_info":   patient_info,
        "sections":       sections,
        "report_type":    report_type,
        "parsed":         parsed,
        "all_values":     all_values,
        "abnormal_flags": abnormal_flags,
        "prefill_heart":  prefill_heart,
        "prefill_cancer": prefill_cancer,
        "summary":        summary,
        "parsed_at":      datetime.now().strftime("%d %b %Y %H:%M"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    SAMPLE = """
    Apollo Hospitals — Diagnostic Laboratory
    Patient Name : Arjun Sharma
    Age          : 52 Years       Sex: Male
    Date         : 14 March 2025
    Referring Dr.: Dr. Priya Nair

    LIPID PROFILE
    Total Cholesterol    : 268 mg/dL
    LDL Cholesterol      : 172 mg/dL
    HDL Cholesterol      : 38  mg/dL
    Triglycerides        : 210 mg/dL
    VLDL                 : 42  mg/dL

    COMPLETE BLOOD COUNT
    Haemoglobin          : 13.4 g/dL
    WBC (TLC)            : 8.2 × 10³/µL
    Platelets            : 210 × 10³/µL

    CARDIAC & METABOLIC
    Blood Pressure       : 142/88 mmHg
    Heart Rate           : 94 bpm
    Fasting Blood Glucose: 118 mg/dL
    HbA1c                : 6.1 %
    Creatinine           : 1.1 mg/dL
    Uric Acid            : 7.8 mg/dL
    SGPT (ALT)           : 52 U/L
    TSH                  : 3.2 mIU/L
    Vitamin D            : 18 ng/mL
    Vitamin B12          : 185 pg/mL
    """

    lang = sys.argv[1] if len(sys.argv) > 1 else "English"
    print(f"\n🔬 MedCore Report Parser v2  |  AI Summary language: {lang}\n")

    result = parse_full_report(SAMPLE)

    print(f"👤 Patient  : {result['patient_info']}")
    print(f"\n⚠  Abnormal Flags ({len(result['abnormal_flags'])}):")
    for f in result["abnormal_flags"]:
        print(f"  [{f['flag']:14}]  {f['key'].replace('_',' ').title():<28}"
              f" {str(f['value'])+' '+f['unit']:<20}  normal: {f['normal']}")

    print(f"\n🤖 AI Summary ({lang}):\n")
    print(ai_full_summary(result, language=lang))