"""
Configuration file for the Intelligent Document ID Processing System.
All tunable parameters are centralized here.
"""

import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models" / "weights"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Document types ───────────────────────────────────────────────────────
DOC_TYPES = ["Aadhaar", "PAN", "Unknown"]

# ── Field definitions per document type ───────────────────────────────────
FIELDS_CONFIG = {
    "Aadhaar": {
        "required": ["name", "dob", "aadhaar_number"],
        "optional": ["address", "gender"],
    },
    "PAN": {
        "required": ["name", "dob", "pan_number"],
        "optional": ["father_name"],
    },
}

# ── Regex patterns ───────────────────────────────────────────────────────
PATTERNS = {
    "pan_number":     r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "aadhaar_number": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "dob_dmy":        r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
    "dob_ymd":        r"\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b",
}

# ── Classification thresholds ────────────────────────────────────────────
CLASSIFICATION = {
    "min_confidence": 0.40,
    "unknown_threshold": 0.30,
}

# ── Image preprocessing defaults ─────────────────────────────────────────
PREPROCESSING = {
    "max_dimension": 2000,
    "denoise_strength": 10,
    "clahe_clip_limit": 2.0,
    "sharpen_strength": 0.8,
    "do_deskew": True,
}

# ── PaddleOCR settings ──────────────────────────────────────────────────
PADDLEOCR_CONFIG = {
    "lang": "en",
}

# ── EasyOCR settings ────────────────────────────────────────────────────
EASYOCR_CONFIG = {
    "langs": ["en"],
    "gpu": False,
    "verbose": False,
}

# ── CNN classifier settings ──────────────────────────────────────────────
# ── Classifier settings ──────────────────────────────────────────────────
CLASSIFIER_CONFIG = {
    "input_size": (224, 224),
    "cnn_model_path": str(MODEL_DIR / "document_classifier.keras"),
    "vit_model_path": str(MODEL_DIR / "vit_classifier"),
}