"""
Minimal post-processing: regex validation + date normalisation only.
No heavy text cleanup — OCR output is trusted as-is.
"""

import re
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_RE_PAN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
_RE_AADHAAR = re.compile(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b")

_DOB_PATTERNS = [
    (re.compile(r"(\d{2})/(\d{2})/(\d{4})"), "dmy"),
    (re.compile(r"(\d{2})-(\d{2})-(\d{4})"), "dmy"),
    (re.compile(r"(\d{2})\.(\d{2})\.(\d{4})"), "dmy"),
    (re.compile(r"(\d{4})/(\d{2})/(\d{2})"), "ymd"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "ymd"),
]


def validate_pan(pan: str) -> Tuple[bool, float]:
    pan = pan.strip().upper().replace(" ", "").replace("-", "")
    if _RE_PAN.fullmatch(pan):
        return True, 0.99
    if len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[9].isalpha():
        return True, 0.85
    return False, 0.0


def validate_aadhaar(aadhaar: str) -> Tuple[bool, float]:
    aadhaar = aadhaar.strip().replace(" ", "").replace("-", "")
    if len(aadhaar) == 12 and aadhaar.isdigit():
        return True, 0.99
    return False, 0.0


def normalize_dob(raw: str) -> Tuple[Optional[str], float]:
    """Normalise DOB to YYYY-MM-DD."""
    for pat, fmt in _DOB_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        g = m.groups()
        try:
            if fmt == "dmy":
                d, mo, y = int(g[0]), int(g[1]), int(g[2])
            else:
                y, mo, d = int(g[0]), int(g[1]), int(g[2])
            if 1900 <= y <= 2025 and 1 <= mo <= 12 and 1 <= d <= 31:
                dt = datetime(y, mo, d)
                return dt.strftime("%Y-%m-%d"), 0.95
        except (ValueError, IndexError):
            continue
    return None, 0.0


def validate_and_format_fields(
    fields: Dict[str, str],
    ocr_confidences: Dict[str, float],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Minimal post-processing: validate format, normalise dates.
    Text from OCR is used as-is (no rewriting/cleaning).
    """
    formatted: Dict[str, str] = {}
    confidences: Dict[str, float] = {}

    for fname, raw in fields.items():
        if not raw or not raw.strip():
            continue
        ocr_c = ocr_confidences.get(fname, 0.5)

        if fname == "pan_number":
            cleaned = raw.strip().upper().replace(" ", "").replace("-", "")
            ok, vc = validate_pan(cleaned)
            if ok:
                formatted[fname] = cleaned
                confidences[fname] = round((ocr_c + vc) / 2, 2)

        elif fname == "aadhaar_number":
            cleaned = raw.strip().replace(" ", "").replace("-", "")
            ok, vc = validate_aadhaar(cleaned)
            if ok:
                formatted[fname] = cleaned
                confidences[fname] = round((ocr_c + vc) / 2, 2)

        elif fname == "dob":
            norm, nc = normalize_dob(raw)
            if norm:
                formatted[fname] = norm
                confidences[fname] = round((ocr_c + nc) / 2, 2)

        else:
            # Name, father_name, address, gender — pass through as-is
            formatted[fname] = raw.strip()
            confidences[fname] = round(ocr_c, 2)

    return formatted, confidences