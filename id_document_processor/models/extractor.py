"""
Field Extractor — supports TWO modes:

Mode 1 — YOLO + OCR (model-based field detection)
    YOLOv8 detects field bounding boxes → OCR runs only on those regions.
    This is a true object-detection + OCR pipeline.

Mode 2 — OCR + Spatial Heuristics (fallback)
    Full-image OCR → label proximity + regex to identify fields.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

from utils.ocr_engine import OCRResult, OCREngine

logger = logging.getLogger(__name__)

# YOLO class indices → field names (matches YOLO training labels)
YOLO_CLASS_MAP = {
    0: "name",
    1: "dob",
    2: "pan_number",
    3: "aadhaar_number",
    4: "father_name",
    5: "gender",
    6: "address",
}


@dataclass
class ExtractionResult:
    fields: Dict[str, str]
    confidences: Dict[str, float]
    bboxes: Dict[str, Tuple[int, int, int, int]]
    all_ocr_results: List[OCRResult]
    extraction_mode: str  # "yolo_ocr" or "heuristic_ocr"


class FieldExtractor:
    """
    Extracts key fields from document images.

    extraction_mode:
        "auto"          — try YOLO first, fall back to heuristic
        "yolo_ocr"      — YOLO detection + OCR on detected regions only
        "heuristic_ocr" — full-image OCR + spatial rules
    """

    def __init__(self, ocr_engine: OCREngine, extraction_mode: str = "auto",
                 yolo_model_path: Optional[str] = None):
        self.ocr = ocr_engine
        self.extraction_mode = extraction_mode
        self.yolo_model = None
        self.yolo_model_path = yolo_model_path
        self._init_yolo()

    def _init_yolo(self):
        """Load YOLO model if available."""
        if self.extraction_mode == "heuristic_ocr":
            return
        try:
            from ultralytics import YOLO
            path = self.yolo_model_path or "yolov8n.pt"
            self.yolo_model = YOLO(path)
            logger.info(f"✅ YOLO model loaded: {path}")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO unavailable, using heuristic")
            self.extraction_mode = "heuristic_ocr"
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e} — using heuristic")
            self.extraction_mode = "heuristic_ocr"

    # ── Public entry point ───────────────────────────────────────────────
    def extract(self, image: np.ndarray, doc_type: str) -> ExtractionResult:
        mode = self._resolve_mode(doc_type)
        logger.info(f"Extraction mode: {mode}")

        if mode == "yolo_ocr":
            return self._extract_yolo(image, doc_type)
        return self._extract_heuristic(image, doc_type)

    def _resolve_mode(self, doc_type: str) -> str:
        if self.extraction_mode in ("yolo_ocr", "heuristic_ocr"):
            return self.extraction_mode
        # "auto" — use YOLO if model loaded
        if self.yolo_model is not None:
            return "yolo_ocr"
        return "heuristic_ocr"

    # ═══════════════════════════════════════════════════════════════════
    # MODE 1: YOLO + OCR  (Object Detection → Cropped OCR)
    # ═══════════════════════════════════════════════════════════════════
    def _extract_yolo(self, image: np.ndarray, doc_type: str) -> ExtractionResult:
        """
        Use YOLO to detect field regions, then run OCR on each crop.
        This is a genuine object-detection + recognition pipeline.
        """
        fields: Dict[str, str] = {}
        confs: Dict[str, float] = {}
        bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        all_ocr: List[OCRResult] = []

        # Run YOLO inference
        results = self.yolo_model(image, verbose=False)
        detections = results[0].boxes if results else None

        if detections is None or len(detections) == 0:
            logger.info("YOLO found no fields — falling back to heuristic")
            return self._extract_heuristic(image, doc_type)

        h, w = image.shape[:2]

        for det in detections:
            cls_id = int(det.cls[0])
            yolo_conf = float(det.conf[0])

            # Map YOLO class to field name
            fname = YOLO_CLASS_MAP.get(cls_id)
            if fname is None:
                continue

            # Filter: only extract fields relevant to doc_type
            if doc_type == "PAN" and fname in ("aadhaar_number", "gender", "address"):
                continue
            if doc_type == "Aadhaar" and fname == "pan_number":
                continue

            # Get bounding box (xyxy format)
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 20 or y2 - y1 < 10:
                continue  # skip tiny detections

            # Crop region and run OCR
            crop = image[y1:y2, x1:x2]
            crop_results = self.ocr.recognize(crop)
            all_ocr.extend(crop_results)

            if crop_results:
                # Use the highest-confidence result in the crop
                best = max(crop_results, key=lambda r: r.confidence)
                # Offset bbox back to full-image coordinates
                abs_bbox = (x1, y1, x2, y2)
                fields[fname] = best.text
                confs[fname] = round((yolo_conf + best.confidence) / 2, 2)
                bboxes[fname] = abs_bbox
            else:
                # YOLO detected region but OCR failed
                fields[fname] = ""
                confs[fname] = round(yolo_conf * 0.3, 2)
                bboxes[fname] = (x1, y1, x2, y2)

        return ExtractionResult(fields, confs, bboxes, all_ocr, "yolo_ocr")

    # ═══════════════════════════════════════════════════════════════════
    # MODE 2: Heuristic OCR  (Full-image OCR + spatial rules)
    # ═══════════════════════════════════════════════════════════════════
    def _extract_heuristic(self, image: np.ndarray, doc_type: str) -> ExtractionResult:
        results = self.ocr.recognize(image)
        if not results:
            return ExtractionResult({}, {}, {}, [], "heuristic_ocr")

        h, w = image.shape[:2]

        if doc_type == "PAN":
            flds, cf, bb = self._pan(results, w, h)
        elif doc_type == "Aadhaar":
            flds, cf, bb = self._aadhaar(results, w, h)
        else:
            flds, cf, bb = {}, {}, {}

        return ExtractionResult(flds, cf, bb, results, "heuristic_ocr")

    # ── PAN heuristic ────────────────────────────────────────────────────
    def _pan(self, res, w, h):
        fields, confs, bbs = {}, {}, {}

        # PAN number
        _re = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
        for r in res:
            m = _re.search(r.text.upper())
            if m:
                fields["pan_number"] = m.group(1)
                confs["pan_number"] = min(r.confidence + 0.10, 0.99)
                bbs["pan_number"] = r.bbox
                break
        if "pan_number" not in fields:
            self._pan_by_label(res, fields, confs, bbs)

        # Name
        nr = self._by_label(res, ["NAME", "Name", "name"],
                            exclude=["FATHER", "Father", "father"],
                            direction="right", img_w=w)
        if nr:
            fields["name"], confs["name"], bbs["name"] = nr.text, nr.confidence, nr.bbox

        # Father's name
        fr = self._by_label(res, ["FATHER'S NAME", "Father's Name", "FATHER NAME"],
                            direction="right", img_w=w)
        if fr:
            fields["father_name"] = fr.text
            confs["father_name"] = fr.confidence
            bbs["father_name"] = fr.bbox

        # DOB
        dob = self._find_dob(res)
        if dob:
            fields["dob"], confs["dob"], bbs["dob"] = dob

        return fields, confs, bbs

    def _pan_by_label(self, res, fields, confs, bbs):
        labels = ["PAN", "Permanent Account Number", "PAN No"]
        label_r = None
        for r in res:
            for lb in labels:
                if lb in r.text:
                    label_r = r; break
            if label_r:
                break
        if not label_r:
            return
        alnum = re.compile(r"[A-Z0-9]+")
        cands = []
        for r in res:
            if r is label_r:
                continue
            m = alnum.search(r.text.upper())
            if m and 8 <= len(m.group()) <= 12:
                yd = abs(r.center[1] - label_r.center[1])
                xd = abs(r.center[0] - label_r.center[0])
                if yd < 100 or xd < 400:
                    cands.append((yd + xd, r, m.group()))
        if cands:
            cands.sort(key=lambda x: x[0])
            _, r, txt = cands[0]
            fields["pan_number"] = txt
            confs["pan_number"] = r.confidence * 0.80
            bbs["pan_number"] = r.bbox

    # ── Aadhaar heuristic ────────────────────────────────────────────────
    def _aadhaar(self, res, w, h):
        fields, confs, bbs = {}, {}, {}

        # Aadhaar number
        _re = re.compile(r"(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})")
        for r in res:
            m = _re.search(r.text)
            if m:
                clean = m.group(1).replace(" ", "").replace("-", "")
                if len(clean) == 12:
                    fields["aadhaar_number"] = clean
                    confs["aadhaar_number"] = min(r.confidence + 0.10, 0.99)
                    bbs["aadhaar_number"] = r.bbox
                    break

        # Name
        nr = self._by_label(res, ["NAME", "Name", "name"],
                            exclude=["FATHER", "Father", "GUARDIAN"],
                            direction="right", img_w=w)
        if nr:
            fields["name"], confs["name"], bbs["name"] = nr.text, nr.confidence, nr.bbox

        # DOB
        dob = self._find_dob(res)
        if dob:
            fields["dob"], confs["dob"], bbs["dob"] = dob

        # Gender
        g = self._find_gender(res)
        if g:
            fields["gender"], confs["gender"], bbs["gender"] = g

        # Address
        addr = self._find_address(res, h)
        if addr:
            fields["address"], confs["address"], bbs["address"] = addr

        return fields, confs, bbs

    # ── Shared helpers ───────────────────────────────────────────────────
    def _by_label(self, res, labels, exclude=None, direction="right", img_w=0):
        exclude = exclude or []
        for lb in labels:
            for lr in res:
                if lb not in lr.text or any(kw in lr.text for kw in exclude):
                    continue
                cands = []
                for r in res:
                    if r is lr or any(kw in r.text for kw in exclude):
                        continue
                    if direction == "right" and r.center[0] > lr.center[0]:
                        yd = abs(r.center[1] - lr.center[1])
                        xd = r.center[0] - lr.center[0]
                        if yd < (lr.bbox[3] - lr.bbox[1]) + 30:
                            cands.append((xd + yd * 2, r))
                    elif direction == "below" and r.center[1] > lr.bbox[3]:
                        xd = abs(r.center[0] - lr.center[0])
                        yd = r.center[1] - lr.bbox[3]
                        if xd < 200:
                            cands.append((yd + xd * 2, r))
                if cands:
                    cands.sort(key=lambda x: x[0])
                    return cands[0][1]
        return None

    def _find_dob(self, res):
        pats = [
            re.compile(r"(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})"),
            re.compile(r"(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})"),
        ]
        for r in res:
            if any(k in r.text.lower() for k in ("dob", "date of birth", "d.o.b")):
                for other in res:
                    for p in pats:
                        m = p.search(other.text)
                        if m:
                            yd = abs(other.center[1] - r.center[1])
                            xd = abs(other.center[0] - r.center[0])
                            if yd < 60 or xd < 350:
                                return m.group(1), other.confidence, other.bbox
                for p in pats:
                    m = p.search(r.text)
                    if m:
                        return m.group(1), r.confidence, r.bbox
        for r in res:
            for p in pats:
                m = p.search(r.text)
                if m:
                    ds = m.group(1)
                    if any(y in ds for y in ("2024", "2025", "2023")):
                        continue
                    return ds, r.confidence, r.bbox
        return None

    def _find_gender(self, res):
        for r in res:
            t = r.text.upper().strip()
            if t in ("MALE", "FEMALE", "TRANSGENDER") and len(t) < 15:
                return t, r.confidence, r.bbox
        return None

    def _find_address(self, res, img_h):
        skip = {"aadhaar", "uidai", "government", "name", "dob",
                "date of birth", "gender", "male", "female",
                "unique identification", "authority"}
        label_r = next((r for r in res if "address" in r.text.lower()), None)
        lines = []
        for r in res:
            if any(k in r.text.lower() for k in skip):
                continue
            if label_r and r.center[1] > label_r.center[1]:
                lines.append(r)
            elif r.center[1] > img_h * 0.55:
                lines.append(r)
        if not lines:
            return None
        lines.sort(key=lambda r: r.center[1])
        chosen = lines[:5]
        addr = " ".join(r.text for r in chosen)
        x1 = min(r.bbox[0] for r in chosen)
        y1 = min(r.bbox[1] for r in chosen)
        x2 = max(r.bbox[2] for r in chosen)
        y2 = max(r.bbox[3] for r in chosen)
        avg_c = sum(r.confidence for r in chosen) / len(chosen)
        return addr, avg_c, (x1, y1, x2, y2)