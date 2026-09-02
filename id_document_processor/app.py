"""
Intelligent Document ID Processing System – Streamlit Front-End
================================================================
Run:  streamlit run app.py
"""

import json
import sys
import time
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ── Ensure project root is on sys.path ───────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PREPROCESSING, CLASSIFIER_CONFIG, DOC_TYPES
from utils.preprocessing import (
    load_image_from_bytes,
    preprocess_document,
    pdf_to_images,
    cv2_to_pil,
)
from utils.postprocessing import validate_and_format_fields
from utils.ocr_engine import OCREngine
from models.classifier import DocumentClassifier
from models.extractor import FieldExtractor

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ID Document Processor",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .card { border-radius: 12px; padding: 20px; margin-bottom: 16px;
            background: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .badge { display: inline-block; padding: 4px 14px; border-radius: 999px;
             font-weight: 600; font-size: 0.85rem; }
    .badge-pan  { background: #dbeafe; color: #1e40af; }
    .badge-aadhaar { background: #d1fae5; color: #065f46; }
    .badge-unknown { background: #f3f4f6; color: #6b7280; }
    .conf-bar { height: 8px; border-radius: 4px; background: #e5e7eb; }
    .conf-fill { height: 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource initialisation ───────────────────────────────────────
# ── Cached resource initialisation ───────────────────────────────────────
@st.cache_resource(show_spinner="Initialising OCR engine …")
def get_ocr(engine_name: str) -> OCREngine:
    return OCREngine(engine_name)


@st.cache_resource(show_spinner="Loading classifier …")
def get_classifier() -> DocumentClassifier:
    return DocumentClassifier(
        vit_model_path=CLASSIFIER_CONFIG["vit_model_path"],
        cnn_model_path=CLASSIFIER_CONFIG["cnn_model_path"],
    )

# ── Visualisation helpers ────────────────────────────────────────────────
FIELD_COLORS = {
    "pan_number":     (0, 200, 0),
    "aadhaar_number": (0, 200, 0),
    "name":           (255, 80, 80),
    "dob":            (0, 165, 255),
    "father_name":    (255, 0, 255),
    "address":        (0, 220, 220),
    "gender":         (0, 255, 180),
}

def draw_bboxes(
    img: np.ndarray,
    bboxes: dict,
    fields: dict,
) -> np.ndarray:
    """Draw coloured bounding boxes with labels on a copy of the image."""
    out = img.copy()
    for fname, bb in bboxes.items():
        colour = FIELD_COLORS.get(fname, (180, 180, 180))
        x1, y1, x2, y2 = bb
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{fname}: {fields.get(fname, '')}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def conf_bar_html(value: float) -> str:
    """Return an HTML confidence bar."""
    pct = int(value * 100)
    colour = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
    return (
        f'<div class="conf-bar">'
        f'<div class="conf-fill" style="width:{pct}%;background:{colour}"></div>'
        f'</div>'
        f'<span style="font-size:0.78rem;color:#6b7280">{pct}%</span>'
    )


def badge_html(doc_type: str) -> str:
    cls = {"PAN": "badge-pan", "Aadhaar": "badge-aadhaar"}.get(doc_type, "badge-unknown")
    return f'<span class="badge {cls}">{doc_type}</span>'


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️  Settings")
        ocr_choice = st.selectbox(
            "OCR Engine",
            ["auto", "paddleocr", "easyocr"],
            index=0,
            help="auto = try PaddleOCR first, fall back to EasyOCR",
        )
        do_deskew = st.checkbox("Auto-deskew", value=True)
        denoise_str = st.slider("Denoise strength", 0, 30, 10)
        clahe_clip = st.slider("Contrast (CLAHE clip)", 1.0, 4.0, 2.0, 0.1)
        sharp_str = st.slider("Sharpen", 0.0, 2.0, 0.8, 0.1)
        extraction_mode = st.selectbox(
            "Extraction Mode",
            ["auto", "heuristic_ocr", "yolo_ocr"],
            index=0,
            help=(
                "auto = try YOLO first, fall back to heuristic\n"
                "yolo_ocr = YOLOv8 detects field regions, then OCR on crops\n"
                "heuristic_ocr = full-image OCR + spatial rules"
            ),
        )

        st.divider()
        st.caption("""
        **Supported documents**
        - 🪪 PAN Card
        - 🪪 Aadhaar Card

        **Accepted formats**
        - JPEG / PNG images
        - PDF (first page)
        """)

    # ── Title ────────────────────────────────────────────────────────────
    st.title("🪪  Intelligent Document ID Processor")
    st.markdown(
        "Upload a PAN or Aadhaar card image. The system will classify the document, "
        "extract key fields, and validate them — all locally, no API calls."
    )

    # ── File uploader ────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload document",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆  Upload an image or PDF to get started.")
        return

    # ── Load image ───────────────────────────────────────────────────────
    try:
        if uploaded.type == "application/pdf":
            pages = pdf_to_images(uploaded.read(), dpi=300)
            if not pages:
                st.error("PDF appears empty.")
                return
            pil_img = pages[0]
            img = np.array(pil_img)[:, :, ::-1]  # RGB → BGR
            st.caption(f"PDF loaded – processing first of {len(pages)} page(s).")
        else:
            img = load_image_from_bytes(uploaded.read())
            pil_img = Image.fromarray(img[:, :, ::-1])
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    # ── Preprocess ───────────────────────────────────────────────────────
    with st.spinner("Preprocessing image …"):
        proc_img, meta = preprocess_document(
            img,
            max_dim=PREPROCESSING["max_dimension"],
            denoise_strength=denoise_str,
            clahe_clip=clahe_clip,
            sharpen_strength=sharp_str,
            do_deskew=do_deskew,
        )

    # ── Initialise engines (cached) ──────────────────────────────────────
    try:
        ocr = get_ocr(ocr_choice)
    except ImportError as e:
        st.error(str(e))
        return

    classifier = get_classifier()
    extractor = FieldExtractor(ocr, extraction_mode=extraction_mode)

    # ── Run pipeline ─────────────────────────────────────────────────────
    with st.spinner("Running classification & extraction …"):
        t0 = time.time()

        # 1. OCR full text (for classifier heuristic)
        full_text = ocr.get_full_text(proc_img)

        # 2. Classify
        doc_type, cls_conf, cls_scores = classifier.classify(proc_img, full_text)

        # 3. Extract
        ext_result = extractor.extract(proc_img, doc_type)

        # 4. Post-process
        fields, confidences = validate_and_format_fields(
            ext_result.fields, ext_result.confidences,
        )

        elapsed = round(time.time() - t0, 2)

    # ── Annotated image ──────────────────────────────────────────────────
    annotated = draw_bboxes(proc_img, ext_result.bboxes, fields)

    # ── Build output JSON ────────────────────────────────────────────────
    output_json = {
        "document_type": doc_type,
        "classification_method": classifier.method,
        "classification_models_available": classifier.available_models,
        "extraction_mode": ext_result.extraction_mode,
        "fields": fields,
        "confidence": confidences,
        "classification_scores": cls_scores,
        "processing_time_seconds": elapsed,
        "ocr_engine": ocr.engine_name,
        "preprocessing": meta,
    }

    # ── Render results ───────────────────────────────────────────────────
    st.divider()
    # Method badges
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.caption(f"🏷️ Classification: **{classifier.method}**")
    with col_m2:
        st.caption(f"🔍 Extraction: **{ext_result.extraction_mode}**")

    # Classification badge row
    col_badge, col_time = st.columns([3, 1])
    with col_badge:
        st.markdown(f"**Document Type:**  {badge_html(doc_type)}", unsafe_allow_html=True)
    with col_time:
        st.caption(f"⏱  {elapsed} s")

    # Confidence breakdown for classification
    # ── Techniques used panel ────────────────────────────────────────────
    with st.expander("🧠 Deep Learning Techniques Used", expanded=False):
        avail = classifier.available_models
        ext_mode = ext_result.extraction_mode

        st.markdown("### Classification")
        for name, active in avail.items():
            icon = "✅" if active else "⬜"
            detail = ""
            if "ViT" in name and active:
                detail = " — *Vision Transformer (google/vit-base-patch16-224)*"
            elif "CNN" in name and active:
                detail = " — *MobileNetV2 backbone, ImageNet pretrained*"
            elif "Heuristic" in name:
                detail = " — *Keyword + regex fallback (always available)*"
            st.markdown(f"{icon} **{name}**{detail}")

        st.markdown("---")
        st.markdown("### Field Extraction")
        if ext_mode == "yolo_ocr":
            st.markdown("✅ **YOLOv8 Object Detection** — detects field bounding boxes, then OCR on each crop")
        else:
            st.markdown("⬜ **YOLOv8 Object Detection** — not active (no trained model)")
            st.markdown("✅ **DL-based Text Detection** — PaddleOCR/EasyOCR DBNet/CRAFT detects text regions")
            st.markdown("✅ **DL-based Text Recognition** — CRNN + CTC recognition on detected regions")

        st.markdown("---")
        st.markdown("### Post-Processing")
        st.markdown("✅ Regex validation (PAN format, Aadhaar 12-digit)")
        st.markdown("✅ Date normalisation → YYYY-MM-DD")
        st.markdown("✅ Per-field confidence scoring")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_results, tab_image, tab_ocr, tab_json = st.tabs(
        ["📋  Extracted Fields", "🖼️  Image View", "🔍  Raw OCR", "{ }  JSON Output"]
    )

    # — Tab 1: Extracted Fields ───────────────────────────────────────────
    with tab_results:
        if not fields:
            st.warning("No fields could be extracted. Try a clearer image.")
        else:
            for fname, fval in fields.items():
                conf = confidences.get(fname, 0)
                with st.container():
                    col_k, col_v, col_c = st.columns([2, 5, 2])
                    col_k.markdown(f"**{fname.replace('_', ' ').title()}**")
                    col_v.code(fval, language=None)
                    col_c.markdown(conf_bar_html(conf), unsafe_allow_html=True)
                    st.divider()

    # — Tab 2: Image View ────────────────────────────────────────────────
    with tab_image:
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)
        with col_ann:
            st.subheader("Annotated")
            st.image(cv2_to_pil(annotated), use_container_width=True)

    # — Tab 3: Raw OCR ───────────────────────────────────────────────────
    with tab_ocr:
        st.subheader(f"All detected text regions ({len(ext_result.all_ocr_results)})")
        if ext_result.all_ocr_results:
            rows = []
            for i, r in enumerate(ext_result.all_ocr_results):
                rows.append({
                    "#": i + 1,
                    "Text": r.text,
                    "Confidence": f"{r.confidence:.2%}",
                    "Position": f"({r.bbox[0]}, {r.bbox[1]}) → ({r.bbox[2]}, {r.bbox[3]})",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No text detected.")

    # — Tab 4: JSON Output ───────────────────────────────────────────────
    with tab_json:
        json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")
        st.download_button(
            "⬇  Download JSON",
            json_str,
            file_name="extracted_data.json",
            mime="application/json",
        )

    # ── Sidebar: preprocessing metadata ──────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Preprocessing")
        for k, v in meta.items():
            if isinstance(v, tuple):
                st.text(f"{k}: {v[0]}×{v[1]}")
            else:
                st.text(f"{k}: {v}")


if __name__ == "__main__":
    main()