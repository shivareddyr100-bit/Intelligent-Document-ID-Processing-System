"""
Image preprocessing pipeline for document images.
Handles: resizing, deskewing, denoising, contrast enhancement, sharpening.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image - file may be corrupted or unsupported.")
    return img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def resize_image(img: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Resize keeping aspect ratio; skip if already smaller."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Fast non-local means denoising (colour)."""
    if strength <= 0:
        return img
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=strength, hColor=strength,
        templateWindowSize=7, searchWindowSize=21,
    )


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the L-channel of LAB colour-space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp-mask sharpening."""
    if strength <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def estimate_skew_angle(img: np.ndarray) -> float:
    """Estimate skew using Hough line detection on near-horizontal lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        # Flatten whatever shape OpenCV returns
        flat = line.flatten()
        if len(flat) != 4:
            continue
        x1, y1, x2, y2 = int(flat[0]), int(flat[1]), int(flat[2]), int(flat[3])
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def deskew(img: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """Rotate image to correct skew."""
    if angle is None:
        angle = estimate_skew_angle(img)
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderColor=(255, 255, 255),
    )


def preprocess_document(
    img: np.ndarray,
    max_dim: int = 2000,
    denoise_strength: int = 10,
    clahe_clip: float = 2.0,
    sharpen_strength: float = 0.8,
    do_deskew: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline.
    Returns (processed_image, metadata_dict).
    """
    meta = {"original_size": img.shape[:2]}

    img = resize_image(img, max_dim)
    meta["resized_size"] = img.shape[:2]

    if do_deskew:
        angle = estimate_skew_angle(img)
        img = deskew(img, angle)
        meta["skew_angle_deg"] = round(angle, 2)

    img = denoise(img, denoise_strength)
    img = enhance_contrast(img, clahe_clip)
    img = sharpen(img, sharpen_strength)

    meta["final_size"] = img.shape[:2]
    return img, meta


def prepare_for_classifier(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize + normalise for CNN input."""
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0


def pdf_to_images(file_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF bytes to list of PIL Images (one per page)."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF support.  "
                "Install with:  pip install PyMuPDF"
            )
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return images