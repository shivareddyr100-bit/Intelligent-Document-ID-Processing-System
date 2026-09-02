"""
Unified OCR engine adapter.
Auto-detects and wraps PaddleOCR or EasyOCR behind a common interface.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Single detected text region."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    text: str
    confidence: float
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.center = (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


class OCREngine:
    """Factory-wrapper that picks the best available OCR backend."""

    def __init__(self, engine: str = "auto"):
        self.engine_name: str = engine
        self._engine = None
        self._init(engine)

    def _init(self, engine: str):
        if engine == "auto":
            for name in ("easyocr","paddleocr" ):
                try:
                    getattr(self, f"_init_{name}")()
                    return
                except ImportError:
                    continue
            raise ImportError(
                "No OCR engine found.\n"
                "Install one of:\n"
                "  pip install easyocr"
                "  pip install paddlepaddle paddleocr\n"
                
            )
        else:
            getattr(self, f"_init_{engine}")()

    def _init_paddleocr(self):
        from paddleocr import PaddleOCR

        # Attempt 1: bare minimum (PaddleOCR v4+)
        try:
            self._engine = PaddleOCR(lang="en")
            self.engine_name = "paddleocr"
            logger.info("PaddleOCR ready (v4+)")
            return
        except Exception:
            pass

        # Attempt 2: legacy params (v2/v3)
        try:
            self._engine = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False,
                det_db_thresh=0.3, det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
            )
            self.engine_name = "paddleocr"
            logger.info("PaddleOCR ready (legacy)")
            return
        except Exception:
            pass

        # Attempt 3: absolutely zero args
        self._engine = PaddleOCR()
        self.engine_name = "paddleocr"
        logger.info("PaddleOCR ready (default)")

    def _init_easyocr(self):
        import easyocr  # type: ignore
        self._engine = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.engine_name = "easyocr"
        logger.info("EasyOCR ready")

    # ── Public API ───────────────────────────────────────────────────────
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        if self.engine_name == "paddleocr":
            return self._run_paddleocr(image)
        return self._run_easyocr(image)

    def get_full_text(self, image: np.ndarray) -> str:
        return " ".join(r.text for r in self.recognize(image))

    # ── PaddleOCR runner ─────────────────────────────────────────────────
    def _run_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        out: List[OCRResult] = []

        # Try calling with NO extra kwargs (v4+ compatible)
        raw = None
        try:
            raw = self._engine.ocr(image)
        except TypeError:
            pass

        if raw is None:
            try:
                raw = self._engine.ocr(image, cls=True)
            except TypeError:
                try:
                    raw = self._engine.ocr(image, use_angle_cls=True)
                except TypeError:
                    raw = self._engine.ocr()

        if raw is None:
            return out

        # Normalise nested vs flat output
        lines = raw
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            first = raw[0]
            if first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
                lines = first

        if not isinstance(lines, list):
            return out

        for item in lines:
            try:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                pts, txt_conf = item
                if not isinstance(txt_conf, (list, tuple)) or len(txt_conf) != 2:
                    continue
                txt, conf = txt_conf
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                out.append(OCRResult(
                    bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    text=str(txt),
                    confidence=float(conf),
                ))
            except (TypeError, IndexError, ValueError):
                continue

        return out

    # ── EasyOCR runner ───────────────────────────────────────────────────
    def _run_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        raw = self._engine.readtext(rgb)
        out: List[OCRResult] = []
        for pts, txt, conf in raw:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append(OCRResult(
                bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                text=txt, confidence=conf,
            ))
        return out