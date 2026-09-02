"""
Document Type Classifier — Multi-model cascade:

  1. ViT (Vision Transformer)  → if fine-tuned model exists
  2. CNN (MobileNetV2)          → if trained weights exist
  3. Heuristic (keywords+regex) → always available as fallback

This design ensures the system always works while using the most
sophisticated available model.
"""

import re
import numpy as np
from typing import Tuple, Dict, Optional
import logging

from models.vit_classifier import ViTClassifier

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Aadhaar", "PAN", "Unknown"]

_SIGNALS = {
    "PAN": {
        "keywords": [
            "income tax", "permanent account", "pan",
            "government of india", "assessment year",
            "income tax department",
        ],
        "pattern": r"[A-Z]{5}[0-9]{4}[A-Z]",
    },
    "Aadhaar": {
        "keywords": [
            "aadhaar", "uidai", "unique identification",
            "authority of india", "aadhaar number",
        ],
        "pattern": r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}",
    },
}


class DocumentClassifier:
    """
    Cascading document classifier.

    Tries models in order of sophistication:
      ViT (Transformer) → CNN → Heuristic
    """

    def __init__(self, vit_model_path: Optional[str] = None,
                 cnn_model_path: Optional[str] = None):
        # Model 1: Vision Transformer
        self.vit = ViTClassifier(vit_model_path)

        # Model 2: CNN
        self.cnn_model = None
        self.cnn_available = False
        self._try_load_cnn(cnn_model_path)

        self._method_used = "heuristic"

    def _try_load_cnn(self, model_path):
        if not model_path:
            return
        try:
            import tensorflow as tf
            import os
            if os.path.exists(model_path):
                self.cnn_model = tf.keras.models.load_model(model_path)
                self.cnn_available = True
                logger.info(f"✅ CNN classifier loaded from {model_path}")
        except ImportError:
            logger.warning("TensorFlow not installed — CNN not available")
        except Exception as e:
            logger.warning(f"Could not load CNN: {e}")

    @property
    def method(self) -> str:
        return self._method_used

    @property
    def available_models(self) -> Dict[str, bool]:
        return {
            "ViT (Transformer)": self.vit.is_available,
            "CNN (MobileNetV2)": self.cnn_available,
            "Heuristic": True,
        }

    def classify(
        self, image: np.ndarray, ocr_text: str = ""
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify document. Tries ViT → CNN → Heuristic.
        """
        # Level 1: Vision Transformer
        if self.vit.is_available:
            result = self.vit.classify(image, ocr_text)
            if result is not None:
                label, conf, scores = result
                if conf >= 0.50:
                    self._method_used = "vit"
                    return label, conf, scores

        # Level 2: CNN
        if self.cnn_available:
            try:
                label, conf, scores = self._classify_cnn(image)
                if conf >= 0.60:
                    self._method_used = "cnn"
                    return label, conf, scores
            except Exception as e:
                logger.warning(f"CNN inference failed: {e}")

        # Level 3: Heuristic
        self._method_used = "heuristic"
        return self._classify_heuristic(ocr_text)

    def _classify_cnn(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        from utils.preprocessing import resize_image, prepare_for_classifier
        img = prepare_for_classifier(resize_image(image, 224), (224, 224))
        preds = self.cnn_model.predict(np.expand_dims(img, 0), verbose=0)[0]
        scores = {n: float(preds[i]) for i, n in enumerate(CLASS_NAMES)}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    def _classify_heuristic(self, ocr_text: str) -> Tuple[str, float, Dict[str, float]]:
        lower = ocr_text.lower()
        scores: Dict[str, float] = {"Aadhaar": 0.0, "PAN": 0.0, "Unknown": 0.0}

        for doc_type, sig in _SIGNALS.items():
            s = 0.0
            for kw in sig["keywords"]:
                if kw in lower:
                    s += 0.20
            if re.search(sig["pattern"], ocr_text):
                s += 0.40
                if doc_type == "PAN" and re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", ocr_text):
                    s += 0.20
                elif doc_type == "Aadhaar":
                    s += 0.20
            scores[doc_type] = min(s, 1.0)

        best = max(scores, key=scores.get)
        best_s = scores[best]
        if best_s < 0.30:
            return "Unknown", 0.20, scores

        total = sum(scores.values()) or 1.0
        norm = {k: round(v / total, 3) for k, v in scores.items()}
        return best, round(best_s, 3), norm