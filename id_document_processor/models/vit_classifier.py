"""
Vision Transformer (ViT) based Document Classifier.

Architecture:
    google/vit-base-patch16-224 (pretrained on ImageNet)
    → Remove classification head
    → Add custom head: Dropout → Dense(256) → ReLU → Dropout → Dense(3)

When a fine-tuned model exists at model_path, it is loaded directly.
Otherwise, the base ViT is used as a feature extractor (no classification).
Fallback chain: ViT → CNN → Heuristic
"""

import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Aadhaar", "PAN", "Unknown"]


class ViTClassifier:
    """
    Vision Transformer document classifier.

    Uses HuggingFace transformers library with google/vit-base-patch16-224
    backbone — a pure Transformer architecture (no CNN).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.vit_available = False
        self._try_load()

    def _try_load(self):
        """Try to load a fine-tuned ViT model."""
        if not self.model_path:
            logger.info("No ViT model path — ViT classifier not active")
            return

        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import torch

            path = Path(self.model_path)
            if path.exists():
                self.processor = ViTImageProcessor.from_pretrained(str(path))
                self.model = ViTForImageClassification.from_pretrained(str(path))
                self.model.eval()
                self.vit_available = True
                logger.info(f"✅ ViT classifier loaded from {path}")
            else:
                logger.info(f"ViT model not found at {path} — not active")
        except ImportError:
            logger.warning("transformers/torch not installed — ViT not available")
        except Exception as e:
            logger.warning(f"Could not load ViT: {e}")

    @property
    def is_available(self) -> bool:
        return self.vit_available

    def classify(
        self, image: np.ndarray, ocr_text: str = ""
    ) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """
        Classify using ViT. Returns None if not available.
        """
        if not self.vit_available:
            return None

        try:
            import torch
            from PIL import Image

            # Convert BGR OpenCV → RGB PIL
            rgb = Image.fromarray(image[:, :, ::-1])

            inputs = self.processor(images=rgb, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            scores = {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(min(len(CLASS_NAMES), len(probs)))
            }
            best = max(scores, key=scores.get)
            return best, scores[best], scores

        except Exception as e:
            logger.warning(f"ViT inference failed: {e}")
            return None