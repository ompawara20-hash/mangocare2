"""Mango presence check via CLIP zero-shot (preliminary signal only; conservative)."""
import logging
import torch
from PIL import Image
from ..config import settings

log = logging.getLogger("mangocare.validator")
LABELS = [
    "a photo of a mango fruit", "a photo of a mango leaf", "a photo of another type of fruit",
    "a photo of a non-mango plant leaf", "a photo of a vegetable",
    "a photo of an animal or person", "a photo of an indoor scene or man-made object",
]
POSITIVE = {0, 1}  # mango fruit / mango leaf


class MangoValidator:
    def __init__(self) -> None:
        self.model = self.processor = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
            self.model = CLIPModel.from_pretrained(settings.clip_model_id).eval()
            self.error = None
        except Exception as e:
            self.model = self.processor = None
            self.error = type(e).__name__
            log.error("Validator failed to load: %s", e)

    @property
    def ready(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def score(self, img: Image.Image) -> float:
        inp = self.processor(text=LABELS, images=img, return_tensors="pt", padding=True)
        probs = self.model(**inp).logits_per_image.softmax(dim=1)[0]
        return float(sum(probs[i] for i in POSITIVE))
