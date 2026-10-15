"""AA-ENet disease classifier: load once, strict checkpoint check, CPU/CUDA, no_grad."""
import json
import logging
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from .arch import AAENet
from ..config import settings

log = logging.getLogger("mangocare.classifier")
LABELS = json.loads((Path(__file__).parent / "labels.json").read_text())


class ModelUnavailable(Exception):
    pass


class MangoClassifier:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Same inference transform the package uses: plain resize to 224x224 + ImageNet norm.
        self.tf = transforms.Compose([
            transforms.Resize((LABELS["input_size"],) * 2),
            transforms.ToTensor(),
            transforms.Normalize(LABELS["mean"], LABELS["std"]),
        ])

    def load(self) -> None:
        try:
            if not settings.model_path.is_file():
                raise FileNotFoundError("checkpoint file not found")
            m = AAENet(len(LABELS["classes"]), dropout=0.276633)  # dropout irrelevant in eval; matches package
            state = torch.load(settings.model_path, map_location=self.device, weights_only=True)
            m.load_state_dict(state, strict=True)  # raises if classes/shape incompatible
            self.model = m.to(self.device).eval()
            self.error = None
            log.info("Disease model loaded on %s", self.device)
        except Exception as e:  # keep details in logs only
            self.model, self.error = None, type(e).__name__
            log.error("Disease model failed to load: %s", e)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img).unsqueeze(0)
        assert tuple(x.shape) == (1, 3, LABELS["input_size"], LABELS["input_size"])
        return x

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> list[dict]:
        if not self.ready:
            raise ModelUnavailable()
        logits = self.model(self.preprocess(img).to(self.device))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        ranked = sorted(
            ({"class_id": LABELS["ids"][i], "class_name": LABELS["classes"][i], "score": round(p, 5)}
             for i, p in enumerate(probs)), key=lambda r: -r["score"])
        return ranked
