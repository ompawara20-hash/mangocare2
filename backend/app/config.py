"""Settings from environment / .env. No secrets; paths resolve relative to backend/."""
import os
from pathlib import Path
from dataclasses import dataclass

BASE = Path(__file__).resolve().parent.parent


def _f(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    app_name: str = _f("APP_NAME", "MangoCare AI")
    app_env: str = _f("APP_ENV", "development")
    frontend_url: str = _f("FRONTEND_URL", "http://localhost:5173")
    max_upload_mb: int = int(_f("MAX_UPLOAD_SIZE_MB", "10"))
    database_path: Path = Path(_f("DATABASE_PATH", str(BASE / "data" / "mangocare.db")))
    model_path: Path = Path(_f("MANGO_MODEL_PATH", str(BASE / "models" / "AA-ENet_proposed.pt")))
    clip_model_id: str = _f("VALIDATOR_MODEL_ID", "openai/clip-vit-base-patch32")
    # Conservative, NOT scientifically calibrated (no validation evidence available).
    confidence_threshold: float = float(_f("CONFIDENCE_THRESHOLD", "0.90"))
    margin_threshold: float = float(_f("MARGIN_THRESHOLD", "0.30"))
    validation_threshold: float = float(_f("VALIDATION_THRESHOLD", "0.50"))
    max_pixels: int = int(_f("MAX_IMAGE_PIXELS", "40000000"))
    min_side: int = int(_f("MIN_IMAGE_SIDE", "64"))


settings = Settings()
