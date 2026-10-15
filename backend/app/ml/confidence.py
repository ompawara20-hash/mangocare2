"""Abstention policy. Thresholds are conservative defaults, NOT validated calibrations."""
from ..config import settings


def decide(ranked: list[dict]) -> str:
    """Return status: healthy_appearance | possible_disease | uncertain."""
    top, second = ranked[0], ranked[1]
    if top["score"] < settings.confidence_threshold or (top["score"] - second["score"]) < settings.margin_threshold:
        return "uncertain"
    return "healthy_appearance" if top["class_id"] == "healthy" else "possible_disease"
