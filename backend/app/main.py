"""MangoCare AI FastAPI backend."""
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import db, report
from .config import settings
from .ml import confidence
from .ml.classifier import LABELS, MangoClassifier, ModelUnavailable
from .ml.validator import MangoValidator
from .utils.image_utils import ImageError, load_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mangocare")
VERSION = "0.1.0"
MODEL_NAME, MODEL_VERSION = "AA-ENet (mango-disease-ai)", "0.1.3"
KB = {d["id"]: d for d in json.loads((Path(__file__).parent / "knowledge" / "disease_info.json").read_text())["diseases"]}
ID_RE = re.compile(r"^[0-9a-f]{32}$")
LIMITATIONS = [
    "Preliminary image-based screening for educational use; not a laboratory diagnosis.",
    "The model targets mango leaf disease classes; fruit or other plant parts are outside its documented scope.",
    "Scores are model outputs, not guaranteed probabilities. No accuracy metrics have been independently verified.",
]

clf, val = MangoClassifier(), MangoValidator()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    clf.load()
    val.load()
    yield


app = FastAPI(title=settings.app_name, version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


def err(status: int, code: str, message: str, suggestion: str | None = None, st: str = "error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "status": st, "error_code": code,
                                                     "message": message, "suggestion": suggestion})


@app.exception_handler(ImageError)
async def _img(_: Request, e: ImageError):
    return err(e.status, e.code, e.message, "Please upload a clear JPEG, PNG or WebP photograph.", "invalid_image")


@app.exception_handler(RequestValidationError)
async def _val(_: Request, __: RequestValidationError):
    return err(400, "BAD_REQUEST", "No valid image file was provided in the 'file' field.", "Send multipart/form-data with a 'file' image.")


@app.exception_handler(Exception)
async def _any(_: Request, e: Exception):
    log.exception("Unhandled error")  # details stay in server logs only
    return err(500, "INTERNAL_ERROR", "An unexpected error occurred.")


@app.get("/api/health")
def health():
    ok = clf.ready and val.ready
    return {"status": "ok" if ok else "degraded", "version": VERSION, "database": "ready",
            "disease_model_ready": clf.ready, "validator_ready": val.ready,
            "device": clf.device, "mode": "full_inference" if ok else "unavailable"}


@app.get("/api/model-info")
def model_info():
    return {"name": MODEL_NAME, "version": MODEL_VERSION, "architecture": "EfficientNet-B0 + CBAM + Transformer encoder",
            "scope": "Mango leaf disease classification", "classes": LABELS["classes"],
            "input_size": [LABELS["input_size"]] * 2, "ready": clf.ready, "validator": settings.clip_model_id,
            "validator_ready": val.ready, "evaluation_metrics": None,
            "metrics_note": "No evaluation metrics have been independently verified.",
            "thresholds": {"confidence": settings.confidence_threshold, "margin": settings.margin_threshold,
                           "validation": settings.validation_threshold, "note": "Conservative defaults; not calibrated."},
            "limitations": LIMITATIONS}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    img = load_image(data)  # decode, size, orientation, RGB (raises ImageError)
    if not (clf.ready and val.ready):
        return err(503, "MODEL_UNAVAILABLE", "The analysis service is not ready.", "Check /api/health and the model setup guide.", "service_unavailable")
    vscore = val.score(img)
    if vscore < settings.validation_threshold:
        return err(422, "MANGO_NOT_DETECTED", "The uploaded image does not appear to contain a mango.",
                   "Please upload a clear photograph of a mango.", "invalid_image")  # nothing stored, no prediction
    try:
        ranked = clf.predict(img)
    except ModelUnavailable:
        return err(503, "MODEL_UNAVAILABLE", "The disease model is not loaded.", None, "service_unavailable")
    status = confidence.decide(ranked)
    top = ranked[0]
    rec = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status,
           "class_id": top["class_id"], "class_name": top["class_name"], "confidence": top["score"], "alternatives": ranked[1:4],
           "validation_score": round(vscore, 4), "model_name": MODEL_NAME, "model_version": MODEL_VERSION}
    db.insert(rec)
    info = KB.get(top["class_id"]) if status != "uncertain" else None
    return {"success": True, "analysis_id": rec["id"], "status": status,
            "image_validation": {"is_mango": True, "status": "accepted", "score": rec["validation_score"]},
            "prediction": {"class_id": top["class_id"], "class_name": top["class_name"], "confidence": top["score"],
                           "interpretation": "Model score; not a guaranteed diagnostic probability"},
            "alternatives": rec["alternatives"], "condition_information": info,
            "model": {"name": MODEL_NAME, "version": MODEL_VERSION}, "created_at": rec["created_at"], "limitations": LIMITATIONS}


def _valid_id(aid: str) -> bool:
    return bool(ID_RE.match(aid))


@app.get("/api/history")
def history(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), status: str | None = None):
    items, total = db.page(page, page_size, status)
    return {"items": items, "total": total, "page": page, "page_size": page_size, "stats": db.stats()}


@app.get("/api/history/{aid}")
def history_one(aid: str):
    r = db.get(aid) if _valid_id(aid) else None
    return r or err(404, "NOT_FOUND", "Analysis not found.")


@app.delete("/api/history/{aid}")
def history_delete(aid: str):
    if _valid_id(aid) and db.delete(aid):
        return {"success": True, "note": "Metadata deleted. Uploaded images are never retained."}
    return err(404, "NOT_FOUND", "Analysis not found.")


@app.get("/api/diseases")
def diseases(q: str | None = None):
    items = [d for d in KB.values() if not q or q.lower() in json.dumps(d).lower()]
    return {"items": items, "supported_by_model": LABELS["ids"]}


@app.get("/api/diseases/{did}")
def disease(did: str):
    return KB.get(did) or err(404, "NOT_FOUND", "Disease not found.")


@app.get("/api/reports/{aid}")
def get_report(aid: str):
    r = db.get(aid) if _valid_id(aid) else None
    if not r:
        return err(404, "NOT_FOUND", "Analysis not found.")
    try:
        pdf = report.build_pdf(r, KB.get(r["class_id"]))
    except Exception:
        log.exception("PDF failure")
        return err(500, "REPORT_FAILED", "The report could not be generated.")
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="mangocare-{aid[:8]}.pdf"'})
