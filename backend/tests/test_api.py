import io, json
from pathlib import Path
import pytest
from app.config import settings
from app.main import clf, val
from app.ml import confidence
from app.ml.classifier import LABELS
from tests.conftest import png_bytes

F = lambda b, n="x.png", t="image/png": {"file": (n, io.BytesIO(b), t)}


def test_health(client):
    j = client.get("/api/health").json()
    assert {"disease_model_ready", "validator_ready", "device", "mode", "version"} <= j.keys()
    assert "path" not in json.dumps(j).lower()


def test_model_info(client):
    j = client.get("/api/model-info").json()
    assert j["classes"] == LABELS["classes"] and j["evaluation_metrics"] is None


def test_missing_upload(client):
    r = client.post("/api/analyze"); assert r.status_code == 400 and r.json()["success"] is False


def test_corrupted_and_renamed_nonimage(client):
    r = client.post("/api/analyze", files=F(b"not an image at all", "a.jpg", "image/jpeg"))
    assert r.status_code == 422 and r.json()["error_code"] == "CORRUPT_IMAGE"
    assert "Traceback" not in r.text


def test_oversized(client):
    r = client.post("/api/analyze", files=F(b"x" * (settings.max_upload_mb * 1024 * 1024 + 10)))
    assert r.status_code == 413


def test_too_small(client):
    assert client.post("/api/analyze", files=F(png_bytes((10, 10)))).json()["error_code"] == "IMAGE_TOO_SMALL"


def test_mock_non_mango_rejected_and_not_stored(client, mock_validator):
    mock_validator(0.05)
    before = client.get("/api/history").json()["total"]
    r = client.post("/api/analyze", files=F(png_bytes()))
    assert r.status_code == 422 and r.json()["error_code"] == "MANGO_NOT_DETECTED"
    assert "prediction" not in r.json() and client.get("/api/history").json()["total"] == before


def test_mock_validator_real_classifier_flow(client, mock_validator):
    """Validator is MOCKED; classifier is the REAL checkpoint on a synthetic image (checks plumbing only)."""
    if not clf.ready:
        pytest.skip("checkpoint not loaded")
    mock_validator(0.99)
    r = client.post("/api/analyze", files=F(png_bytes()))
    assert r.status_code == 200
    j = r.json(); aid = j["analysis_id"]
    assert j["prediction"]["class_id"] in LABELS["ids"] and j["status"] in {"healthy_appearance", "possible_disease", "uncertain"}
    assert client.get(f"/api/history/{aid}").json()["id"] == aid
    pdf = client.get(f"/api/reports/{aid}")
    assert pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
    assert client.delete(f"/api/history/{aid}").json()["success"] is True
    assert client.get(f"/api/history/{aid}").status_code == 404


def test_model_missing_returns_503(client, mock_validator, monkeypatch):
    mock_validator(0.99)
    monkeypatch.setattr(clf, "model", None)
    r = client.post("/api/analyze", files=F(png_bytes()))
    assert r.status_code == 503 and r.json()["error_code"] == "MODEL_UNAVAILABLE"


@pytest.mark.parametrize("path", ["/api/history/..%2F..%2Fetc", "/api/history/nope", "/api/reports/nope", "/api/diseases/nope"])
def test_invalid_ids(client, path):
    assert client.get(path).status_code == 404


def test_diseases_search(client):
    assert client.get("/api/diseases?q=powdery").json()["items"][0]["id"] == "powdery_mildew"


def test_class_mapping_and_kb_consistency():
    assert LABELS["classes"] == ["Anthracnose", "Bacterial Canker", "Healthy", "Powdery Mildew", "Scab", "Sooty Mould", "Stem End Rot"]
    kb = json.loads((Path(__file__).parents[1] / "app/knowledge/disease_info.json").read_text())["diseases"]
    assert {d["id"] for d in kb} == set(LABELS["ids"])


def test_preprocess_shape():
    from PIL import Image
    assert tuple(clf.preprocess(Image.new("RGB", (500, 300))).shape) == (1, 3, 224, 224)


def _r(top, second, tid="anthracnose"):
    return [{"class_id": tid, "class_name": tid, "score": top}, {"class_id": "scab", "class_name": "Scab", "score": second}]


def test_abstention_policy():
    assert confidence.decide(_r(0.5, 0.4)) == "uncertain"
    assert confidence.decide(_r(0.95, 0.04)) == "possible_disease"
    assert confidence.decide(_r(0.97, 0.01, "healthy")) == "healthy_appearance"


def test_real_checkpoint_loads_strictly():
    assert clf.ready, f"checkpoint failed to load: {clf.error}"


@pytest.mark.skipif(not val.ready, reason="CLIP validator not available offline (needs Hugging Face download)")
def test_real_validator_scores_in_range():
    from PIL import Image
    assert 0.0 <= val.score(Image.new("RGB", (224, 224), (200, 200, 200))) <= 1.0
