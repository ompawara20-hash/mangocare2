import io, os, tempfile
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "test.db")  # isolated DB; set before app import
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app, clf, val


def png_bytes(size=(300, 300), color=(120, 160, 60)):
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, "PNG"); return b.getvalue()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:   # runs lifespan: DB init + real model load
        yield c


@pytest.fixture
def mock_validator(monkeypatch):
    """MOCK: forces the CLIP validator result. Never evidence that the real validator works."""
    def make(score):
        monkeypatch.setattr(type(val), "ready", property(lambda s: True))
        monkeypatch.setattr(type(val), "score", lambda s, img: score)
    return make
