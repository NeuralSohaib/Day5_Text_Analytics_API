# tests/test_main.py
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_sentiment():
    r = client.post("/analyze/sentiment", json={"text": "I love this product. It's great!"})
    assert r.status_code == 200
    assert "label" in r.json()


def test_keywords():
    r = client.post("/analyze/keywords", json={"text": "FastAPI is fast and easy. FastAPI is great."})
    assert r.status_code == 200
    assert "keywords" in r.json()


def test_readability():
    r = client.post("/analyze/readability", json={"text": "Simple short sentence. Another one."})
    assert r.status_code == 200
    assert "flesch_reading_ease" in r.json()


def test_pii():
    r = client.post("/analyze/pii", json={"text": "Contact me at test@example.com or +1 555 123 4567"})
    assert r.status_code == 200
    assert "pii_entities" in r.json()


def test_full():
    text = "My name is John Doe. I love Python. Email: john@example.com"
    r = client.post("/analyze/full", json={"text": text})
    assert r.status_code == 200
    data = r.json()
    assert "language" in data
    assert "sentiment" in data
    assert "pii" in data
    assert "summary" in data
