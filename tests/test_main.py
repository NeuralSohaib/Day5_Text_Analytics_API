from fastapi.testclient import TestClient
from src.main import app  

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sentiment():
    response = client.post("/analyze/sentiment", params={"text": "I love AI!"})
    assert response.status_code == 200
    assert "sentiment" in response.json()


def test_keywords():
    response = client.post("/analyze/keywords", params={"text": "FastAPI makes building APIs easy"})
    assert response.status_code == 200
    assert "keywords" in response.json()





def test_pii():
    response = client.post("/analyze/pii", params={"text": "My phone number is 123-456-7890"})
    assert response.status_code == 200
    assert "pii_entities" in response.json()


def test_entities():
    response = client.post("/analyze/entities", params={"text": "Barack Obama was the 44th President of the USA."})
    assert response.status_code == 200
    assert "entities" in response.json()


def test_summarize():
    response = client.post(
        "/analyze/summarize",
        params={"text": "FastAPI is a modern web framework for building APIs quickly. It is fast, efficient, and easy to use."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "summary" in data["summary"]  # <- checks nested dict


    data = response.json()
    assert "sentiment" in data
    assert "keywords" in data
    assert "pii" in data
    assert "entities" in data
    assert "summary" in data
   


