from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_status" in data
    assert "whisper_asr" in data["models_status"]
    assert "vit_vision" in data["models_status"]
    assert "rag_embeddings" in data["models_status"]
