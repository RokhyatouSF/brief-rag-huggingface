import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_support_ticket_empty_request_returns_400():
    response = client.post("/support-ticket")
    assert response.status_code == 400
    assert "Veuillez fournir au moins une note vocale" in response.json()["detail"]


def test_support_ticket_text_only_request():
    response = client.post(
        "/support-ticket",
        data={"description": "J'ai reçu mon colis avec un verre fendu et une rayure importante."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "ticket_id" in data
    assert data["ticket_id"].startswith("TICK-")
    assert data["status"] in ["Remboursable", "À vérifier", "Refusé"]
    assert "J'ai reçu mon colis avec un verre fendu" in data["customer_claim_text"]
    assert data["policy_match"] is not None


def test_support_ticket_with_dummy_image():
    # Génération d'une fausse image de test PNG de 10x10 pixels
    from PIL import Image
    img_bytes = io.BytesIO()
    img = Image.new('RGB', (10, 10), color='red')
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    files = {
        'image': ('test_damaged_product.jpg', img_bytes, 'image/jpeg')
    }
    data = {
        'description': "Le produit est cassé dans son emballage"
    }

    response = client.post("/support-ticket", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["vision_analysis"]["processed"] is True
    assert res_data["status"] in ["Remboursable", "À vérifier", "Refusé"]


def test_support_ticket_invalid_audio_format():
    files = {
        'audio': ('test.exe', b'fake binary content', 'application/octet-stream')
    }
    response = client.post("/support-ticket", files=files)
    assert response.status_code == 400
    assert "Format audio non supporté" in response.json()["detail"]
