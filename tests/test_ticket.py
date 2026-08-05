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
    assert data["status"] in ["Remboursable", "À vérifier", "Refusé", "Échange gratuit", "Expédition de pièce", "Non remboursable - Retard mineur", "Dédommagement 10%", "Remboursable - Colis perdu", "En attente de justificatifs"]
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
    assert res_data["status"] in ["Remboursable", "À vérifier", "Refusé", "Échange gratuit", "Expédition de pièce", "Non remboursable - Retard mineur", "Dédommagement 10%", "Remboursable - Colis perdu", "En attente de justificatifs"]


def test_support_ticket_smarthelp_wrong_item_rule():
    response = client.post(
        "/support-ticket",
        data={"description": "Bonjour, la couleur du T-shirt reçu ne correspond pas à ma commande, j'ai reçu un rouge au lieu de bleu."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["policy_match"]["rule_code"] == "2.1"
    assert data["status"] == "Échange gratuit"


def test_support_ticket_smarthelp_missing_part_rule():
    response = client.post(
        "/support-ticket",
        data={"description": "Bonjour, il manque une pièce et un accessoire dans le kit reçu."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["policy_match"]["rule_code"] == "2.2"
    assert data["status"] == "Expédition de pièce"


def test_support_ticket_invalid_audio_format():
    files = {
        'audio': ('test.exe', b'fake binary content', 'application/octet-stream')
    }
    response = client.post("/support-ticket", files=files)
    assert response.status_code == 400
    assert "Format audio non supporté" in response.json()["detail"]


def test_support_ticket_mpeg_audio_supported():
    files = {
        'audio': ('audio_whatsapp.mpeg', b'dummy mpeg audio content', 'audio/mpeg')
    }
    response = client.post("/support-ticket", files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["audio_analysis"]["processed"] is True


def test_support_ticket_damaged_image_with_vague_delivery_text(monkeypatch):
    from PIL import Image
    from app.services.vision_service import VisionService
    
    # Mocking Vision Service to return a damaged product result
    monkeypatch.setattr(
        VisionService,
        "analyze_image_bytes",
        lambda self, bytes_data, fname: {
            "processed": True,
            "label": "a damaged, broken or cracked product with visible physical defects",
            "condition_status": "Produit endommagé / cassé",
            "confidence": 0.9526,
            "detected_defects": ["Fissure matérielle visible"],
            "file_name": fname,
            "error": None
        }
    )

    img_bytes = io.BytesIO()
    img = Image.new('RGB', (10, 10), color='red')
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    files = {'image': ('images.jpg', img_bytes, 'image/jpeg')}
    data = {'description': "je l'ai recu comme ca , a la livraison"}

    response = client.post("/support-ticket", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "Remboursable"
    assert "Règle 1.1" in res_data["applied_rule"]


def test_support_ticket_client_fault_refused(monkeypatch):
    from PIL import Image
    from app.services.vision_service import VisionService
    
    monkeypatch.setattr(
        VisionService,
        "analyze_image_bytes",
        lambda self, bytes_data, fname: {
            "processed": True,
            "label": "a damaged, broken or cracked product with visible physical defects",
            "condition_status": "Produit endommagé / cassé",
            "confidence": 0.9526,
            "detected_defects": ["Fissure matérielle visible"],
            "file_name": fname,
            "error": None
        }
    )

    img_bytes = io.BytesIO()
    img = Image.new('RGB', (10, 10), color='red')
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    files = {'image': ('images.jpg', img_bytes, 'image/jpeg')}
    data = {'description': "C'est avec moi qu'il s'est cassee"}

    response = client.post("/support-ticket", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "Refusé"
    assert "Règle 4.1" in res_data["applied_rule"]


def test_support_ticket_delay_exceeded_rule_1_2(monkeypatch):
    from PIL import Image
    from app.services.vision_service import VisionService
    
    monkeypatch.setattr(
        VisionService,
        "analyze_image_bytes",
        lambda self, bytes_data, fname: {
            "processed": True,
            "label": "a damaged, broken or cracked product with visible physical defects",
            "condition_status": "Produit endommagé / cassé",
            "confidence": 0.9526,
            "detected_defects": ["Fissure matérielle visible"],
            "file_name": fname,
            "error": None
        }
    )

    img_bytes = io.BytesIO()
    img = Image.new('RGB', (10, 10), color='red')
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    files = {'image': ('images.jpg', img_bytes, 'image/jpeg')}
    data = {'description': "elle s'est cassee apres 2 jours de la livraison"}

    response = client.post("/support-ticket", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "À vérifier"
    assert "Règle 1.2" in res_data["applied_rule"]




