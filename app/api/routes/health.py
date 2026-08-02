from fastapi import APIRouter
from app.config import settings
from app.services import get_audio_service, get_vision_service, get_rag_service

router = APIRouter(tags=["Health & Status"])


@router.get("/health", summary="Vérification de l'état de santé du service et des modèles IA")
async def health_check():
    """
    Retourne l'état de santé de l'API ainsi que l'état d'initialisation en mémoire
    des Singleton pour Whisper ASR, ViT Vision et le moteur RAG.
    """
    audio_srv = get_audio_service()
    vision_srv = get_vision_service()
    rag_srv = get_rag_service()

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": settings.DEVICE,
        "models_status": {
            "whisper_asr": {
                "model_id": settings.WHISPER_MODEL_ID,
                "loaded_in_memory": audio_srv.initialized
            },
            "vit_vision": {
                "model_id": settings.VIT_MODEL_ID,
                "loaded_in_memory": vision_srv.initialized
            },
            "rag_embeddings": {
                "model_id": settings.EMBEDDING_MODEL_ID,
                "knowledge_documents_count": len(rag_srv.documents),
                "loaded_in_memory": rag_srv.initialized
            }
        }
    }
