import uuid
import datetime
import logging
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from app.schemas.ticket import (
    SupportTicketDiagnosticResponse,
    AudioAnalysisResult,
    VisionAnalysisResult
)
from app.services import (
    get_audio_service,
    get_vision_service,
    get_rag_service,
    DecisionEngineService
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Support Ticket Ingestion"])


@router.post(
    "/support-ticket",
    response_model=SupportTicketDiagnosticResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingestion et analyse IA multimodale d'un ticket de réclamation support",
    description="""
    Endpoint unique d'ingestion acceptant une requête **multipart/form-data** contenant:
    - **audio** *(optionnel)* : Fichier audio de la note vocale du client (`.mp3`, `.wav`)
    - **image** *(optionnel)* : Photo de la preuve du produit (`.png`, `.jpg`, `.jpeg`)
    - **description** *(optionnel)* : Texte explicatif saisi par le client
    
    **Traitements effectués:**
    1. **ASR (Speech-to-Text)**: Transcription automatique de la note vocale via le modèle **Whisper**.
    2. **Vision (Classification ViT)**: Analyse visuelle de l'état du produit (détection de cassures/défauts).
    3. **RAG (Recherche Sémantique)**: Interrogation de la base de connaissances CGV/FAQ avec SentenceTransformers.
    4. **Diagnostic Structuré**: Recommandation d'un statut de ticket (`Remboursable`, `À vérifier`, `Refusé`) et préconisations.
    """
)
async def create_support_ticket(
    audio: Optional[UploadFile] = File(None, description="Fichier vocal (.mp3, .wav)"),
    image: Optional[UploadFile] = File(None, description="Photo de preuve (.png, .jpg)"),
    description: Optional[str] = Form(None, description="Description textuelle complémentaire")
):
    # Validation qu'au moins un élément explicatif est fourni
    if not audio and not image and not (description and description.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Veuillez fournir au moins une note vocale, une image ou une description textuelle pour qualifier votre réclamation."
        )

    # Validation des extensions si fichiers fournis
    if audio and audio.filename:
        audio_ext = audio.filename.lower().rsplit(".", 1)[-1]
        if audio_ext not in ["wav", "mp3", "ogg", "flac", "m4a", "mpeg", "mp4", "aac", "webm"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Format audio non supporté (.{audio_ext}). Veuillez envoyer un fichier .wav, .mp3 ou .mpeg."
            )

    if image and image.filename:
        image_ext = image.filename.lower().rsplit(".", 1)[-1]
        if image_ext not in ["png", "jpg", "jpeg", "webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Format d'image non supporté (.{image_ext}). Veuillez envoyer un fichier .png ou .jpg."
            )

    ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"
    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Traitement Audio ASR (Whisper)
    audio_service = get_audio_service()
    audio_analysis = AudioAnalysisResult(processed=False)
    transcription_text = ""

    if audio and audio.filename:
        try:
            audio_bytes = await audio.read()
            text, err = audio_service.transcribe_audio_bytes(audio_bytes, audio.filename)
            if err:
                audio_analysis = AudioAnalysisResult(
                    processed=True,
                    file_name=audio.filename,
                    error=err
                )
            else:
                transcription_text = text or ""
                audio_analysis = AudioAnalysisResult(
                    processed=True,
                    transcription=transcription_text,
                    file_name=audio.filename
                )
        except Exception as e:
            logger.error(f"Erreur lors du traitement audio: {e}")
            audio_analysis = AudioAnalysisResult(
                processed=True,
                file_name=audio.filename,
                error=str(e)
            )

    # 2. Traitement Visuel (ViT)
    vision_service = get_vision_service()
    vision_analysis = VisionAnalysisResult(processed=False)

    if image and image.filename:
        try:
            image_bytes = await image.read()
            vision_dict = vision_service.analyze_image_bytes(image_bytes, image.filename)
            vision_analysis = VisionAnalysisResult(**vision_dict)
        except Exception as e:
            logger.error(f"Erreur lors du traitement visuel: {e}")
            vision_analysis = VisionAnalysisResult(
                processed=True,
                file_name=image.filename,
                error=str(e)
            )

    # 3. Synthèse de la réclamation textuelle globale pour le RAG
    combined_texts = []
    if transcription_text:
        combined_texts.append(f"Transcription vocale: {transcription_text}")
    if description and description.strip():
        combined_texts.append(f"Description client: {description.strip()}")

    customer_claim_text = " | ".join(combined_texts) if combined_texts else "Aucun texte rédigé."

    # 4. Recherche Documentaire RAG (CGV / FAQ)
    rag_service = get_rag_service()
    policy_match = rag_service.search_policy(customer_claim_text)

    # 5. Moteur de Décision & Recommandation
    ticket_status, applied_rule, summary, recommended_actions = DecisionEngineService.evaluate_ticket(
        customer_claim_text=customer_claim_text,
        audio_result=audio_analysis,
        vision_result=vision_analysis,
        policy_match=policy_match
    )

    return SupportTicketDiagnosticResponse(
        ticket_id=ticket_id,
        timestamp=timestamp_iso,
        status=ticket_status,
        applied_rule=applied_rule,
        summary=summary,
        customer_claim_text=customer_claim_text,
        audio_analysis=audio_analysis,
        vision_analysis=vision_analysis,
        policy_match=policy_match,
        recommended_actions=recommended_actions
    )
