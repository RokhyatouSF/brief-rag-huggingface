from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    REMBOURSABLE = "Remboursable"
    A_VERIFIER = "À vérifier"
    REFUSE = "Refusé"
    ECHANGE_GRATUIT = "Échange gratuit"
    EXPEDITION_PIECE = "Expédition de pièce"
    NON_REMBOURSABLE_RETARD_MINEUR = "Non remboursable - Retard mineur"
    DEDOMMAGEMENT_10 = "Dédommagement 10%"
    REMBOURSABLE_COLIS_PERDU = "Remboursable - Colis perdu"
    EN_ATTENTE_JUSTIFICATIFS = "En attente de justificatifs"


class AudioAnalysisResult(BaseModel):
    processed: bool = Field(..., description="Indique si un fichier audio a été traité")
    transcription: Optional[str] = Field(None, description="Texte transcrit à partir de la note vocale")
    file_name: Optional[str] = Field(None, description="Nom du fichier audio soumis")
    language_detected: Optional[str] = Field("fr", description="Langue détectée")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel lors de la transcription")


class VisionAnalysisResult(BaseModel):
    processed: bool = Field(..., description="Indique si une photo de produit a été traitée")
    label: Optional[str] = Field(None, description="Classe brute issue du modèle ViT")
    condition_status: Optional[str] = Field(None, description="État du produit (ex: Produit endommagé / cassé, Produit intact / conforme)")
    confidence: Optional[float] = Field(None, description="Score de confiance de l'analyse visuelle (0.0 à 1.0)")
    detected_defects: List[str] = Field(default_factory=list, description="Liste des anomalies ou défauts visuels identifiés")
    file_name: Optional[str] = Field(None, description="Nom du fichier image soumis")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel lors de l'analyse d'image")


class PolicyMatch(BaseModel):
    article_id: str = Field(..., description="Identifiant de l'article CGV / FAQ")
    rule_code: Optional[str] = Field(None, description="Code officiel de la règle SmartHelp (ex: 1.1, 2.1)")
    title: str = Field(..., description="Titre de la règle de gestion ou clause CGV")
    category: str = Field(..., description="Catégorie (ex: Retours, Garantie, Remboursement)")
    explicit_rule: Optional[str] = Field(None, description="Texte explicite intégral de la règle appliquée")
    excerpt: str = Field(..., description="Extrait du texte de la règle appliquée")
    similarity_score: float = Field(..., description="Score de pertinence sémantique s'étalant de 0 à 1")
    refund_eligible: bool = Field(..., description="Eligibilité au remboursement selon cette règle")
    status_associated: Optional[str] = Field(None, description="Statut associé par la règle SmartHelp")


class SupportTicketDiagnosticResponse(BaseModel):
    ticket_id: str = Field(..., description="Identifiant unique du ticket généré")
    timestamp: str = Field(..., description="Horodatage ISO de l'ingestion")
    status: TicketStatus = Field(..., description="Statut préconisé pour l'aiguillage automatique")
    applied_rule: Optional[str] = Field(None, description="Nom et libellé explicites de la règle appliquée (ex: Règle 1.1 (Casse / Dommage visible))")
    summary: str = Field(..., description="Synthèse claire du diagnostic global")
    customer_claim_text: str = Field(..., description="Texte global de la réclamation (transcription audio ou description textuelle)")
    audio_analysis: AudioAnalysisResult = Field(..., description="Résultats du traitement ASR de la note vocale")
    vision_analysis: VisionAnalysisResult = Field(..., description="Résultats du traitement visuel de la photo du produit")
    policy_match: Optional[PolicyMatch] = Field(None, description="Règle de gestion / CGV la plus pertinente identifiée via le RAG")
    recommended_actions: List[str] = Field(..., description="Actions recommandées pour les agents du service client")
