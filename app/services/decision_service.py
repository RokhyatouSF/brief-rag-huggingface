import logging
from typing import Optional, List, Tuple
from app.schemas.ticket import (
    AudioAnalysisResult,
    VisionAnalysisResult,
    PolicyMatch,
    TicketStatus
)

logger = logging.getLogger(__name__)


class DecisionEngineService:
    """
    Moteur de décision central combinant l'analyse vocale ASR, la classification vision ViT
    et les règles sémantiques RAG pour recommander le statut du ticket et les actions associées.
    """

    @staticmethod
    def evaluate_ticket(
        customer_claim_text: str,
        audio_result: AudioAnalysisResult,
        vision_result: VisionAnalysisResult,
        policy_match: Optional[PolicyMatch]
    ) -> Tuple[TicketStatus, str, List[str]]:
        """
        Synthétise les analyses multimodales pour déterminer le statut préconisé.
        Retourne (status, summary_explanation, recommended_actions).
        """
        claim_lower = customer_claim_text.lower()
        has_audio = audio_result.processed and bool(audio_result.transcription)
        has_vision = vision_result.processed
        is_damaged_image = has_vision and (vision_result.condition_status == "Produit endommagé / cassé")
        is_intact_image = has_vision and (vision_result.condition_status == "Produit conforme / intact")

        actions = []

        # 1. Cas d'exclusion explicite ou produit intact confirmé
        if is_intact_image and ("cassé" in claim_lower or "endommagé" in claim_lower or "fissure" in claim_lower):
            status = TicketStatus.REFUSE
            summary = (
                "Contradiction majeure détectée: la note vocale ou le texte signale un produit cassé, "
                "mais l'analyse visuelle par ViT confirme que l'article sur la photo est intact et conforme."
            )
            actions = [
                "Demander au client une nouvelle photo nette sous un autre angle",
                "Refuser le remboursement automatique immédiat",
                "Transmettre au niveau 2 si le client conteste l'analyse"
            ]

        # 2. Cas de produit endommagé confirmé avec politique éligible
        elif is_damaged_image and (policy_match is None or policy_match.refund_eligible):
            status = TicketStatus.REMBOURSABLE
            summary = (
                "Réclamation valide et prouvée: l'analyse visuelle ViT confirme des dommages matériels visibles "
                f"({', '.join(vision_result.detected_defects) or 'Défaut physique'}), ce qui est conforme à la clause "
                f"'{policy_match.title if policy_match else 'Produit endommagé'}' de la politique de retour."
            )
            actions = [
                "Valider le remboursement ou l'expédition d'un produit de remplacement sans frais",
                "Envoyer au client une étiquette de retour prépayée si retour requis",
                "Clôturer le ticket avec le statut Remboursable"
            ]

        # 3. Cas de mauvaise réclamation sans preuves suffisantes
        elif policy_match and not policy_match.refund_eligible:
            status = TicketStatus.REFUSE
            summary = (
                f"Réclamation non éligible d'après la règle interne '{policy_match.title}'. "
                f"Motif: {policy_match.excerpt[:150]}..."
            )
            actions = [
                "Notifier le client du refus selon les CGV applicables",
                "Fournir un lien vers les CGV de l'entreprise"
            ]

        # 4. Cas incertains ou informations partielles
        elif not has_vision and not has_audio and len(customer_claim_text.strip()) < 10:
            status = TicketStatus.A_VERIFIER
            summary = "Éléments insuffisants fournis par le client (absence de fichier audio et photo claire)."
            actions = [
                "Relancer le client pour obtenir une note vocale ou une photo du produit endommagé"
            ]

        else:
            status = TicketStatus.A_VERIFIER
            summary = (
                "Diagnostic mixte nécessitant une validation manuelle par un conseiller support. "
                f"La règle CGV la plus proche est '{policy_match.title if policy_match else 'FAQ Générale'}'."
            )
            actions = [
                "Vérifier manuellement les pièces jointes",
                "Consulter l'historique d'achats du client",
                "Valider ou ajuster le statut avant remboursement"
            ]

        return status, summary, actions
