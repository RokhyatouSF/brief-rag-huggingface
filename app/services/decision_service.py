import logging
from typing import Optional, List, Tuple
from app.schemas.ticket import (
    AudioAnalysisResult,
    VisionAnalysisResult,
    PolicyMatch,
    TicketStatus
)
from app.services.llm_reasoning_service import get_llm_reasoning_service

logger = logging.getLogger(__name__)


class DecisionEngineService:
    """
    Moteur de décision multimodal combinant l'analyse vocale ASR, la classification vision ViT,
    le raisonnement sémantique IA (Zero-Shot NLI / LLM) et la base RAG SmartHelp.
    """

    @staticmethod
    def evaluate_ticket(
        customer_claim_text: str,
        audio_result: AudioAnalysisResult,
        vision_result: VisionAnalysisResult,
        policy_match: Optional[PolicyMatch]
    ) -> Tuple[TicketStatus, Optional[str], str, List[str]]:
        """
        Synthétise les analyses multimodales via un modèle de raisonnement IA dynamique.
        Retourne (status, applied_rule, summary_explanation, recommended_actions).
        """
        claim_lower = customer_claim_text.lower()
        has_audio = audio_result.processed and bool(audio_result.transcription)
        has_vision = vision_result.processed
        is_damaged_image = has_vision and (vision_result.condition_status == "Produit endommagé / cassé")
        is_intact_image = has_vision and (vision_result.condition_status == "Produit conforme / intact")

        # 1. Analyse d'intention dynamique par IA (LLM / Zero-Shot NLI Reasoning)
        llm_service = get_llm_reasoning_service()
        llm_analysis = llm_service.analyze_claim_intent(
            customer_claim_text=customer_claim_text,
            vision_status=vision_result.condition_status if has_vision else None
        )
        predicted_intent = llm_analysis.get("top_intent")
        llm_reasoning = llm_analysis.get("reasoning", "")

        actions = []

        # 2. Absence de preuve ou justificatifs (Règle 4.2)
        if not has_vision and not has_audio and (not customer_claim_text or len(customer_claim_text.strip()) < 10 or customer_claim_text == "Aucun texte rédigé."):
            status = TicketStatus.EN_ATTENTE_JUSTIFICATIFS
            applied_rule = "Règle 4.2 (Absence de preuve)"
            summary = "Absence de justificatifs probants: ni photo ni note vocale/description explicite fournie (Règle 4.2)."
            actions = [
                "Relancer le client pour obtenir une note vocale ou une photo claire du produit",
                "Mettre le ticket en attente de pièces complémentaires"
            ]
            return status, applied_rule, summary, actions

        # 3. Arbitrage dynamique : Contradiction visuelle ou Faute client / Usure (Règle 4.1)
        if predicted_intent == "4.1" or (is_intact_image and any(kw in claim_lower for kw in ["cassé", "endommagé", "fissure", "abîmé"])):
            status = TicketStatus.REFUSE
            applied_rule = "Règle 4.1 (Usure normale / Mauvaise utilisation)"
            summary = (
                f"Raisonnement sémantique IA (Règle 4.1) : {llm_reasoning}"
                if predicted_intent == "4.1" else
                "Contradiction majeure détectée par l'IA: la note vocale ou le texte signale un produit cassé, "
                "mais l'analyse visuelle ViT confirme que l'article sur la photo est intact et conforme (Règle 4.1)."
            )
            actions = [
                "Notifier le client du refus selon la Règle 4.1 (responsabilité client / usure)",
                "Fournir un lien vers la documentation CGV / politique de retour",
                "Transmettre au niveau 2 si le client conteste l'analyse"
            ]
            return status, applied_rule, summary, actions

        # 4. Arbitrage dynamique : Casse réclamée après le délai de 48 heures (Règle 1.2)
        if predicted_intent == "1.2":
            status = TicketStatus.A_VERIFIER
            applied_rule = "Règle 1.2 (Délai dépassé)"
            summary = f"Raisonnement sémantique IA (Règle 1.2) : {llm_reasoning}"
            actions = [
                "Transmettre le dossier au manager pour validation manuelle de dérogation",
                "Vérifier la date exacte d'achat et la date de livraison effective auprès du transporteur"
            ]
            return status, applied_rule, summary, actions

        # 5. Preuve visuelle directe de dommage / casse dans les délais (Règle 1.1)
        if is_damaged_image and predicted_intent in ["1.1", "1.2", "4.2", None]:
            if predicted_intent != "1.2":
                status = TicketStatus.REMBOURSABLE
                applied_rule = "Règle 1.1 (Casse / Dommage visible)"
                summary = (
                    "Raisonnement multimodal IA (Règle 1.1) : Preuve visuelle de dommage confirmée par l'analyse d'image (ViT) "
                    "dans le délai imparti. Dossier éligible au remboursement ou remplacement."
                )
                actions = [
                    "Valider le remboursement intégral ou l'expédition d'un produit de remplacement sans frais",
                    "Émettre l'étiquette de retour prépayée si nécessaire",
                    "Clôturer le ticket avec le statut Remboursable"
                ]
                return status, applied_rule, summary, actions

        # 6. Application sémantique directe de la règle RAG sélectionnée
        if policy_match and policy_match.status_associated:
            try:
                status = TicketStatus(policy_match.status_associated)
            except ValueError:
                status = TicketStatus.REMBOURSABLE if policy_match.refund_eligible else TicketStatus.REFUSE

            applied_rule = f"{policy_match.title}" if policy_match.title else "Règle SmartHelp non identifiée"
            rule_prefix = f"[{policy_match.rule_code}] " if policy_match.rule_code else ""
            summary = (
                f"Application de la règle SmartHelp {rule_prefix}'{policy_match.title}'. "
                f"Diagnostic sémantique IA : {llm_reasoning}"
            )

            # Actions personnalisées selon le statut SmartHelp
            if status in [TicketStatus.REMBOURSABLE, TicketStatus.REMBOURSABLE_COLIS_PERDU]:
                actions = [
                    "Valider le remboursement intégral ou l'expédition d'un produit de remplacement sans frais",
                    "Émettre l'étiquette de retour prépayée si nécessaire",
                    "Clôturer le ticket avec le statut Remboursable"
                ]
            elif status == TicketStatus.ECHANGE_GRATUIT:
                actions = [
                    "Générer une étiquette de retour prépayée pour le mauvais article reçu",
                    "Déclencher l'expédition du bon modèle/couleur/taille sous 24h",
                    "Informer le client de la prise en charge gratuite de l'échange"
                ]
            elif status == TicketStatus.EXPEDITION_PIECE:
                actions = [
                    "Identifier la référence exacte de la pièce ou de l'accessoire manquant",
                    "Programmer l'expédition de la pièce manquante sous 3 jours ouvrés",
                    "Notifier le client du numéro de suivi du colis complémentaire"
                ]
            elif status == TicketStatus.DEDOMMAGEMENT_10:
                actions = [
                    "Générer un code promo / bon d'achat de 10% valable sur la prochaine commande",
                    "Envoyer le bon d'achat par email au client pour compenser le retard majeur de livraison",
                    "Clôturer le ticket avec dédommagement"
                ]
            elif status == TicketStatus.NON_REMBOURSABLE_RETARD_MINEUR:
                actions = [
                    "Notifier le client de la livraison imminente selon les informations du transporteur",
                    "Expliquer poliment qu'un retard <= 3 jours ouvrés ne donne pas lieu à compensation financière (Règle 3.1)"
                ]
            elif status == TicketStatus.REFUSE:
                actions = [
                    "Notifier le client du refus selon les termes de la politique SmartHelp",
                    "Fournir un lien vers la documentation interne/CGV"
                ]
            elif status == TicketStatus.A_VERIFIER:
                actions = [
                    "Transmettre le dossier au manager pour validation manuelle",
                    "Vérifier la date exacte d'achat et de réclamation (contrôle du délai de 48h)"
                ]
            else:
                actions = [
                    "Vérifier les pièces du dossier et traiter selon les préconisations SmartHelp"
                ]

            return status, applied_rule, summary, actions

        # 7. Fallback si aucune règle n'est déduite
        status = TicketStatus.A_VERIFIER
        applied_rule = "Règle indéterminée / Diagnostic mixte"
        summary = f"Diagnostic mixte IA nécessitant une validation manuelle. {llm_reasoning}"
        actions = [
            "Vérifier manuellement les pièces jointes et réclamation du client",
            "Consulter la politique SmartHelp et ajuster le statut"
        ]
        return status, applied_rule, summary, actions
