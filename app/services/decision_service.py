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
    et les règles sémantiques RAG (SmartHelp) pour recommander le statut du ticket et les actions associées.
    """

    @staticmethod
    def evaluate_ticket(
        customer_claim_text: str,
        audio_result: AudioAnalysisResult,
        vision_result: VisionAnalysisResult,
        policy_match: Optional[PolicyMatch]
    ) -> Tuple[TicketStatus, Optional[str], str, List[str]]:
        """
        Synthétise les analyses multimodales pour déterminer le statut préconisé.
        Retourne (status, applied_rule, summary_explanation, recommended_actions).
        """
        claim_lower = customer_claim_text.lower()
        has_audio = audio_result.processed and bool(audio_result.transcription)
        has_vision = vision_result.processed
        is_damaged_image = has_vision and (vision_result.condition_status == "Produit endommagé / cassé")
        is_intact_image = has_vision and (vision_result.condition_status == "Produit conforme / intact")

        actions = []

        client_fault_keywords = [
            "avec moi", "de ma faute", "ma faute", "fait tomber", "chute", 
            "usure", "mauvaise manipulation", "j'ai cassé", "je l'ai cassé", 
            "cassé par moi", "mal manipulé", "tombé par terre"
        ]
        is_client_fault = any(kw in claim_lower for kw in client_fault_keywords)

        time_delay_exceeded_keywords = [
            "apres 2 jours", "après 2 jours", "apres 48h", "après 48h", 
            "plus de 48h", "plus de 48 heures", "3 jours après", "3 jours apres", 
            "délai dépassé", "apres 48 heures", "après 48 heures", "quelques jours après",
            "2 jours après", "2 jours apres"
        ]
        is_delay_exceeded = any(kw in claim_lower for kw in time_delay_exceeded_keywords)

        # 1. Contradiction majeure ou mauvaise utilisation / faute du client (Règle 4.1)
        if is_client_fault or (is_intact_image and any(kw in claim_lower for kw in ["cassé", "endommagé", "fissure", "abîmé"])) or (policy_match and policy_match.rule_code == "4.1"):
            status = TicketStatus.REFUSE
            applied_rule = "Règle 4.1 (Usure normale / Mauvaise utilisation)"
            summary = (
                "Refus au titre de la Règle 4.1 : La réclamation indique que le dommage résulte d'un évènement survenu après réception "
                "ou d'une manipulation/responsabilité du client."
                if is_client_fault else
                "Contradiction majeure détectée: la note vocale ou le texte signale un produit cassé, "
                "mais l'analyse visuelle par ViT confirme que l'article sur la photo est intact et conforme (Règle 4.1)."
            )
            actions = [
                "Notifier le client du refus selon la Règle 4.1 (responsabilité client / usure)",
                "Fournir un lien vers la documentation CGV / politique de retour",
                "Transmettre au niveau 2 si le client conteste l'analyse"
            ]
            return status, applied_rule, summary, actions

        # 2. Casse / Dommage signalé après le délai réglementaire de 48h (Règle 1.2)
        if is_delay_exceeded or (policy_match and policy_match.rule_code == "1.2"):
            status = TicketStatus.A_VERIFIER
            applied_rule = "Règle 1.2 (Délai dépassé)"
            summary = (
                "Règle 1.2 (Délai dépassé) : La réclamation pour produit cassé/endommagé mentionne un délai supérieur à 48 heures "
                "suivant la livraison (ex: 'après 2 jours'). Le dossier nécessite une validation manuelle du manager."
            )
            actions = [
                "Transmettre le dossier au manager pour validation manuelle de dérogation",
                "Vérifier la date exacte d'achat et la date de livraison effective auprès du transporteur"
            ]
            return status, applied_rule, summary, actions

        # 3. Absence de preuve ou justificatifs (Règle 4.2)
        if not has_vision and not has_audio and (not customer_claim_text or len(customer_claim_text.strip()) < 10 or customer_claim_text == "Aucun texte rédigé."):
            status = TicketStatus.EN_ATTENTE_JUSTIFICATIFS
            applied_rule = "Règle 4.2 (Absence de preuve)"
            summary = "Absence de justificatifs probants: ni photo ni note vocale/description explicite fournie (Règle 4.2)."
            actions = [
                "Relancer le client pour obtenir une note vocale ou une photo claire du produit",
                "Mettre le ticket en attente de pièces complémentaires"
            ]
            return status, applied_rule, summary, actions

        # 4. Preuve visuelle directe de dommage / casse à la livraison dans les délais (Règle 1.1)
        if is_damaged_image:
            # Si le RAG n'a pas sélectionné une règle d'exception spécifique (ex: Règle 1.2 Délai dépassé ou Règle 4.1 Usure)
            if not policy_match or policy_match.rule_code not in ["1.2", "4.1"]:
                status = TicketStatus.REMBOURSABLE
                applied_rule = "Règle 1.1 (Casse / Dommage visible)"
                summary = (
                    "Preuve visuelle de dommage ou fissure confirmée par l'analyse d'image (ViT) dans le délai de 48h. "
                    "Application de la Règle 1.1 (Casse / Dommage visible) : le dossier est éligible au remboursement ou remplacement."
                )
                actions = [
                    "Valider le remboursement intégral ou l'expédition d'un produit de remplacement sans frais",
                    "Émettre l'étiquette de retour prépayée si nécessaire",
                    "Clôturer le ticket avec le statut Remboursable"
                ]
                return status, applied_rule, summary, actions

        # 3. Application directe du statut recommandé par la règle SmartHelp sélectionnée par le RAG
        if policy_match and policy_match.status_associated:
            try:
                status = TicketStatus(policy_match.status_associated)
            except ValueError:
                status = TicketStatus.REMBOURSABLE if policy_match.refund_eligible else TicketStatus.REFUSE

            applied_rule = f"{policy_match.title}" if policy_match.title else "Règle SmartHelp non identifiée"
            rule_prefix = f"[{policy_match.rule_code}] " if policy_match.rule_code else ""
            summary = (
                f"Application explicite de la règle SmartHelp {rule_prefix}'{policy_match.title}' "
                f"(Catégorie: {policy_match.category}). Règle officielle: {policy_match.explicit_rule or policy_match.excerpt}"
            )

            # Actions personnalisées selon le statut SmartHelp
            if status == TicketStatus.REMBOURSABLE or status == TicketStatus.REMBOURSABLE_COLIS_PERDU:
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

        # 4. Fallback général si aucune règle n'est sélectionnée
        status = TicketStatus.A_VERIFIER
        applied_rule = "Règle indéterminée / Diagnostic mixte"
        summary = "Diagnostic mixte nécessitant une validation manuelle par un conseiller support."
        actions = [
            "Vérifier manuellement les pièces jointes et réclamation du client",
            "Consulter la politique SmartHelp et ajuster le statut"
        ]
        return status, applied_rule, summary, actions
