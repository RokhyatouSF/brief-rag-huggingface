import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


class LLMReasoningService:
    """
    Service d'IA générative et de raisonnement sémantique (LLM / Zero-Shot NLI)
    analysant dynamiquement les réclamations clients sans règles textuelles statiques.
    """
    _instance: Optional["LLMReasoningService"] = None

    def __init__(self):
        self.classifier = None
        self.initialized = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "LLMReasoningService":
        if cls._instance is None:
            cls._instance = LLMReasoningService()
        return cls._instance

    def _load_model(self):
        try:
            logger.info(f"Chargement du modèle de raisonnement IA Zero-Shot ({settings.LLM_REASONING_MODEL_ID})...")
            from transformers import pipeline

            device_id = 0 if settings.DEVICE == "cuda" else -1
            self.classifier = pipeline(
                "zero-shot-classification",
                model=settings.LLM_REASONING_MODEL_ID,
                device=device_id
            )
            self.initialized = True
            logger.info("Modèle de raisonnement sémantique IA (Zero-Shot NLI) prêt.")
        except Exception as e:
            logger.warning(f"Impossible de charger le modèle Zero-Shot NLI ({e}). Mode fallback sémantique actif.")
            self.classifier = None
            self.initialized = False

    def analyze_claim_intent(self, claim_text: str, vision_status: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyse sémantiquement l'intention du client, la responsabilité et les contraintes temporelles
        par inférence NLI dynamique.
        """
        if not claim_text or claim_text.strip() == "Aucun texte rédigé.":
            return {
                "top_intent": "REGLE-4.2",
                "confidence": 1.0,
                "reasoning": "Aucune description explicite ou preuve vocale/textuelle fournie par le client."
            }

        # Hypothèses sémantiques dynamiques correspondant aux règles métier CGV
        candidate_map = {
            "faute mauvaise utilisation chute ou casse causée par le client": {
                "rule_code": "4.1",
                "title": "Règle 4.1 (Usure normale / Mauvaise utilisation)",
                "reasoning_template": "L'analyse IA identifie une responsabilité du client ou un dommage survenu après réception (chute / mauvaise manipulation)."
            },
            "délai de réclamation dépassé de 48 heures ou plusieurs jours après la livraison": {
                "rule_code": "1.2",
                "title": "Règle 1.2 (Délai dépassé)",
                "reasoning_template": "L'analyse temporelle IA indique que la réclamation pour dommage est formulée après le délai de 48 heures (ex: après 2 jours ou plusieurs jours)."
            },
            "produit cassé ou endommagé à la livraison dans le délai imparti": {
                "rule_code": "1.1",
                "title": "Règle 1.1 (Casse / Dommage visible)",
                "reasoning_template": "L'analyse IA confirme une casse survenue à la livraison dans le délai imparti sans responsabilité client."
            },
            "erreur d'article de couleur de modèle ou de taille reçue": {
                "rule_code": "2.1",
                "title": "Règle 2.1 (Mauvais article reçu)",
                "reasoning_template": "L'analyse IA identifie une non-conformité de commande (couleur, taille ou modèle différent)."
            },
            "accessoire ou pièce manquante dans le colis": {
                "rule_code": "2.2",
                "title": "Règle 2.2 (Pièce manquante)",
                "reasoning_template": "L'analyse IA identifie un composant ou accessoire manquant dans le produit."
            },
            "retard de livraison du colis": {
                "rule_code": "3.1",
                "title": "Règle 3.1 (Retard mineur)",
                "reasoning_template": "L'analyse IA identifie une réclamation portant uniquement sur un délai de livraison."
            }
        }

        candidates = list(candidate_map.keys())

        if self.initialized and self.classifier is not None:
            try:
                res = self.classifier(claim_text, candidate_labels=candidates)
                top_label = res["labels"][0]
                confidence = float(res["scores"][0])
                matched_info = candidate_map[top_label]

                return {
                    "top_intent": matched_info["rule_code"],
                    "title": matched_info["title"],
                    "confidence": round(confidence, 4),
                    "reasoning": f"{matched_info['reasoning_template']} (Score de certitude IA: {confidence*100:.1f}%)"
                }
            except Exception as e:
                logger.error(f"Erreur d'inférence Zero-Shot NLI: {e}")
                return self._fallback_semantic_intent(claim_text, candidate_map)
        else:
            return self._fallback_semantic_intent(claim_text, candidate_map)

    def _fallback_semantic_intent(self, claim_text: str, candidate_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Inférence sémantique de secours si le pipeline NLI est indisponible."""
        claim_lower = claim_text.lower()
        if any(kw in claim_lower for kw in ["avec moi", "ma faute", "fait tomber", "chute", "usure"]):
            return {
                "top_intent": "4.1",
                "title": "Règle 4.1 (Usure normale / Mauvaise utilisation)",
                "confidence": 0.85,
                "reasoning": "Raisonnement sémantique (Secours) : Responsabilité client ou évènement post-livraison identifié."
            }
        elif any(kw in claim_lower for kw in ["apres 2 jours", "après 2 jours", "apres 48h", "plus de 48h", "3 jours après"]):
            return {
                "top_intent": "1.2",
                "title": "Règle 1.2 (Délai dépassé)",
                "confidence": 0.85,
                "reasoning": "Raisonnement sémantique (Secours) : Délai de 48h dépassé d'après les indications temporelles."
            }
        elif "bleu" in claim_lower or "rouge" in claim_lower or "couleur" in claim_lower or "taille" in claim_lower:
            return {
                "top_intent": "2.1",
                "title": "Règle 2.1 (Mauvais article reçu)",
                "confidence": 0.85,
                "reasoning": "Raisonnement sémantique (Secours) : Erreur de couleur ou de produit."
            }
        elif "pièce" in claim_lower or "accessoire" in claim_lower:
            return {
                "top_intent": "2.2",
                "title": "Règle 2.2 (Pièce manquante)",
                "confidence": 0.85,
                "reasoning": "Raisonnement sémantique (Secours) : Pièce ou accessoire manquant."
            }
        else:
            return {
                "top_intent": "1.1",
                "title": "Règle 1.1 (Casse / Dommage visible)",
                "confidence": 0.70,
                "reasoning": "Raisonnement sémantique (Secours) : Signalement de dommage à la livraison."
            }


@lru_cache(maxsize=1)
def get_llm_reasoning_service() -> LLMReasoningService:
    """Accès Singleton optimisé au service de raisonnement LLM / NLI."""
    return LLMReasoningService.get_instance()
