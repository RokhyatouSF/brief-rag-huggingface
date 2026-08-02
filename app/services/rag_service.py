import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache
import numpy as np

from app.config import settings
from app.schemas.ticket import PolicyMatch

logger = logging.getLogger(__name__)


class RAGKnowledgeService:
    """
    Service RAG (Retrieval-Augmented Generation) pour rechercher les articles des CGV,
    la politique de retour et les FAQ les plus pertinents par rapport à la réclamation.
    """
    _instance: Optional["RAGKnowledgeService"] = None

    def __init__(self):
        self.embedding_model = None
        self.documents: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.initialized = False
        self._load_knowledge_base()
        self._init_embedding_model()

    @classmethod
    def get_instance(cls) -> "RAGKnowledgeService":
        if cls._instance is None:
            cls._instance = RAGKnowledgeService()
        return cls._instance

    def _load_knowledge_base(self):
        kb_file = settings.KNOWLEDGE_BASE_DIR / "cgv_retours.json"
        if not kb_file.exists():
            logger.error(f"Fichier de base de connaissances non trouvé: {kb_file}")
            self.documents = []
            return

        try:
            with open(kb_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Base de connaissances RAG chargée avec {len(self.documents)} articles CGV/FAQ.")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la base RAG: {e}")
            self.documents = []

    def _init_embedding_model(self):
        try:
            logger.info(f"Chargement du modèle d'embedding sémantique ({settings.EMBEDDING_MODEL_ID})...")
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID, device=settings.DEVICE)
            
            # Pré-calcul des embeddings des documents de la base de connaissances
            if self.documents:
                texts = [f"{doc['title']} - {doc['category']}: {doc['content']}" for doc in self.documents]
                self.doc_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
                logger.info("Embeddings sémantiques pré-calculés et indexés en mémoire.")
            
            self.initialized = True
        except Exception as e:
            logger.warning(f"Impossible de charger SentenceTransformers ({e}). Activation du mode de secours TF-IDF / Mots-clés pour RAG.")
            self.embedding_model = None
            self.initialized = False

    def search_policy(self, query_text: str) -> Optional[PolicyMatch]:
        """
        Effectue une recherche sémantique RAG à partir du texte de la réclamation.
        Retourne la règle CGV la plus pertinente.
        """
        if not query_text or not self.documents:
            return None

        clean_query = query_text.strip().lower()

        # RAG Vectoriel avec SentenceTransformers si disponible
        if self.initialized and self.embedding_model is not None and self.doc_embeddings is not None:
            try:
                query_vec = self.embedding_model.encode([clean_query], normalize_embeddings=True)[0]
                # Calcul des similarités cosinus (dot product sur vecteurs normalisés)
                similarities = np.dot(self.doc_embeddings, query_vec)
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                best_doc = self.documents[best_idx]

                return PolicyMatch(
                    article_id=best_doc["article_id"],
                    title=best_doc["title"],
                    category=best_doc["category"],
                    excerpt=best_doc["content"],
                    similarity_score=round(max(0.0, min(1.0, best_score)), 4),
                    refund_eligible=best_doc.get("refund_eligible", True)
                )
            except Exception as e:
                logger.error(f"Erreur de recherche vectorielle RAG: {e}")
                return self._keyword_search_fallback(clean_query)
        else:
            return self._keyword_search_fallback(clean_query)

    def _keyword_search_fallback(self, query: str) -> Optional[PolicyMatch]:
        """Méthode de recherche de secours par chevauchement de mots-clés sémantiques."""
        best_doc = None
        highest_matches = -1

        for doc in self.documents:
            score = 0
            keywords = doc.get("keywords", [])
            for kw in keywords:
                if kw.lower() in query:
                    score += 2
            # Bonus si des mots du titre apparaissent
            for word in doc["title"].lower().split():
                if len(word) > 3 and word in query:
                    score += 1

            if score > highest_matches:
                highest_matches = score
                best_doc = doc

        if best_doc is None and self.documents:
            best_doc = self.documents[0]

        calc_score = round(min(0.95, 0.40 + (highest_matches * 0.15)), 2)

        return PolicyMatch(
            article_id=best_doc["article_id"],
            title=best_doc["title"],
            category=best_doc["category"],
            excerpt=best_doc["content"],
            similarity_score=calc_score,
            refund_eligible=best_doc.get("refund_eligible", True)
        )


@lru_cache(maxsize=1)
def get_rag_service() -> RAGKnowledgeService:
    """Accès thread-safe et optimisé au Singleton du service RAG."""
    return RAGKnowledgeService.get_instance()
