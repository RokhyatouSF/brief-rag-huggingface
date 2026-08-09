# Modalités d'Exécution du Projet & Gestion de Projet (Kanban & Git Flow)

---

## 1. Organisation du Projet

- **Format d'exécution** : **Individuel**
- **Développeur Lead / Backend & IA** : Étudiant / Développeur unique
- **Lien du Tableau de Bord Kanban (Trello / GitHub Projects)** :
  
  - **Suivi des Tâches** : Backlog, In Progress, Code Review, Done.

---

## 2. Répartition Individuelle des Tâches

| Module / Domaine | Description des Tâches Réalisées en Individuel | Statut |
| :--- | :--- | :---: |
| **Architecture & API Core** | Conception de l'architecture modulaire FastAPI, configuration Pydantic Settings, middleware CORS et gestionnaire global d'exceptions. | Terminé |
| **Service Audio (ASR)** | Implémentation du pattern Singleton / `@lru_cache` pour le chargement unique du modèle Whisper (`openai/whisper-small`), gestion des flux binaires audio (`.mp3`, `.wav`) et fallback. | Terminé |
| **Service Vision (ViT)** | Intégration du modèle ViT (`google/vit-base-patch16-224`) sous forme de Singleton pour la détection de produits cassés/conformes. | Terminé |
| **Service RAG & Knowledge Base** | Structuration de la base de connaissances CGV/FAQ en JSON, vectorisation avec `SentenceTransformers` (`all-MiniLM-L6-v2`) et recherche sémantique par similarité cosinus. | Terminé |
| **Moteur de Décision** | Algorithme de synthèse multimodale attribuant automatiquement le statut du ticket (`Remboursable`, `À vérifier`, `Refusé`) avec préconisations agents. | Terminé |
| **Tests & Démo** | Création de la suite de tests `pytest` (`test_health.py`, `test_ticket.py`) et du script de génération des assets de démonstration Swagger (`generate_test_assets.py`). | Terminé |
| **Documentation Millimétrique** | Rédaction du guide complet `README.md` avec diagrammes Mermaid, règles d'optimisation mémoire et guide Swagger. | Terminé |

---

## 3. Application Stricte de la Stratégie Git Flow

Le dépôt respecte la stratégie **Git Flow** :

### Branches principales :
- `main` : Code de production stable et validé.
- `develop` : Branche d'intégration continue des fonctionnalités.

### Branches de fonctionnalités (`feature/...`) :
- `feature/setup-architecture` : Initialisation du projet, FastAPI, Pydantic et structure globale.
- `feature/audio-whisper-asr` : Service ASR Whisper Singleton.
- `feature/vision-vit` : Service Vision ViT.
- `feature/rag-knowledge-base` : Vector Store et recherche sémantique CGV.
- `feature/decision-engine` : Synthèse multimodale et préconisation de statut.
- `feature/tests-and-documentation` : Suite de tests pytest et documentation millimétrique.

### Convention de Commits :
```text
feat(audio): implémentation du singleton Whisper ASR avec lru_cache
feat(vision): ajout du classifieur ViT pour l'analyse de conformité produit
feat(rag): indexation de la base de connaissances CGV et recherche vectorielle
feat(decision): intégration du moteur de règles et statuts de ticket
test(api): ajout des tests d'intégration pytest pour /support-ticket
docs(readme): ajout des diagrammes d'architecture et du guide d'utilisation Swagger
```
