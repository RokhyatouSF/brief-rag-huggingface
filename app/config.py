import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Informations Générales sur l'API
    APP_NAME: str = "E-Commerce Support Ticket AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Configuration et Pinning des Modèles Hugging Face
    # Fixation explicite des identifiants et des révisions pour la stabilité des builds
    WHISPER_MODEL_ID: str = os.getenv("WHISPER_MODEL_ID", "openai/whisper-small")
    WHISPER_REVISION: str = os.getenv("WHISPER_REVISION", "main")
    
    VIT_MODEL_ID: str = os.getenv("VIT_MODEL_ID", "google/vit-base-patch16-224")
    VIT_REVISION: str = os.getenv("VIT_REVISION", "main")
    
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_REVISION: str = os.getenv("EMBEDDING_REVISION", "main")
    
    # Dispositif matériel (cpu ou cuda)
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # Chemins des données
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    KNOWLEDGE_BASE_DIR: Path = BASE_DIR / "app" / "data" / "knowledge_base"
    
    # Seuils de décision RAG & Vision
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    VISION_CONFIDENCE_THRESHOLD: float = 0.50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
