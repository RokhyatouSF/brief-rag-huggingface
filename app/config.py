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
    WHISPER_REVISION: str = os.getenv("WHISPER_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d")
    
    VIT_MODEL_ID: str = os.getenv("VIT_MODEL_ID", "openai/clip-vit-base-patch32")
    VIT_REVISION: str = os.getenv("VIT_REVISION", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_REVISION: str = os.getenv("EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    
    LLM_REASONING_MODEL_ID: str = os.getenv("LLM_REASONING_MODEL_ID", "typeform/distilbert-base-uncased-mnli")
    
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
