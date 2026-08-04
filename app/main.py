import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes import health_router, ticket_router
from app.services import get_audio_service, get_vision_service, get_rag_service

# Configuration des logs
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI (Lifespan).
    Préchauffe et charge les modèles IA en mémoire au démarrage (Singleton).
    """
    logger.info("=== Démarrage de l'API Support Ticket AI ===")
    logger.info("Pré-chargement des modèles IA en mémoire (Singleton / LRU Cache)...")
    
    # Initialisation unique des Singletons
    try:
        _ = get_audio_service()
        _ = get_vision_service()
        _ = get_rag_service()
        logger.info("Tous les modèles (Whisper ASR, ViT Vision, RAG SentenceTransformers) sont préchauffés en mémoire.")
    except Exception as e:
        logger.error(f"Avertissement lors du préchargement des modèles: {e}")

    yield

    logger.info("=== Arrêt de l'API Support Ticket AI ===")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## API de Qualification & Ingestion Automatisée de Tickets Support E-Commerce

    Cette API résout la surcharge du service client e-commerce en analysant automatiquement:
    - **Notes vocales client (ASR)** via Hugging Face **Whisper**
    - **Photos de preuves (Vision)** via Vision Transformer **ViT**
    - **Règles de gestion & CGV (RAG)** via **SentenceTransformers** & recherche vectorielle

    ---
    ### Endpoint principal:
    - `POST /support-ticket` : Ingestion multimodale (`multipart/form-data`) retournant un diagnostic JSON structuré et un statut recommandé (`Remboursable`, `À vérifier`, `Refusé`).
    
    ### Health check:
    - `GET /health` : État des modèles pré-chargés en mémoire.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Activation CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire global des exceptions non rattrapées."""
    logger.error(f"Exception globale capturée: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Une erreur interne s'est produite lors du traitement du ticket support.",
            "details": str(exc) if settings.DEBUG else None
        }
    )


# Inclusion des routeurs
app.include_router(health_router)
app.include_router(ticket_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
