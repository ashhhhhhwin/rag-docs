from functools import lru_cache

from app.config import get_settings
from app.generation.client import OpenRouterClient
from app.generation.service import GenerationService
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import make_store


@lru_cache
def get_store():
    return make_store()


@lru_cache
def get_generation_service() -> GenerationService:
    return GenerationService(OpenRouterClient())


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()

    return Retriever(
        store=get_store(),
        embedder=get_embedder(),
        hybrid=settings.hybrid_enabled,
        rrf_k=settings.rrf_k,
    )
