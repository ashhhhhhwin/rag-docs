from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (OpenRouter)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"  # verify on openrouter.ai/models
    llm_timeout_s: float = 60.0

    # Embeddings (local)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    chunk_strategy: str = "recursive"  # recursive | llm_context
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5
    hybrid_enabled: bool = True
    rerank_enabled: bool = False
    min_relevance_score: float = 0.0  # retrieval gate, tuned later with evals

    # Storage
    chroma_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # Guardrails
    max_question_chars: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()