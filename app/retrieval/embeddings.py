from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed. No API key, no rate limits."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # lazy: model load is slow

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embedder() -> Embedder:
    return FastEmbedder(get_settings().embedding_model)