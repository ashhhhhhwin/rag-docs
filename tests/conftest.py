import math
import uuid
import zlib

import chromadb
import pytest

from app.retrieval.bm25 import tokenize
from app.retrieval.store import ChromaStore


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. Keeps tests offline and fast."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def store(chroma_client):
    # unique collection per test, since the ephemeral client is shared
    return ChromaStore(chroma_client, f"test_{uuid.uuid4().hex}")