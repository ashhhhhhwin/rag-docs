from dataclasses import dataclass

from app.ingestion.models import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float                  # final ranking score (cosine, or RRF when hybrid)
    vector_score: float | None    # cosine similarity; None if only BM25 found it