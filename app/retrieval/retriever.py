from app.ingestion.models import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from app.retrieval.store import ChromaStore


class Retriever:
    def __init__(
        self,
        store: ChromaStore,
        embedder: Embedder,
        hybrid: bool = True,
        rrf_k: int = 60,
        pool_multiplier: int = 4,
    ):
        self._store = store
        self._embedder = embedder
        self._hybrid = hybrid
        self._rrf_k = rrf_k
        self._pool = pool_multiplier
        self._bm25: BM25Index | None = None
        self._bm25_size = -1

    def _bm25_index(self) -> BM25Index:
        # Rebuild lazily whenever the number of stored chunks changes
        n = self._store.count()
        if self._bm25 is None or n != self._bm25_size:
            self._bm25 = BM25Index(self._store.all_chunks())
            self._bm25_size = n
        return self._bm25

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        pool = top_k * self._pool
        vector_hits = self._store.query(self._embedder.embed_query(query), pool)
        if not self._hybrid:
            return vector_hits[:top_k]

        bm25_hits = self._bm25_index().search(query, pool)
        fused = reciprocal_rank_fusion(
            [[h.chunk.id for h in vector_hits], [c.id for c, _ in bm25_hits]], self._rrf_k
        )
        vector_by_id = {h.chunk.id: h for h in vector_hits}
        bm25_by_id: dict[str, Chunk] = {c.id: c for c, _ in bm25_hits}

        results = []
        for id_ in sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]:
            hit = vector_by_id.get(id_)
            chunk = hit.chunk if hit else bm25_by_id[id_]
            results.append(RetrievedChunk(chunk, fused[id_], hit.vector_score if hit else None))
        return results