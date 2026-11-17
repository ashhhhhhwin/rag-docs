import chromadb

from app.config import get_settings
from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedChunk

NO_PAGE = -1  # Chroma metadata can't hold None


def _to_chunk(id_: str, text: str, meta: dict) -> Chunk:
    page = meta["page"] if meta["page"] != NO_PAGE else None
    return Chunk(id_, meta["source"], page, meta["index"], text)


class ChromaStore:
    def __init__(self, client, collection_name: str):
        self._col = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._col.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page if c.page is not None else NO_PAGE,
                    "index": c.index,
                }
                for c in chunks
            ],
        )

    def query(self, vector: list[float], n: int) -> list[RetrievedChunk]:
        n = min(n, self.count())
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            sim = 1.0 - dist  # Chroma returns cosine distance
            hits.append(RetrievedChunk(_to_chunk(id_, doc, meta), sim, sim))
        return hits

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        return [
            _to_chunk(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.all_chunks():
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})


def make_store() -> ChromaStore:
    s = get_settings()
    return ChromaStore(chromadb.PersistentClient(path=s.chroma_dir), s.collection_name)
