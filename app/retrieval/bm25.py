import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import Chunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, n: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:n] if s > 0]