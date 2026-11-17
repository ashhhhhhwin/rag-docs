from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import SUPPORTED, load_file
from app.retrieval.embeddings import Embedder
from app.retrieval.store import ChromaStore


def ingest_file(path: Path, store: ChromaStore, embedder: Embedder, size: int, overlap: int) -> int:
    """Index one file. Re-ingesting replaces the old chunks so edits leave no stale data."""
    chunks = chunk_pages(load_file(path), size, overlap)
    store.delete_source(path.name)
    if not chunks:
        return 0
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return len(chunks)


def ingest_dir(path: Path, store: ChromaStore, embedder: Embedder, size: int, overlap: int):
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    for f in files:
        if f.suffix.lower() in SUPPORTED:
            yield f.name, ingest_file(f, store, embedder, size, overlap)
