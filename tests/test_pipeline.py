from app.ingestion.pipeline import ingest_file


def test_ingest_is_idempotent_and_replaces_edits(tmp_path, store, embedder):
    f = tmp_path / "note.txt"
    f.write_text("Alpha beta gamma. " * 80, encoding="utf-8")
    n1 = ingest_file(f, store, embedder, size=300, overlap=30)
    n2 = ingest_file(f, store, embedder, size=300, overlap=30)
    assert n1 > 1 and n1 == n2 == store.count()

    f.write_text("Completely different content.", encoding="utf-8")
    ingest_file(f, store, embedder, size=300, overlap=30)
    assert store.count() == 1
